"""Standalone Spark ML training script.

Trains the three classifiers from the main notebook (LR, RandomForest, GBT) on
a single shared Spark ML Pipeline, tunes each with 3-fold CrossValidator, and
persists the best model + a metrics report to disk.

Usage
-----
    python src/spark_ml_pipeline.py
    python src/spark_ml_pipeline.py --data data/bank.csv --out models/best_pipeline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CAT_COLS,
    NUM_COLS_MODEL,
    build_spark,
    clean_and_engineer,
    fill_categorical_nulls,
    load_bank_csv,
    resolve_data_path,
)


RANDOM_SEED = 42


def build_preprocessing_stages():
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in CAT_COLS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
                for c in CAT_COLS]
    feature_cols = NUM_COLS_MODEL + [f"{c}_ohe" for c in CAT_COLS]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=False, withStd=True)
    return indexers + encoders + [assembler, scaler]


def evaluate(predictions, label: str = "y") -> dict[str, float]:
    auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction",
                                           metricName="f1").evaluate(predictions)
    prec = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction",
                                             metricName="weightedPrecision").evaluate(predictions)
    rec = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction",
                                            metricName="weightedRecall").evaluate(predictions)
    return {"ROC-AUC": auc, "F1": f1, "Precision": prec, "Recall": rec}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="models/best_pipeline", type=Path)
    p.add_argument("--report", default="reports/ml_metrics.json", type=Path)
    args = p.parse_args()

    spark = build_spark("BankSparkML")
    data_path = args.data or resolve_data_path()
    print(f"Loading bank data from: {data_path}")
    df = clean_and_engineer(load_bank_csv(spark, data_path))
    df = fill_categorical_nulls(df).cache()
    print(f"Training rows total: {df.count():,}")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    pos_rate = train_df.selectExpr("avg(y)").collect()[0][0]
    print(f"Positive rate (train): {pos_rate:.4f}")

    train_weighted = train_df.withColumn(
        "weight", F.when(F.col("y") == 1, F.lit((1 - pos_rate) / pos_rate)).otherwise(F.lit(1.0))
    )
    test_weighted = test_df.withColumn("weight", F.lit(1.0))

    preproc = build_preprocessing_stages()
    auc_eval = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPrediction",
                                             metricName="areaUnderROC")

    # ----- Logistic Regression -----
    lr = LogisticRegression(labelCol="y", featuresCol="features",
                            weightCol="weight", maxIter=50)
    lr_pipe = Pipeline(stages=preproc + [lr])
    lr_grid = (ParamGridBuilder()
               .addGrid(lr.regParam, [0.0, 0.01, 0.1])
               .addGrid(lr.elasticNetParam, [0.0, 0.5])
               .build())
    lr_cv = CrossValidator(estimator=lr_pipe, estimatorParamMaps=lr_grid,
                           evaluator=auc_eval, numFolds=3, parallelism=2, seed=RANDOM_SEED)
    print("Fitting LR with 3-fold CV ...")
    lr_model = lr_cv.fit(train_weighted)
    lr_metrics = evaluate(lr_model.transform(test_weighted))
    print(f"  LR metrics: {lr_metrics}")

    # ----- Random Forest -----
    rf = RandomForestClassifier(labelCol="y", featuresCol="features",
                                numTrees=100, maxDepth=8, seed=RANDOM_SEED)
    rf_pipe = Pipeline(stages=preproc + [rf])
    rf_grid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [50, 150])
               .addGrid(rf.maxDepth, [6, 10])
               .build())
    rf_cv = CrossValidator(estimator=rf_pipe, estimatorParamMaps=rf_grid,
                           evaluator=auc_eval, numFolds=3, parallelism=2, seed=RANDOM_SEED)
    print("Fitting RF with 3-fold CV ...")
    rf_model = rf_cv.fit(train_df)
    rf_metrics = evaluate(rf_model.transform(test_df))
    print(f"  RF metrics: {rf_metrics}")

    # ----- Gradient-Boosted Trees -----
    gbt = GBTClassifier(labelCol="y", featuresCol="features", maxIter=80, maxDepth=5, seed=RANDOM_SEED)
    gbt_pipe = Pipeline(stages=preproc + [gbt])
    gbt_grid = (ParamGridBuilder()
                .addGrid(gbt.maxDepth, [4, 6])
                .addGrid(gbt.maxIter, [50, 100])
                .build())
    gbt_cv = CrossValidator(estimator=gbt_pipe, estimatorParamMaps=gbt_grid,
                            evaluator=auc_eval, numFolds=3, parallelism=2, seed=RANDOM_SEED)
    print("Fitting GBT with 3-fold CV ...")
    gbt_model = gbt_cv.fit(train_df)
    gbt_metrics = evaluate(gbt_model.transform(test_df))
    print(f"  GBT metrics: {gbt_metrics}")

    # ----- Pick the best by ROC-AUC, persist -----
    candidates = {
        "LogReg": (lr_model.bestModel, lr_metrics),
        "RandomForest": (rf_model.bestModel, rf_metrics),
        "GBT": (gbt_model.bestModel, gbt_metrics),
    }
    best_name = max(candidates, key=lambda k: candidates[k][1]["ROC-AUC"])
    best_model, best_metrics = candidates[best_name]
    print(f"\nBest model: {best_name}  (ROC-AUC = {best_metrics['ROC-AUC']:.4f})")

    if args.out.exists():
        shutil.rmtree(args.out)
    best_model.write().overwrite().save(str(args.out))
    print(f"Saved best pipeline to: {args.out}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {
            "best_model": best_name,
            "best_metrics": best_metrics,
            "all_metrics": {n: m for n, (_, m) in candidates.items()},
        },
        indent=2,
    ))
    print(f"Wrote metrics report to: {args.report}")

    spark.stop()


if __name__ == "__main__":
    main()
