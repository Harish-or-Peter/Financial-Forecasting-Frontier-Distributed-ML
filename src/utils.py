"""Shared helpers for the Bank Distributed ML project.

Importable from notebooks and standalone scripts. Keeps the SparkSession setup,
data-path resolution, and the cleaning transformations in one place so the
notebook and the .py scripts cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, functions as F


# -- Canonical column groupings --------------------------------------------------
CAT_COLS = ["job", "marital", "education", "contact", "poutcome",
            "age_bucket", "balance_bucket"]

# duration is intentionally excluded from the feature set — it leaks the target.
NUM_COLS_MODEL = ["age", "balance", "day", "campaign", "pdays", "previous",
                  "default", "housing", "loan", "never_contacted_before"]

UNKNOWN_CAT_COLS = ["job", "education", "contact", "poutcome"]
BINARY_YESNO_COLS = ["default", "housing", "loan", "y"]


def build_spark(app_name: str = "BankDistributedML",
                shuffle_partitions: int = 8,
                hive: bool = True) -> SparkSession:
    """Return a configured SparkSession (with Hive support by default)."""
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", "spark-warehouse")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )
    if hive:
        builder = builder.enableHiveSupport()
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def resolve_data_path(candidates: Iterable[str] = ("data/bank.csv", "bank.csv", "../data/bank.csv")
                      ) -> str:
    """Pick the first existing path from the candidates list."""
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        f"bank.csv not found in any of: {list(candidates)} — pass --data <path>."
    )


def load_bank_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw CSV with header + schema inference."""
    return (
        spark.read.option("header", True).option("inferSchema", True).csv(path)
    )


def clean_and_engineer(df: DataFrame) -> DataFrame:
    """Apply the canonical wrangling + feature-engineering steps.

    1. Coerce 'unknown' to NULL in categorical columns.
    2. Split pdays into a binary 'never_contacted_before' flag + a non-negative count.
    3. Convert yes/no flags (default/housing/loan/y) to int.
    4. Add age_bucket / balance_bucket / duration_min.
    """
    out = df
    for c in UNKNOWN_CAT_COLS:
        out = out.withColumn(c, F.when(F.col(c) == "unknown", None).otherwise(F.col(c)))

    out = (
        out.withColumn("never_contacted_before", (F.col("pdays") == -1).cast("int"))
        .withColumn("pdays", F.when(F.col("pdays") == -1, 0).otherwise(F.col("pdays")))
    )

    for c in BINARY_YESNO_COLS:
        out = out.withColumn(c, (F.col(c) == "yes").cast("int"))

    out = (
        out.withColumn(
            "age_bucket",
            F.when(F.col("age") < 30, "young")
            .when(F.col("age") < 45, "mid")
            .when(F.col("age") < 60, "senior")
            .otherwise("retired"),
        )
        .withColumn(
            "balance_bucket",
            F.when(F.col("balance") < 0, "negative")
            .when(F.col("balance") < 500, "low")
            .when(F.col("balance") < 2000, "mid")
            .when(F.col("balance") < 10000, "high")
            .otherwise("vip"),
        )
        .withColumn("duration_min", F.round(F.col("duration") / 60, 2))
    )
    return out


def fill_categorical_nulls(df: DataFrame, cols: Iterable[str] = CAT_COLS,
                            token: str = "missing") -> DataFrame:
    """Replace NULLs in categorical columns with a constant token."""
    return df.fillna({c: token for c in cols})
