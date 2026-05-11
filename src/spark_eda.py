"""Standalone Spark EDA + Hive warehouse-build script.

This script does the heavy lifting that Sections 1-5 of the main notebook do,
but headlessly so it can be wired into Airflow / a CI job. It:

1. Loads bank.csv into a Spark DataFrame.
2. Cleans + engineers features (see src/utils.clean_and_engineer).
3. Writes a partitioned Parquet warehouse (HDFS-style) to warehouse/bank_parquet.
4. Registers a Hive external table over the warehouse and prints summary queries.
5. Saves headline EDA aggregations as CSVs in reports/.

Usage
-----
    python src/spark_eda.py
    python src/spark_eda.py --data data/bank.csv --warehouse warehouse/bank_parquet
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pyspark.sql import functions as F

# Allow `from utils import ...` whether you invoke from project root or from src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    build_spark,
    clean_and_engineer,
    load_bank_csv,
    resolve_data_path,
)


def write_warehouse(df, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    (df.write.mode("overwrite").partitionBy("month").parquet(str(path)))
    print(f"Wrote partitioned Parquet warehouse -> {path}")


def register_hive_table(spark, warehouse_path: Path) -> None:
    spark.sql("CREATE DATABASE IF NOT EXISTS bank_warehouse")
    spark.sql("USE bank_warehouse")
    spark.sql("DROP TABLE IF EXISTS bank_clients")
    spark.sql(
        f"""
        CREATE EXTERNAL TABLE bank_clients (
            age INT, job STRING, marital STRING, education STRING,
            `default` INT, balance INT, housing INT, loan INT,
            contact STRING, day INT, duration INT, campaign INT,
            pdays INT, previous INT, poutcome STRING, y INT,
            never_contacted_before INT,
            age_bucket STRING, balance_bucket STRING, duration_min DOUBLE
        )
        PARTITIONED BY (month STRING)
        STORED AS PARQUET
        LOCATION '{warehouse_path.resolve().as_uri()}'
        """
    )
    spark.sql("MSCK REPAIR TABLE bank_clients")
    print("Hive table 'bank_warehouse.bank_clients' registered.")


def run_summary_queries(spark, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "subscription_rate_by_job": """
            SELECT job, COUNT(*) AS n_clients, SUM(y) AS n_subscribed,
                   ROUND(100.0 * AVG(y), 2) AS subscription_rate_pct
            FROM bank_clients WHERE job IS NOT NULL
            GROUP BY job ORDER BY subscription_rate_pct DESC
        """,
        "subscription_rate_by_month": """
            SELECT month, COUNT(*) AS n_calls,
                   ROUND(AVG(duration), 1) AS avg_duration_sec,
                   ROUND(100.0 * AVG(y), 2) AS subscription_rate_pct
            FROM bank_clients GROUP BY month
            ORDER BY subscription_rate_pct DESC
        """,
        "subscription_rate_by_balance_bucket": """
            SELECT balance_bucket, COUNT(*) AS n_clients,
                   ROUND(AVG(balance), 0) AS avg_balance,
                   ROUND(100.0 * AVG(y), 2) AS subscription_rate_pct
            FROM bank_clients GROUP BY balance_bucket
            ORDER BY subscription_rate_pct DESC
        """,
        "subscription_rate_by_poutcome": """
            SELECT COALESCE(poutcome,'unknown') AS poutcome,
                   COUNT(*) AS n_clients,
                   ROUND(100.0 * AVG(y), 2) AS subscription_rate_pct
            FROM bank_clients GROUP BY poutcome
            ORDER BY subscription_rate_pct DESC
        """,
    }
    for name, sql in queries.items():
        df = spark.sql(sql)
        print(f"\n== {name} ==")
        df.show(truncate=False)
        df.toPandas().to_csv(out_dir / f"{name}.csv", index=False)
    print(f"\nSummary CSVs -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--data", default=None)
    p.add_argument("--warehouse", default="warehouse/bank_parquet", type=Path)
    p.add_argument("--reports", default="reports", type=Path)
    args = p.parse_args()

    spark = build_spark("BankSparkEDA")
    data_path = args.data or resolve_data_path()
    print(f"Loading bank data from: {data_path}")

    df = load_bank_csv(spark, data_path)
    df_clean = clean_and_engineer(df).cache()
    print(f"Cleaned rows: {df_clean.count():,}")

    write_warehouse(df_clean, args.warehouse)
    register_hive_table(spark, args.warehouse)
    run_summary_queries(spark, args.reports)

    spark.stop()


if __name__ == "__main__":
    main()
