"""Spark Structured-Streaming consumer for the banking-transaction demo.

Reads JSON micro-batches from a watched directory, applies a tumbling-window
aggregation per job category, and writes the result to console + an in-memory
sink for inspection. A naive amount threshold flags 'suspicious' transactions
that would feed downstream fraud workflows in a real system.

Usage
-----
    # Run for 60 seconds (default), reading from ./stream_input
    python src/streaming/streaming_consumer.py

    # Watch a custom directory, run forever (Ctrl-C to stop):
    python src/streaming/streaming_consumer.py --input my_stream/ --forever

    # Larger window and stricter threshold:
    python src/streaming/streaming_consumer.py --window 120 --threshold 10000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

TXN_SCHEMA = StructType(
    [
        StructField("txn_id", StringType()),
        StructField("account_id", StringType()),
        StructField("job", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("txn_time", TimestampType()),
    ]
)


def build_spark(app_name: str = "BankStreamingConsumer") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.streaming.schemaInference", "true")
        .getOrCreate()
    )


def run(
    input_dir: Path,
    window_seconds: int,
    threshold: float,
    checkpoint_dir: Path,
    run_seconds: int,
    forever: bool,
) -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    stream = (
        spark.readStream.schema(TXN_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .json(str(input_dir))
    )

    tagged = (
        stream.withColumn("suspicious", (F.col("amount") > threshold).cast("int"))
        .withWatermark("txn_time", "2 minutes")
    )

    windowed = (
        tagged.groupBy(F.window("txn_time", f"{window_seconds} seconds"), F.col("job"))
        .agg(
            F.count("*").alias("n_txn"),
            F.round(F.sum("amount"), 2).alias("total_amount"),
            F.sum("suspicious").alias("n_suspicious"),
        )
        .orderBy(F.col("window").desc(), F.col("total_amount").desc())
    )

    query = (
        windowed.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", False)
        .option("numRows", 30)
        .option("checkpointLocation", str(checkpoint_dir))
        .start()
    )

    print(f"Started streaming query. input_dir={input_dir}  window={window_seconds}s  threshold={threshold}")

    if forever:
        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            pass
    else:
        deadline = time.time() + run_seconds
        while time.time() < deadline and query.isActive:
            time.sleep(2)

    query.stop()
    spark.stop()
    print("Stream stopped cleanly.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--input", default="stream_input", type=Path, help="Watched directory")
    p.add_argument("--window", default=60, type=int, help="Tumbling-window size in seconds")
    p.add_argument("--threshold", default=5000.0, type=float, help="Suspicious-amount threshold (EUR)")
    p.add_argument("--checkpoint", default="spark-warehouse/_chk_stream_consumer", type=Path)
    p.add_argument("--run-seconds", default=60, type=int, help="Stop after N seconds (ignored with --forever)")
    p.add_argument("--forever", action="store_true", help="Run until interrupted")
    args = p.parse_args()

    args.input.mkdir(parents=True, exist_ok=True)
    args.checkpoint.mkdir(parents=True, exist_ok=True)

    run(
        input_dir=args.input,
        window_seconds=args.window,
        threshold=args.threshold,
        checkpoint_dir=args.checkpoint,
        run_seconds=args.run_seconds,
        forever=args.forever,
    )


if __name__ == "__main__":
    main()
