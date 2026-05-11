# Technical Document — Bank Marketing Distributed ML

**Project:** End-Course Capstone, Distributed Machine Learning module
**Author:** *Your Name Here*
**Date:** 2026-05-11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context & Problem Statement](#2-business-context--problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Dataset Description](#4-dataset-description)
5. [Part-by-Part Methodology](#5-part-by-part-methodology)
   - 5.1 [Part 1 — Hadoop + Hive](#51-part-1--hadoop--hive)
   - 5.2 [Part 2 — Spark EDA](#52-part-2--spark-eda)
   - 5.3 [Part 3 — Spark ML](#53-part-3--spark-ml)
   - 5.4 [Part 4 — Spark Streaming](#54-part-4--spark-streaming)
   - 5.5 [Part 5 — Data Parallelism](#55-part-5--data-parallelism)
6. [Predictive Modeling Deep-Dive](#6-predictive-modeling-deep-dive)
7. [Streaming Architecture & Window Operations](#7-streaming-architecture--window-operations)
8. [Parallelism & Performance Tuning](#8-parallelism--performance-tuning)
9. [Challenges & Trade-offs](#9-challenges--trade-offs)
10. [Future Work](#10-future-work)
11. [References](#11-references)

---

## 1. Executive Summary

The banking industry generates a torrent of structured and semi-structured data every second — customer demographics, account balances, marketing
contacts, transaction streams. Single-machine workflows in Python and R cannot meaningfully scale on this data: a single big-bank fact table can
easily exceed a billion rows per quarter, and intraday streaming feeds (card swipes, ATM withdrawals, wires) routinely exceed thousands of events
per second.

This project builds a **complete distributed-ML pipeline** on the canonical UCI bank-marketing dataset (`bank.csv`, 4,521 records, 17 features),
exercising **five distinct distributed primitives** in a single reproducible notebook:

1. **Hadoop-equivalent storage** via partitioned Parquet.
2. **Hive warehousing** via Spark SQL with Hive support, queried with eight production-grade HiveQL analytical statements.
3. **Spark exploratory data analysis** with 17 charts following the UBM rule and three formal hypothesis tests.
4. **Spark ML predictive modelling** — three classifiers (Logistic Regression, Random Forest, Gradient-Boosted Trees) built on one shared Pipeline,
   tuned by 3-fold cross-validation, and evaluated on metrics chosen specifically for an imbalanced binary target.
5. **Spark Structured Streaming** for real-time transaction monitoring with windowed aggregations and a fraud-flagging rule.

The headline business finding: the bank can materially lift marketing ROI by **scoring leads pre-call** with a tuned GBT model (≈0.85 ROC-AUC),
**re-allocating call-centre capacity** to the high-conversion months (Mar/Sep/Oct/Dec), and **re-targeting prior-success clients** on a strict
contact cadence. The headline engineering finding: the same distributed platform that powers offline lead scoring also serves the real-time
fraud-signal stream with zero rewrites — exactly the unified-architecture story that justifies the move to Spark in the first place.

---

## 2. Business Context & Problem Statement

### 2.1 Why distributed?

A typical Tier-1 retail bank operates with:

- **Volume** — tens to hundreds of TB of cumulative customer data; multi-GB nightly batches.
- **Variety** — relational core-banking tables, semi-structured CRM exports, JSON event logs, free-text complaint records, and image-based KYC artefacts.
- **Velocity** — millions of card-and-account events per day, with sub-second SLAs on fraud-screening.

Single-machine pandas/scikit-learn cannot keep up. Distributed engines (Spark) and distributed storage (HDFS / object stores) are not a *premium
upgrade* — they are the *only* viable substrate at this scale.

### 2.2 The specific problem

The bank runs **direct-marketing phone campaigns** for term-deposit products. Each call costs money — agent time, telephony, opportunity cost — and
the historical conversion rate is ~11.5%. The marketing director's three questions:

1. **Who is most likely to subscribe** so we can rank-order calls?
2. **Which campaign tactics actually convert** so we can refine the playbook?
3. **Can the same data plumbing also monitor live transactions** for fraud signals?

This project answers all three.

### 2.3 Business objective

> **Maximise expected revenue per call** by scoring every client with a predicted-subscription probability, calling them in descending order of
> probability up to a daily capacity, and recycling the cheapest improvement levers (channel, month, prior-outcome targeting) immediately.

---

## 3. System Architecture

```
                  ┌───────────────────────┐
                  │   bank.csv (source)   │
                  └───────────┬───────────┘
                              │  spark.read.csv (Section 1)
                              ▼
                  ┌──────────────────────────────────────┐
                  │   Cleaned & engineered DataFrame      │
                  │   (UDFs in src/utils.clean_and_eng…) │
                  └─────────────┬────────────────────────┘
            ┌─────────────┐ partition=month
            │ HDFS-style  │◄─────────────────────────────
            │ Parquet WH  │
            │ warehouse/  │
            └──────┬──────┘
                   │ Hive EXTERNAL TABLE
                   ▼
            ┌─────────────┐      ┌─────────────┐
            │  HiveQL × 8 │ ───► │  reports/   │
            │  analytical │      │  *.csv      │
            └─────────────┘      └─────────────┘
                   │ Spark SQL
                   ▼
       ┌─────────────────────────────┐
       │   Spark EDA (UBM × 17 + 3HT) │
       └─────────────┬───────────────┘
                     │ Pipeline stages
                     ▼
       ┌─────────────────────────────┐         ┌──────────────────┐
       │  Spark ML Pipeline           │ ───►   │ models/best_     │
       │  LR · RF · GBT + CV tuning   │         │  pipeline/        │
       └─────────────────────────────┘         └──────────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │  Data Parallelism / .explain │
       │  benchmarks                  │
       └─────────────────────────────┘

           ╔══════════════════════════════════════════════╗
           ║      PARALLEL TRACK — Structured Streaming    ║
           ║                                                ║
           ║   transaction_producer.py                      ║
           ║   ──► stream_input/*.json (JSON micro-batches)║
           ║                  │                             ║
           ║                  ▼                             ║
           ║   streaming_consumer.py                        ║
           ║   readStream → withWatermark(2m)               ║
           ║   → window(60s) groupBy(job)                   ║
           ║   → suspicious-flag → sink                     ║
           ╚══════════════════════════════════════════════╝
```

### 3.1 Why these components?

- **Spark** as the unified compute layer for batch, SQL, ML, and streaming — one cluster, one API, one operations team.
- **Parquet** as the storage format — columnar, splittable, compressed; the de-facto cold-data format in modern warehouses.
- **Hive** as the table-metadata layer — the SQL surface that BI and operations teams already know.
- **Spark Structured Streaming** as the real-time engine — the same DataFrame API as batch, with explicit watermarking and event-time semantics.

---

## 4. Dataset Description

`bank.csv` — 4,521 rows × 17 columns. UCI Bank Marketing Dataset (S. Moro, P. Cortez, P. Rita).

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `age` | int | 18–95 range |
| 2 | `job` | string | 12 levels; 38 `unknown` |
| 3 | `marital` | string | married / single / divorced |
| 4 | `education` | string | primary / secondary / tertiary / unknown |
| 5 | `default` | string | binary yes/no |
| 6 | `balance` | int | EUR — heavy-tailed, some negative |
| 7 | `housing` | string | binary yes/no |
| 8 | `loan` | string | binary yes/no |
| 9 | `contact` | string | cellular / telephone / unknown |
| 10 | `day` | int | 1–31 |
| 11 | `month` | string | jan…dec |
| 12 | `duration` | int | seconds — **leaks the target** (known only after the call) |
| 13 | `campaign` | int | contact count in this campaign |
| 14 | `pdays` | int | -1 means never previously contacted |
| 15 | `previous` | int | contacts before this campaign |
| 16 | `poutcome` | string | success / failure / other / unknown |
| 17 | `y` | string | **target** — yes/no |

Class balance: 4,000 `no` / 521 `yes` → 11.5% positive.

---

## 5. Part-by-Part Methodology

### 5.1 Part 1 — Hadoop + Hive

**Objective.** Demonstrate distributed storage and SQL-style warehousing on banking data.

**Approach.**

1. Write the cleaned DataFrame to **partitioned Parquet** under `warehouse/bank_parquet/month=<jan…dec>/`. This mirrors a production HDFS layout
   exactly — Hadoop / S3 / ADLS clients would discover the same directory pattern.
2. Register a **Hive external table** over the Parquet warehouse via `SparkSession.enableHiveSupport()`. We use `EXTERNAL` so dropping the table
   does not delete the underlying files (the canonical pattern for governance-sensitive bank data).
3. Reconstruct partition metadata via `MSCK REPAIR TABLE` — the same Hive incantation a real DBA would type.
4. Run **8 HiveQL analytical queries** that surface the marketing team's daily questions (see [`src/hive/analytical_queries.hql`](../src/hive/analytical_queries.hql)).

**Results.** Headline aggregations:

- `student` and `retired` jobs convert at >25%, multiples above the baseline.
- `mar` / `sep` / `oct` / `dec` have markedly higher subscription rates than May.
- Clients with `poutcome=success` carry forward to ~64% conversion in the new campaign.

**Trade-offs.**

- Partitioning by `month` (12 partitions) is generous for this dataset but realistic at scale. For TB-scale fact tables we would partition by
  `year/month` and then bucket by `client_id` to keep partition counts in the low thousands.
- Hive on Spark via `enableHiveSupport()` runs on a local Derby metastore — fine for development; in production this would be a Postgres-backed
  metastore behind a Hive-server-2 gateway.

### 5.2 Part 2 — Spark EDA

**Objective.** Surface trends, patterns and anomalies on banking-scale data using Spark's distributed compute.

**Approach.**

- Aggregations stay in Spark (`groupBy.agg`) so the same code runs on 5 GB or 50 TB; we only `.toPandas()` small summary frames for plotting.
- 17 visualisations following the **UBM** rule (Univariate → Bivariate → Multivariate), each annotated with the rubric's three-question
  triple (*why this chart · what insight · business impact*).
- Three formal hypothesis tests:
  - **Welch's t-test** on `duration` between subscribers and non-subscribers (p ≪ 0.001).
  - **Chi-square** for `housing` vs `y` (p ≪ 0.001).
  - **Chi-square** for `education` vs `y` (p ≪ 0.05).

**Key findings.**

- `duration` is the single strongest correlate of `y` (r ≈ +0.40) but **leaks the target** — usable for call-quality models, not for pre-call
  lead scoring.
- `poutcome=success`, `contact=cellular`, and `housing=no` are the strongest pre-call signals.
- The high-volume month (May) is also the *low-conversion* month — re-allocating calls is a near-free uplift.

**Trade-offs.**

- Pulling the cleaned data into pandas for matplotlib plotting is fine at 4.5k rows; at 4.5B rows we would compute aggregations in Spark and only
  bring the summary table back, or use Spark-native plotting helpers.

### 5.3 Part 3 — Spark ML

**Objective.** Train a production-grade predictive model for term-deposit subscription using Spark ML.

**Approach.**

- One shared **`Pipeline`** with stages: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler` → estimator.
- Three estimators benchmarked: `LogisticRegression` (class-weighted), `RandomForestClassifier`, `GBTClassifier`.
- **`CrossValidator` + `ParamGridBuilder`** for 3-fold tuning on `areaUnderROC`.
- Metric suite: **ROC-AUC**, **F1**, **weighted precision**, **weighted recall** — chosen explicitly for the imbalanced target.

**Why the metric choice?**

- A constant-`no` classifier scores 88.5% accuracy. **Accuracy is a trap** on imbalanced targets.
- ROC-AUC is invariant to class balance, so it measures the *ranking ability* of the model — which is exactly what a call-list scorer needs.
- F1, precision, recall complete the picture: precision drives **cost-per-acquisition**, recall drives **revenue coverage**.

**Why class weights instead of SMOTE?**

- SMOTE inflates training data and shuffle volume in Spark.
- Class weights are deterministic, single-pass, and supported natively by `LogisticRegression`'s `weightCol` parameter.
- Tree ensembles already handle imbalance well; we leave them unweighted to compare.

**Results.** GBT wins on ROC-AUC after tuning; RF is a strong second. Feature importance consistently surfaces: `poutcome_ohe`, `month_ohe`,
`housing`, `balance`, `age`, `contact_ohe`.

### 5.4 Part 4 — Spark Streaming

**Objective.** Demonstrate real-time transaction monitoring on the same Spark cluster.

**Approach.**

- A synthetic **transaction producer** (`src/streaming/transaction_producer.py`) generates JSON micro-batches into a watched directory. Account
  identities are seeded from `bank.csv` so job-category aggregations downstream are representative.
- A **Structured-Streaming consumer** (`src/streaming/streaming_consumer.py`) reads them with `readStream.json(...)`, applies a 2-minute
  watermark, runs a 60-second tumbling window grouped by job category, and writes to an aggregation sink.
- A naive **fraud rule** (`amount > 5000 EUR`) flags suspicious transactions per window.

**Why file-source streaming?**

- Zero external dependencies (no Kafka, no Pulsar, no JDBC binlog reader).
- Fully reproducible by any grader: clone the repo, run the producer, run the consumer, watch the console output.
- The same code with a one-line change (`.format("kafka")`) works against a real broker.

### 5.5 Part 5 — Data Parallelism

**Objective.** Make the cost of distributed computation visible and quantify common optimisation levers.

**Approach.** Six scenarios benchmarked end-to-end:

1. Default partitioning
2. `repartition(16, "job")` — co-locate grouping key
3. `coalesce(2)` — reduce partitions without shuffle
4. 3× repeated aggregation, no cache
5. 3× repeated aggregation, cached
6. Shuffle join vs broadcast join (with `.explain()`)

**Findings.** Repartition wins on the first aggregation, cache wins on repeated aggregations, broadcast joins win when one side is small —
exactly the rules of thumb every Spark engineer learns.

---

## 6. Predictive Modeling Deep-Dive

### 6.1 Feature engineering

Categorical NULLs (from the `unknown`-to-NULL coercion in wrangling) are filled with a constant `missing` token so `StringIndexer` treats them as
their own category — preserving the information that the value was missing, which itself is predictive (e.g. `contact=unknown` correlates with
low conversion).

`pdays = -1` is split into:

- A binary `never_contacted_before` flag.
- A non-negative `pdays` count (0 if never contacted).

This avoids tree splits on a fake negative magnitude and lets linear models distinguish "never contacted" from "contacted a long time ago".

### 6.2 Pipeline composition

| Stage | Inputs | Output |
|---|---|---|
| `StringIndexer × 7` | each categorical column | `<col>_idx` |
| `OneHotEncoder × 7` | each `_idx` column | `<col>_ohe` (sparse) |
| `VectorAssembler` | numeric + all `_ohe` | `features_raw` |
| `StandardScaler` | `features_raw` | `features` |
| **Estimator** | `features`, label `y` | model |

Wrapping all stages in `Pipeline.fit()` ensures the *exact same* transformations are applied at training and inference — the only reproducible
way to deploy.

### 6.3 Hyperparameter grids

| Model | Grid |
|---|---|
| Logistic Regression | `regParam ∈ {0, 0.01, 0.1}`, `elasticNetParam ∈ {0, 0.5}` |
| Random Forest | `numTrees ∈ {50, 150}`, `maxDepth ∈ {6, 10}` |
| GBT | `maxDepth ∈ {4, 6}`, `maxIter ∈ {50, 100}` |

3-fold CV (more folds are overkill on 4.5k rows). `parallelism=2` lets Spark fit two grid points in parallel on a developer laptop.

### 6.4 Feature importance interpretation

Top contributors from the best GBT model:

1. **`poutcome_ohe[success]`** — past behaviour is the dominant predictor.
2. **`month_ohe`** features — seasonality.
3. **`housing`** — mortgaged customers convert less.
4. **`balance`** — wealth proxy.
5. **`age`**, **`contact_ohe`**, **`campaign`** — secondary signals.

These directly inform the business recommendations in §12 of the notebook.

---

## 7. Streaming Architecture & Window Operations

### 7.1 Watermarking

We declare a **2-minute watermark** on `txn_time`. This tells Spark:

> "Any event arriving more than 2 minutes after its event timestamp can be dropped — its window has already been finalised."

Without a watermark, Structured Streaming keeps state forever for late-arriving events. The 2-minute value reflects the bank's tolerance for
out-of-order events at the ingestion edge.

### 7.2 Tumbling vs sliding

We use **tumbling** 60-second windows — non-overlapping, gives a clean per-minute aggregate. For trending visualisations a **sliding** window
(e.g. 60-second window, 10-second slide) gives a smoother stream of partial aggregates at the cost of state-store overhead.

### 7.3 Output modes

| Mode | Notes |
|---|---|
| `append` | Only new rows emitted; requires watermark for windowed aggs |
| `update` | Changed rows emitted; lower latency for dashboards |
| `complete` | Full result table re-emitted each trigger (used here for the demo) |

In a production deployment we would write to **Kafka** or a **Delta table** in `append` mode behind a Grafana board.

### 7.4 Fraud-rule extensibility

The current rule (`amount > 5000`) is intentionally trivial. In production it would be:

1. A **per-account z-score** computed from a rolling 30-day window of historical amounts.
2. A pre-trained **anomaly model** scored row-by-row via `mapInPandas` or `transform`.
3. A **multi-signal** rule combining amount, geography (IP-vs-card-country), time-of-day, and device fingerprint.

---

## 8. Parallelism & Performance Tuning

### 8.1 Partitioning rules of thumb

- Aim for ~128 MB partitions on cold data. Too small → metadata overhead; too large → out-of-memory + skew.
- `repartition(n, col)` shuffles. `coalesce(n)` does not — it just merges partitions. Coalesce can starve parallelism if `n` is too small.
- For joins, **co-partition the two sides on the join key** to avoid shuffle.

### 8.2 Caching

- `cache()` = `persist(MEMORY_AND_DISK)`. Use it when the same DataFrame is consumed by multiple downstream operations.
- Always call an action after `cache()` to materialise (e.g. `.count()`).
- Forgetting to `unpersist()` is a common executor-OOM cause in long-running notebooks.

### 8.3 Broadcast joins

- Spark auto-broadcasts when one side is below `spark.sql.autoBroadcastJoinThreshold` (default 10 MB).
- Force it explicitly with `F.broadcast(small_df)` when you know one side is small but Spark's stats might disagree.
- `.explain()` shows `BroadcastHashJoin` vs `SortMergeJoin` — reading these plans is the day-to-day reality of Spark performance work.

### 8.4 Catalyst plans (`.explain()`)

A typical broadcast join plan looks like:

```
== Physical Plan ==
*(2) BroadcastHashJoin [job#10], [job#20], Inner, BuildRight
:- *(2) Filter isnotnull(job#10)
:  +- *(2) FileScan parquet [...]
+- BroadcastExchange HashedRelationBroadcastMode(...)
   +- *(1) Filter isnotnull(job#20)
      +- *(1) FileScan parquet [sector_map]
```

The `BroadcastExchange` is what we want — the small side is shipped once to every executor, no shuffle of the big side.

---

## 9. Challenges & Trade-offs

| # | Challenge | Resolution |
|---|---|---|
| 1 | No real Hadoop cluster on a Windows laptop | Use PySpark + `enableHiveSupport()` + partitioned Parquet to *simulate* HDFS+Hive semantically. Real `.hql` scripts shipped alongside. |
| 2 | Data leakage of `duration` into pre-call lead scoring | Exclude `duration` from the feature set; document the leakage explicitly in EDA. |
| 3 | 11.5% positive class | Class weights for LR; metric-driven (ROC-AUC/F1) tuning for all three; threshold tuning deferred to deployment. |
| 4 | Streaming demo must terminate cleanly inside a notebook | Use `format("memory")` sink + an explicit `time.sleep(25)` → `query.stop()`; the standalone consumer supports `--forever`. |
| 5 | Hive metastore lock on re-runs (Windows Derby quirk) | Always call `spark.stop()` at end of notebook; delete `metastore_db/` if the lock persists. |
| 6 | Reproducibility across Colab and local Jupyter | Single `requirements.txt`; cell 1 detects environment and pip-installs PySpark only if missing. |

---

## 10. Future Work

| Area | Upgrade |
|---|---|
| Streaming source | Kafka topic with schema-registry; remove file-source crutch |
| Experiment tracking | MLflow for parameters, metrics, and model registry |
| Orchestration | Airflow DAG: daily EDA refresh → ML retraining → batch scoring → call-list export |
| Cluster | Move from local-mode to a Databricks / EMR cluster with autoscaling |
| Monitoring | Prometheus exporters on Spark executors; Grafana dashboard |
| Fairness | Disparate-impact audit on age/gender before any model gates real outreach |
| Online inference | A FastAPI service wrapping the saved `PipelineModel`, called by the dialer pre-call |

---

## 11. References

- S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 2014.
  [UCI repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
- Apache Spark documentation — [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).
- Apache Spark documentation — [ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html).
- Apache Hive documentation — [LanguageManual DDL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL).
- *High Performance Spark* — Holden Karau & Rachel Warren, O'Reilly.
