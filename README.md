# Bank Marketing — Distributed Machine Learning

End-to-end **distributed machine-learning pipeline** for predicting term-deposit subscription on the UCI bank marketing dataset.
Built for the AlmaBetter Distributed Machine Learning end-course capstone — exercises every primitive of the modern data stack in one
reproducible Colab-ready notebook.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harish-or-Peter/Financial-Forecasting-Frontier-Distributed-ML/blob/main/notebooks/Bank_Distributed_ML_Project.ipynb)

> 🔗 **Repository:** https://github.com/Harish-or-Peter/Financial-Forecasting-Frontier-Distributed-ML

---

## Project at a glance

| | |
|---|---|
| **Domain** | Retail banking — direct-marketing campaigns |
| **Dataset** | UCI bank marketing (`bank.csv`, 4,521 rows × 17 cols) |
| **Target** | `y` — term-deposit subscription (yes/no, ~11.5% positive) |
| **Stack** | Hadoop (simulated via partitioned Parquet) · Hive (via Spark SQL + Hive support) · Apache Spark · Spark ML · Spark Structured Streaming |
| **Best model** | Tuned Random Forest by ROC-AUC; GBT wins on F1 (see results) |
| **Streaming** | File-source Structured Streaming · 60-second tumbling windows · fraud-flag rule |
| **Deliverables** | Colab notebook · standalone `.py` scripts · HiveQL scripts · technical document · video script |

---

## Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                  DATA INGESTION                        │
                  │   bank.csv  (header + 17 typed columns, 4.5k rows)     │
                  └────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              PART 1 — HADOOP + HIVE                    │
                  │  Spark writes partitioned Parquet  →  HDFS-style       │
                  │  layout (month=jan/, month=feb/, …)                    │
                  │  Hive EXTERNAL TABLE bank_warehouse.bank_clients       │
                  │  ↓  8 HiveQL analytical queries                        │
                  └────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              PART 2 — SPARK EDA  (UBM)                 │
                  │   17 charts · Univariate · Bivariate · Multivariate    │
                  │   Hypothesis testing (Welch's t, chi-square × 2)       │
                  └────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            PART 3 — SPARK ML PIPELINE                  │
                  │  StringIndexer → OneHotEncoder → VectorAssembler →     │
                  │  StandardScaler → {LogReg | RandomForest | GBT}        │
                  │  CrossValidator + ParamGridBuilder · 3-fold            │
                  │  Metrics: ROC-AUC · F1 · Precision · Recall            │
                  └────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │       PART 4 — SPARK STRUCTURED STREAMING              │
                  │  transaction_producer.py  ──►  stream_input/*.json     │
                  │  streaming_consumer.py:                                │
                  │     readStream → withWatermark(2m) → window(60s) →     │
                  │     groupBy(job) → aggregate → suspicious-flag sink    │
                  └────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                  ┌────────────────────────────────────────────────────────┐
                  │       PART 5 — DATA PARALLELISM & OPTIMISATION         │
                  │   repartition · coalesce · cache/persist · broadcast   │
                  │   join · .explain() Catalyst plans · timing benchmarks │
                  └────────────────────────────────────────────────────────┘
```

---

## Repository structure

```
bank-distributed-ml/
├── README.md                                  # ← you are here
├── LICENSE                                    # MIT
├── requirements.txt
├── .gitignore
├── data/
│   └── bank.csv                                # source dataset
├── notebooks/
│   └── Bank_Distributed_ML_Project.ipynb       # ★ main deliverable
├── src/
│   ├── utils.py                                # shared helpers (Spark session, cleaning)
│   ├── spark_eda.py                            # headless EDA + Hive warehouse build
│   ├── spark_ml_pipeline.py                    # headless ML training + persist
│   ├── hive/
│   │   ├── create_tables.hql                   # Hive DDL
│   │   └── analytical_queries.hql              # 8 HiveQL queries
│   └── streaming/
│       ├── transaction_producer.py             # synthetic transaction generator
│       └── streaming_consumer.py               # Structured-Streaming consumer
├── stream_input/                               # 5 sample streaming batches (committed)
├── models/                                     # populated by training scripts
├── reports/                                    # populated by EDA / ML scripts
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── TECHNICAL_DOCUMENT.md                   # ★ detailed technical report
│   └── REFLECTIVE_SUMMARY.md
├── video/
│   └── VIDEO_SCRIPT.md                         # ★ presentation script
└── tools/
    ├── build_notebook.py                       # dev: builds the notebook
    ├── validate_notebook.py                    # dev: static checks
    └── generate_sample_streams.py              # dev: emits sample batches
```

---

## Quick start

### Option A — Google Colab (recommended)

1. Open `notebooks/Bank_Distributed_ML_Project.ipynb` in Colab (`File → Upload notebook`).
2. Upload `data/bank.csv` to `/content/data/` (or let cell 4 auto-download).
3. `Runtime → Run all`. The first cell pip-installs PySpark; everything else runs top-to-bottom.

### Option B — Local Jupyter

```bash
# 1. Clone
git clone https://github.com/Harish-or-Peter/Financial-Forecasting-Frontier-Distributed-ML.git
cd Financial-Forecasting-Frontier-Distributed-ML

# 2. (Optional) Create a virtualenv
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\Activate.ps1

# 3. Install deps. JDK 11+ is required for PySpark.
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook notebooks/Bank_Distributed_ML_Project.ipynb
```

### Option C — Headless `.py` scripts

```bash
# EDA + Hive warehouse build (writes warehouse/bank_parquet + reports/*.csv)
python src/spark_eda.py

# ML training + best-model persistence (writes models/best_pipeline + reports/ml_metrics.json)
python src/spark_ml_pipeline.py

# Streaming demo — open two terminals:
python src/streaming/transaction_producer.py --continuous --interval 2    # producer
python src/streaming/streaming_consumer.py --forever                       # consumer
```

---

## The five distributed parts

| # | Part | Where in the notebook | Standalone script |
|---|---|---|---|
| 1 | **Hadoop + Hive** — partitioned-Parquet warehouse + 8 HiveQL queries | §4 | `src/spark_eda.py`, `src/hive/*.hql` |
| 2 | **Spark EDA** — 17 UBM charts + 3 hypothesis tests | §5–6 | `src/spark_eda.py` |
| 3 | **Spark ML** — LR + RF + GBT on one Pipeline, CV-tuned | §7–8 | `src/spark_ml_pipeline.py` |
| 4 | **Spark Streaming** — windowed aggregations + fraud-flag rule | §9 | `src/streaming/*.py` |
| 5 | **Data Parallelism** — repartition · cache · broadcast · `.explain()` | §10 | (inside notebook) |

---

## Headline results (held-out test set, executed run)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (tuned, class-weighted) | 0.7212 | 0.7334 | 0.8405 | 0.6799 |
| **Random Forest (tuned)** ★ *(best ROC-AUC)* | **0.7356** | 0.8472 | 0.8652 | 0.8859 |
| **GBT (tuned)** ★ *(best F1 / recall)* | 0.6960 | **0.8579** | 0.8630 | **0.8871** |

> Class imbalance (~11.5% positive) makes ROC-AUC and F1 the right headline metrics; accuracy alone would mislead. **Random Forest** is the
> ROC-AUC winner and the one persisted to `models/best_pipeline/`; **GBT** has the better F1 — a different production scenario (e.g. fixed
> call-centre capacity) would justify shipping GBT instead.

Top feature importances (from the best tree model) consistently surface: `poutcome`, `month`, `housing`, `balance`, `age`, `contact` —
matching the EDA storyline.

---

## Documentation

- 📘 **[Technical Document](docs/TECHNICAL_DOCUMENT.md)** — architecture, methodology, results, trade-offs (15–20 pages).
- 📝 **[Reflective Summary](docs/REFLECTIVE_SUMMARY.md)** — challenges faced + learning outcomes.
- 🎥 **[Video Script](video/VIDEO_SCRIPT.md)** — 8-section presentation script targeting ≥15 minutes.
- 📋 **[Project Plan](docs/PROJECT_PLAN.md)** — locked decisions and deliverable map.

---

## Reproducibility notes

- Every random operation uses `seed=42` (`RANDOM_SEED`).
- Spark configuration (`spark.sql.shuffle.partitions=8`, driver memory 2 GB) is set inline so behaviour is identical across machines.
- Hive metadata lives in a local Derby metastore inside `spark-warehouse/`; delete that folder to reset between runs.
- The streaming demo runs for ~30 seconds when invoked from the notebook so the cell terminates cleanly; use `--forever` on the standalone
  consumer for a continuous loop.

---

## Future work

- **Kafka-backed streaming** to replace the file-source ingestion.
- **MLflow** for experiment tracking and model registry.
- **Apache Airflow** to schedule the daily lead-scoring batch.
- **Cluster deployment** on Databricks / EMR with autoscaling.
- **Fairness audit** before the model gates real customer outreach.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- UCI Machine Learning Repository — *Bank Marketing Data Set* (S. Moro, P. Cortez, P. Rita).
- AlmaBetter — Distributed Machine Learning course materials.
