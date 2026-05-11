# Project Plan — Bank Distributed ML

**Project:** Distributed Machine Learning on Banking Data (`bank.csv`)
**Module:** End-Course Summative Assignment — Distributed Machine Learning
**Mode:** Individual submission
**Date locked:** 2026-05-11

---

## 1. Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction target | `y` (term deposit subscription) | Aligns with dataset's natural target; strong marketing-ROI business framing |
| Hadoop/Hive approach | PySpark + Spark SQL with Hive support (`enableHiveSupport()`) | Reproducible in Colab, simulates HDFS via partitioned Parquet, evaluator can run one-click. Real `.hql` scripts shipped as artifacts. |
| Notebook environment | Colab-primary, locally runnable | Submission requires viewable Colab. `pip install pyspark` in cell 1; also runs in local Jupyter. |
| Git workflow | Full automation via `gh` CLI | Pending repo name + visibility confirmation from user. |
| Streaming source | File-source Structured Streaming (JSON chunks dropped into a watched folder) | Zero external dependencies (no Kafka/Flume); fully reproducible. |

---

## 2. The 5 Distributed Parts (mapped to project objectives)

1. **Hadoop + Hive** → distributed storage & SQL-style querying of bank data
2. **Spark EDA** → trends, patterns, anomalies on the dataset
3. **Spark ML** → predictive modeling (term-deposit subscription)
4. **Spark Streaming** → real-time transaction monitoring (fraud / live insights)
5. **Data Parallelism** → optimization & efficiency (partitioning, caching, broadcast joins, query plans)

---

## 3. Four Deliverables

### Deliverable 1 — Main Notebook
`notebooks/Bank_Distributed_ML_Project.ipynb`
Single Colab-ready notebook merging **AlmaBetter template** with the **5 distributed parts**.

| Section | Content |
|---|---|
| Header | Project Name, Type (Classification + Distributed Systems), Individual, 500–600 word Summary, GitHub Link, Problem Statement, Business Objective |
| 0. Environment | Install PySpark, start SparkSession with Hive support |
| 1. Know Your Data | Load via Spark, first look, shape, info, nulls, duplicates |
| 2. Understanding Variables | Columns, describe, unique values, variable descriptions |
| 3. Data Wrangling | Cleaning, type-fixing, `unknown`/`-1` handling |
| **4. PART 1 — Hadoop + Hive** | Partitioned Parquet write (HDFS sim), Hive DB + tables, 8 HiveQL analytical queries |
| **5. PART 2 — Spark EDA** | 15+ UBM charts each with the 3 mandatory markdown answers (why / insight / business impact) + correlation heatmap + pair plot |
| 6. Hypothesis Testing | 3 hypotheses with chi-square / t-tests |
| 7. Feature Engineering | Missing values, outliers, encoding (StringIndexer + OneHotEncoder via Spark ML Pipeline), scaling, class imbalance |
| **8. PART 3 — Spark ML** | Logistic Regression, Random Forest, GBT — Spark ML Pipeline; CrossValidator + ParamGridBuilder; ROC-AUC, F1, precision/recall; feature importance |
| **9. PART 4 — Spark Streaming** | Synthetic transaction stream → Structured Streaming → windowed aggregations + fraud rules |
| **10. PART 5 — Data Parallelism** | `repartition`/`coalesce`, `cache()`/`persist()`, broadcast joins, `.explain()`, before/after timing |
| 11. Save & Reload | Persist best model, reload, sanity-check prediction |
| 12. Solution + Conclusion | Business recommendation, conclusion |

### Deliverable 2 — GitHub Repository

```
bank-distributed-ml/
├── README.md                          # Badges, overview, architecture, setup, results
├── LICENSE                            # MIT
├── requirements.txt                   # pyspark, pandas, matplotlib, seaborn, scipy
├── .gitignore
├── data/
│   └── bank.csv
├── notebooks/
│   └── Bank_Distributed_ML_Project.ipynb
├── src/
│   ├── hive/
│   │   ├── create_tables.hql
│   │   └── analytical_queries.hql
│   ├── spark_eda.py
│   ├── spark_ml_pipeline.py
│   ├── streaming/
│   │   ├── transaction_producer.py    # synthetic stream generator
│   │   └── streaming_consumer.py      # structured streaming job
│   └── utils.py
├── stream_input/                      # sample streaming chunks
├── models/                            # saved best model
├── docs/
│   ├── PROJECT_PLAN.md                # this file
│   ├── TECHNICAL_DOCUMENT.md
│   ├── REFLECTIVE_SUMMARY.md
│   └── architecture.png
└── video/
    └── VIDEO_SCRIPT.md
```

### Deliverable 3 — Technical Document
`docs/TECHNICAL_DOCUMENT.md` — 15–20 pages of Markdown:

1. Executive Summary
2. Business Context & Problem Statement (volume/variety/velocity in banking)
3. System Architecture (ingestion → HDFS/Hive → Spark batch → Spark ML → Streaming)
4. Dataset Description
5. Part-by-Part Methodology (5 subsections: objective → approach → code highlights → results → trade-offs)
6. Predictive Modeling Deep-Dive (algorithms, hyperparameters, metric justification for imbalanced data)
7. Streaming Architecture & Window Operations (tumbling vs sliding, watermarking)
8. Parallelism & Performance Tuning (partitioning strategy, caching, query plans, benchmarks)
9. Challenges & Trade-offs
10. Future Work (Kafka, MLflow, Airflow, cloud deployment)
11. References

### Deliverable 4 — Video Script
`video/VIDEO_SCRIPT.md` — follows guideline's 8-section structure (≥15 min total):

| # | Section | Time |
|---|---|---|
| 1 | Introduction | 1.5 min |
| 2 | Problem Understanding | 2 min |
| 3 | Data Storage & Processing Architecture | 2 min |
| 4 | Exploratory Analysis with Spark | 2.5 min |
| 5 | Predictive Modeling with Spark ML | 2 min |
| 6 | Real-Time Streaming & Data Parallelism | 2 min |
| 7 | Challenges, Optimization & Trade-offs | 1.5 min |
| 8 | Learnings & Practical Value | 1.5 min |

Each section: spoken script + on-screen cues + Q&A prep for the 5 follow-up questions.

---

## 4. Evaluation Rubric Coverage Map

| Rubric criterion (weight) | Where covered |
|---|---|
| 1. Distributed Computing Concepts (10%) | All 5 parts; architecture diagram; technical doc Section 3 |
| 2. Data Analysis & Management — Hadoop/Hive (15%) | Part 1 + HiveQL scripts + Technical Doc §5.1 |
| 3. EDA & Data Preprocessing (15%) | Part 2 (15+ charts) + Section 7 of notebook |
| 4. Model Development & Validation (15%) | Part 3 (3 models, CV, hyperparams, metrics, feature importance) |
| 5. Real-Time Data Processing (10%) | Part 4 (Structured Streaming, windowed aggregations) |
| 6. Data Parallelism & Efficiency (10%) | Part 5 (benchmarks, partitioning, caching) |
| 7. Innovation & Creativity (5%) | Synthetic transaction generator + fraud rules + business storytelling |
| 8. Documentation & Presentation (20%) | README + Technical Doc + Reflective Summary + Video Script |

---

## 5. Execution Order

1. ✅ Save this plan (current step)
2. **Phase 1** — Build the main notebook end-to-end
3. **Phase 2** — Generate streaming scripts + sample chunks
4. **Phase 3** — Write README + ancillary `.hql` and `.py` scripts
5. **Phase 4** — Write Technical Document + Reflective Summary
6. **Phase 5** — Write Video Script
7. **Phase 6** — `git init` + `gh` repo creation + push (pause for repo name + visibility)

---

## 6. Open Items (need user input before Phase 6)

- GitHub repository name (suggested: `bank-distributed-ml`)
- Visibility: public / private
- Confirmation that `gh auth status` is authenticated on this machine
