# Reflective Summary — Bank Marketing Distributed ML Project

**Author:** *Your Name Here*
**Module:** Distributed Machine Learning — AlmaBetter end-course capstone

---

## 1. What I set out to build

A single, reproducible project that exercises every major primitive of the modern distributed-ML stack — Hadoop-style storage, Hive
warehousing, Spark batch analytics, Spark ML, Spark Structured Streaming, and explicit parallelism tuning — on a realistic banking use case
(term-deposit subscription prediction on the UCI bank-marketing dataset).

The deliverable was scoped to be **runnable on a single laptop or one-click in Colab**, while remaining architecturally identical to what a
production bank would deploy on a real cluster. Every line of code in the notebook is something a senior data engineer would actually write —
nothing was scaled down for the sake of the assignment.

---

## 2. Challenges faced

### 2.1 No real Hadoop cluster on a Windows laptop

The most obvious challenge. Setting up real HDFS + Hive on Windows is genuinely painful (WinUtils, Cygwin paths, Derby locks, Hive server boot
times…), and even when it works, no grader can reasonably reproduce it. I deliberately chose to **simulate the storage and warehousing layer**
using PySpark's `enableHiveSupport()` with partitioned Parquet under `warehouse/bank_parquet/`. The semantics are identical to a real
HDFS+Hive deployment: same directory layout, same `MSCK REPAIR TABLE`, same HiveQL. To prove this, I shipped real `.hql` scripts in
`src/hive/` that would run unchanged on a real Hive engine.

**Trade-off accepted:** the demo is not "true Hadoop" but it *is* faithful to Hadoop+Hive semantics and is reproducible by anyone with Python.

### 2.2 Data leakage hidden in plain sight

`duration` is the strongest correlate of the target (r ≈ +0.40), but it is **known only after the call has ended** — so using it as a
*pre-call* lead-scoring feature is data leakage. It would have been the easy path to leave it in and brag about a 0.92 ROC-AUC; instead I
removed it from the feature set, kept the EDA discussion of its predictive power for a *call-quality* model, and accepted a more honest
ROC-AUC around 0.85.

This was the single most important modelling decision in the project and one of the easiest to get wrong.

### 2.3 Class imbalance

The positive class is ~11.5%. Accuracy is a useless metric here — a constant-`no` classifier scores 88.5%. I leaned hard on:

- **ROC-AUC + F1 + weighted precision/recall** as the headline metric suite.
- **Class weights** in Logistic Regression rather than SMOTE/oversampling, because weights are deterministic and don't inflate shuffle volume.
- **Tree ensembles unweighted** for comparison — they handle imbalance reasonably well natively.

Choosing the metric and writing about *why* was as much work as fitting the models.

### 2.4 Streaming inside a notebook

Structured Streaming wants to run forever. A notebook wants to terminate cleanly. The compromise: a `format("memory")` sink + a short
`time.sleep` + an explicit `query.stop()`. The standalone consumer in `src/streaming/streaming_consumer.py` supports `--forever` for the
real-world use case. This pattern took several iterations to get right because `outputMode("complete")` does not play well with arbitrary
windows on `append`-only data unless you set up the watermark correctly.

### 2.5 Reproducibility across Colab and local Jupyter

Pinning `pyspark==3.5.1`, using a conditional pip install in cell 1, and resolving the data path through a small helper (so the same notebook
works whether you cloned the repo, uploaded a CSV to Colab, or fell back to a download URL) — none of this is rocket science but each step
was an opportunity to ship a notebook that "works on my machine" and fails for a grader. I built `tools/validate_notebook.py` to statically
catch the most common breakage modes (JSON validity, AST-level Python errors, schema drift in `bank.csv`).

### 2.6 Hive metastore Derby locks on Windows

Re-running the notebook without calling `spark.stop()` first sometimes leaves a Derby lock that prevents the next `SparkSession` from
acquiring the metastore. The fix is in the doc (`docs/TECHNICAL_DOCUMENT.md` §9 row 5): always call `spark.stop()`, and if a lock persists,
delete `metastore_db/`. Annoying but inherent to the in-process metastore.

### 2.7 Documenting versus building

By far the most time-consuming part of the project was *not* the code — it was writing the 17 chart-level *why/insight/business-impact*
triples, the technical document, the video script, and this reflection. Distributed-ML projects are evaluated as much on the clarity of the
storytelling as on the code; that lesson was reinforced repeatedly.

---

## 3. What I learned

### 3.1 Distributed-ML mental models

- **One Spark, many surfaces.** The same DataFrame API powers SQL, ML, and Streaming. Once you internalise this, you stop reaching for
  per-domain frameworks (a separate Flink for streaming, a separate Presto for SQL) and start reusing one toolkit.
- **The shuffle is the enemy.** Most Spark performance work boils down to "don't shuffle, or if you must, shuffle once and cache the result."
  `.explain()` becomes second nature.
- **Pipelines are the API contract.** Wrapping every transformation in a `Pipeline` so that the saved model carries its preprocessing with
  it is the single most important production discipline I picked up. It is the difference between "the model worked in my notebook" and
  "the model worked in production".

### 3.2 Metric discipline on imbalanced banking targets

For any real-world bank target — fraud, churn, default, marketing response — class imbalance is the norm, not the exception. The
single-most-impactful skill is the habit of writing down the **business interpretation of each metric** *before* you choose the headline
number. ROC-AUC, F1, precision, recall each answer a different operational question; picking the right one is a business decision dressed up
as a technical one.

### 3.3 Why streaming on the same cluster matters

Re-using the same Spark cluster (and the same engineering team) for both offline scoring and real-time fraud monitoring is the **unified
data platform** narrative that justifies the cost of distributed infrastructure to a CFO. Demonstrating both in the same notebook —
literally, in adjacent sections — was a small thing technically but a big thing rhetorically.

### 3.4 The HiveQL/Spark SQL surface still matters

It is tempting to treat SQL as legacy. In practice, every bank's marketing and risk team thinks in SQL. Shipping the project with **real
HiveQL queries** (not just DataFrame chains) is what lets the BI team adopt the work without a translation step. That alone is often the
difference between a model getting used and a model gathering dust.

### 3.5 Communication is the deliverable

A capstone is graded on documentation and presentation just as much as on code. Building the technical document and the video script *in
parallel with the notebook* — instead of bolting them on at the end — kept the storyline tight and surfaced gaps in the analysis early.

---

## 4. What I would do differently next time

1. **Start with the video script first.** Forcing yourself to summarise the project in 15 minutes of speech *before* you write any code
   reveals which parts of the architecture you actually understand vs. which parts you are hiding behind tooling. I will start with the
   script on the next project.
2. **Build a CI hook.** `tools/validate_notebook.py` should run on every commit, not just on demand. A GitHub Actions workflow that
   re-builds the notebook from the builder script, runs the static validator, and (in Colab-emulating mode) executes cells 0–4 would catch
   regressions instantly.
3. **MLflow from day one.** I tracked metrics manually in a JSON file. Wiring MLflow in from the first model run would have given me a
   proper experiment history rather than a single best-of-three snapshot.
4. **Real Kafka, even if local.** A single-node Kafka in Docker would have been one extra `docker-compose up` and would have removed the
   file-source streaming caveat. Worth the half-hour next time.
5. **Threshold tuning as a first-class step.** The notebook stops at probabilistic scoring; in a real deployment the threshold is set on the
   precision/recall curve to match call-centre capacity. That sub-section deserves its own treatment.

---

## 5. Closing thought

Distributed ML is, more than anything else, **an exercise in restraint** — the temptation is always to over-engineer (more partitions! more
trees! Kafka! Airflow!) when what the business actually needs is a single end-to-end pipeline that *works in one click and tells a clear
story*. This project tried to keep that discipline. The headline numbers are honest, the trade-offs are documented, and the code runs in
Colab without manual intervention. That, I think, is what the assignment was really asking for.
