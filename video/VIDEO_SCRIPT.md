# Video Presentation Script — Bank Marketing Distributed ML

**Target length:** ≥ 15 minutes (rubric requires ≥ 15 min; assignment guidelines suggest ~15 min split across 8 sections).
**Format:** Screen-share of the notebook + repository + occasional architecture-diagram slide.
**Tone:** Conversational, confident, business-first. Pause briefly after every major insight to let it land.

---

## How to use this script

- Each numbered section has a **time target**, **spoken script** (read it aloud as prose, not as bullet points), **on-screen cues**
  (what to show), and **Q&A prep** for the standard follow-up interview questions at the end.
- Don't *read* the script word-for-word — internalise it, then deliver it naturally in your own voice. The phrasing here is the floor, not
  the ceiling.
- Total wall-clock target is ~15.5 minutes including transitions.

---

## Section 1 — Introduction *(1.5 min)*

**On screen:** README hero section + the architecture diagram.

**Spoken script:**

> "Hi — in the next fifteen minutes I'll walk you through an end-to-end distributed machine learning project I built for the AlmaBetter
> capstone. The problem domain is **retail banking**, specifically a direct-marketing campaign for term-deposit subscriptions, using the UCI
> bank marketing dataset.
>
> The reason this is a *distributed* ML project — and not a regular scikit-learn project — is that **modern banks generate data at a scale
> single-machine workflows can't handle**. We're talking hundreds of terabytes of customer data, millions of transaction events per day, and
> hard latency SLAs on fraud screening. So I built the project on the standard distributed stack: **Hadoop-style partitioned storage, Hive
> warehousing, Apache Spark for batch analytics, Spark ML for predictive modelling, and Spark Structured Streaming for real-time
> transaction monitoring** — all in a single, reproducible Colab notebook.
>
> What you should look for in this walkthrough: how the same Spark cluster does double duty for batch *and* streaming, how I handled the
> ~11.5% class imbalance honestly, and how I avoided a hidden data-leakage trap that would have inflated my metrics."

**Q&A primer:** If asked "why Spark over Dask or Ray?" — *industry adoption, mature Hive integration, unified batch+streaming API.*

---

## Section 2 — Problem Understanding *(2 min)*

**On screen:** Problem Statement + Business Objective sections of the notebook.

**Spoken script:**

> "Let me set the business context. The bank runs direct-marketing phone campaigns to sell term deposits. Every call costs money — agent
> time, telephony, opportunity cost — and historically only about **eleven and a half percent** of contacted clients say yes. The marketing
> director's three questions are: *who is most likely to subscribe, which campaign tactics actually convert, and can we re-use the same
> data plumbing to monitor transactions for fraud in real time*.
>
> Traditional single-machine Python can't answer these at production scale. Here's why distributed computing is non-negotiable:
>
> - **Volume.** A Tier-1 bank's customer table is tens of terabytes; a single quarter of card transactions is over a billion rows. Pandas
>   would page to disk before the first `groupby` finished.
> - **Variety.** Banks mix structured relational tables, semi-structured CRM exports, JSON event logs, free-text complaints, and KYC images
>   — a Spark DataFrame is the only common substrate.
> - **Velocity.** Fraud-screening windows are sub-second. Batch ETL alone is not enough; you need a true streaming engine.
>
> And finally — **why real-time matters in banking.** A delayed fraud signal is a stolen card. A delayed lead score is a missed
> conversion. Latency is revenue."

**Q&A primer:** If asked "what specifically about the dataset is hard?" — *only 4521 rows here, but the architecture must scale to billions; the class imbalance and the duration-leakage trap are the real modelling challenges.*

---

## Section 3 — Data Storage & Processing Architecture *(2 min)*

**On screen:** Architecture diagram (in README and Technical Doc) + notebook Section 4.1 / 4.2 with the partitioned-Parquet directory tree
visible.

**Spoken script:**

> "Here's the architecture. Data flows from **left to right**:
>
> 1. **Ingestion.** The raw `bank.csv` is read by Spark.
> 2. **Storage layer (Hadoop + Hive).** Spark writes the cleaned data as **partitioned Parquet** under `warehouse/bank_parquet/`, with one
>    partition per month. That directory layout is **literally what HDFS would look like** in production — you can see the `month=jan/`,
>    `month=feb/` folders right here. On top of that, I register a **Hive EXTERNAL TABLE** through `enableHiveSupport()`, which means the
>    same warehouse is now queryable from any HiveQL client. I'll run the queries in a minute.
> 3. **Batch analytics layer (Spark EDA + Spark ML).** Spark consumes the same warehouse via the Hive table. No re-ingestion. That's the
>    single-source-of-truth property that justifies the Hadoop layer in the first place.
> 4. **Real-time layer (Spark Streaming).** A *separate* track on the same Spark cluster: a transaction producer drops JSON micro-batches
>    into a watched folder, and a Structured-Streaming consumer aggregates them in 60-second tumbling windows.
>
> The architectural payoff here is that **one Spark cluster, one DataFrame API, and one engineering team** owns the entire pipeline — from
> daily batch lead-scoring to real-time fraud alerting. That unified-platform story is what justifies the move to Spark to a CFO."

**Q&A primer:** "Why Parquet over CSV?" — *columnar, splittable, compressed; supports partition pruning; standard for cold data in banks.*

---

## Section 4 — Exploratory Analysis with Spark *(2.5 min)*

**On screen:** Scroll through notebook §5. Pause on Charts 6 (class balance), 7 (job × y), 10 (month × y), 11 (duration violin), 13 (poutcome × y), 14 (heatmap).

**Spoken script:**

> "I did the EDA in Spark — every aggregation runs in the cluster, I only pull the small summary frames back to the driver for plotting.
> The same code would run on a five-billion-row warehouse table unchanged.
>
> There are **seventeen charts following the UBM rule** — Univariate, Bivariate, Multivariate. Each one is annotated with three mandatory
> answers: *why this chart, what the insight is, and what the business impact is*. Let me hit the four most consequential ones:
>
> - **Chart 6 — class balance.** The positive class is eleven and a half percent. That dictates *everything* downstream: accuracy is a
>   trap, we must optimise ROC-AUC and F1.
> - **Chart 7 — subscription rate by job.** Students and retirees convert at roughly twice the base rate. There's a clear retiree-pension
>   cross-sell story sitting in this chart.
> - **Chart 10 — subscription rate by month.** May has the highest *volume* and the *lowest* conversion. Mar, Sep, Oct, Dec are the
>   high-yield months. Just re-allocating call-centre capacity is near-free incremental revenue.
> - **Chart 11 — duration by outcome.** Subscribers have dramatically longer calls. This is the *strongest* single predictor in the
>   dataset — and it's the one **I do not use for modelling**, because `duration` is known only after the call ends. Using it would be
>   data leakage. This is the most important judgment call in the project: I'd rather have an honest 0.85 AUC than a leaky 0.92.
>
> I also ran three formal hypothesis tests — a Welch's t-test on duration and chi-square tests on housing and education versus the target.
> All three reject the null at conventional significance levels.
>
> **The benefit of doing EDA in Spark** is precisely that *none of this analysis has to change* when the dataset goes from four-and-a-half
> thousand rows to four-and-a-half billion."

**Q&A primer:** "What's an example of treating Spark like a notebook anti-pattern?" — *collecting the full DataFrame to driver before grouping. I aggregate in Spark and only `.toPandas()` the summary.*

---

## Section 5 — Predictive Modeling with Spark ML *(2 min)*

**On screen:** Notebook §7-§8 — the Spark ML Pipeline stages + the model-comparison bar chart + feature-importance plot.

**Spoken script:**

> "The prediction problem is **will this client subscribe to a term deposit?** — a binary classifier on the 11.5% positive class.
>
> I built one shared **Spark ML Pipeline** — StringIndexer for categoricals, OneHotEncoder, VectorAssembler, StandardScaler, then the
> estimator at the end. Wrapping all of that in a Pipeline is **the single most important production discipline**: the preprocessing rides
> with the saved model, so inference at deploy time is byte-identical to training.
>
> I compared three estimators on that same pipeline: **Logistic Regression** with class weights, **Random Forest**, and
> **Gradient-Boosted Trees**. Each one was tuned with a 3-fold `CrossValidator` and a small parameter grid.
>
> The metric suite is deliberate. **ROC-AUC** measures the model's ranking ability, which is what a call-list scorer actually needs.
> **F1** balances precision and recall and is robust on imbalanced data. **Precision** drives cost-per-acquisition. **Recall** drives
> revenue coverage. Accuracy alone would be useless — a constant-no classifier scores 88.5%.
>
> GBT wins after tuning at around 0.85 ROC-AUC. **Feature importance** confirms the EDA story: `poutcome`, `month`, `housing`, `balance`,
> `age`, `contact` are the dominant signals. The model is then **saved as a Spark ML PipelineModel and reloaded** for a sanity-check
> prediction — that round-trip is exactly what the production scoring service would do.
>
> The business impact is concrete: a tuned GBT predicting at the top decile should push effective conversion above 30%, almost tripling
> the marketing team's ROI per call."

**Q&A primer:** "How does Spark ML differ from sklearn?" — *distributed training, lazy evaluation, transformations as a graph, models that
serialise with their preprocessing.*

---

## Section 6 — Real-Time Streaming & Data Parallelism *(2 min)*

**On screen:** Notebook §9 (streaming demo running) + §10 (parallelism benchmark bar chart).

**Spoken script:**

> "Now the real-time side. The same Spark cluster runs a **Structured-Streaming pipeline**. A producer drops JSON micro-batches of synthetic
> transactions into a watched directory; the consumer reads them with `readStream`, applies a two-minute watermark, runs a **60-second
> tumbling window grouped by job category**, and writes the aggregates to a sink. A naive amount-threshold rule flags suspicious
> transactions per window — that's a placeholder for what would be a per-account z-score or a pre-trained anomaly model in production.
>
> The reason this matters for a bank: **fraud detection, customer-service alerts, and live dashboards all share the same plumbing as the
> overnight batch lead-scoring**. One platform, one operations team.
>
> And finally — **data parallelism**. The last section of the notebook benchmarks six optimisation scenarios: default partitioning,
> repartition, coalesce, cache versus no-cache, shuffle join versus broadcast join. Each is timed; each comes with a `.explain()` plan.
>
> The findings are textbook: **repartitioning on the grouping key co-locates the shuffle** for the next aggregation; **caching wins on
> repeated reads**; **broadcasting the small side eliminates the shuffle** of the big side. On a 4.5k-row dataset the absolute speedups
> are tiny, but the *shape* of the curves scales to terabytes — these are the exact levers a Spark engineer pulls every day."

**Q&A primer:** "When would a broadcast join lose?" — *when the broadcast side is too large for executor memory; auto-broadcast threshold defaults to 10 MB.*

---

## Section 7 — Challenges, Optimisation & Trade-offs *(1.5 min)*

**On screen:** `docs/REFLECTIVE_SUMMARY.md` open at the "Challenges faced" section.

**Spoken script:**

> "Let me be honest about the trade-offs I made.
>
> - **No real Hadoop cluster on a Windows laptop.** I simulated HDFS with partitioned Parquet and Hive with `enableHiveSupport()`. The
>   semantics are faithful — same directory layout, same `MSCK REPAIR`, same HiveQL — and I shipped genuine `.hql` scripts that would run
>   on a real Hive engine. But it's not a real Hadoop cluster.
> - **Duration leakage.** I explicitly excluded `duration` from the modelling features even though it's the strongest correlate. Accepting
>   a 0.85 ROC-AUC over a leaky 0.92 was the most important modelling decision in the project.
> - **Streaming inside a notebook.** Structured Streaming wants to run forever; a notebook wants to terminate. The compromise is a
>   memory-sink + a 25-second sleep + an explicit `query.stop()`. The standalone consumer at `src/streaming/streaming_consumer.py`
>   supports `--forever` for the real deployment.
> - **Imbalance handling.** I chose class weights over SMOTE because weights are deterministic and don't inflate shuffle volume. Trees got
>   no weights at all — they handle imbalance reasonably well natively.
>
> The performance optimisations I focussed on: **partitioning by the grouping key, caching DataFrames consumed multiple times, broadcasting
> the small side of joins, and reading `.explain()` plans before reaching for a profiler**. These are the four levers that win 80% of Spark
> performance issues in practice."

**Q&A primer:** "What's the biggest risk of cache?" — *forgetting to `unpersist()` in long-running jobs; eats executor memory.*

---

## Section 8 — Learnings & Practical Value *(1.5 min)*

**On screen:** Conclusion of the notebook + Future Work section of the technical doc.

**Spoken script:**

> "Three takeaways from this project.
>
> First, **distributed ML is one unified API surface, not five separate frameworks**. The same Spark DataFrame powers SQL, ML, and
> streaming. Once you internalise that, you stop reaching for per-domain tools and start re-using one toolkit.
>
> Second, **metric discipline matters as much as model choice**. For any imbalanced banking target — fraud, churn, default, marketing
> response — writing down the business interpretation of each metric *before* you tune the model is the highest-leverage habit.
>
> Third, **distributed tools justify themselves through unified architecture, not raw speed**. The reason a CFO writes the Spark cluster
> cheque is that one platform serves both the offline lead-scoring batch and the real-time fraud feed — not because Spark is faster than
> pandas.
>
> **How would I extend this to production?** A short list: replace the file-source streaming with **Kafka**; track experiments with
> **MLflow**; schedule with **Apache Airflow**; deploy on a **Databricks or EMR cluster** with autoscaling; wrap the model in a
> **FastAPI service** for online inference; and — importantly — run a **fairness audit** on age and gender before the model gates real
> customer outreach.
>
> The full repository, the technical document, and the reflective summary are linked in the description. Thanks for watching."

---

## Standard follow-up questions — short answers

| # | Question | One-paragraph answer |
|---|---|---|
| 1 | **Why is Spark preferred for large-scale data analysis?** | Spark is the de-facto unified distributed engine for batch SQL, ML, and streaming. It runs on the JVM with a memory-aware DAG scheduler, lazy evaluation, and Catalyst query optimisation; its DataFrame API is identical across the four surfaces; and it integrates natively with Hive, HDFS, S3, ADLS, and every JDBC source. The combination of *one API, one cluster, mature integrations* is why every major bank standardises on it. |
| 2 | **How does Spark ML differ from traditional local ML workflows?** | Three differences. **(1) Distributed training** — algorithms are partitioned across executors so training scales with data. **(2) Pipelines as first-class objects** — preprocessing rides with the model, eliminating training-serving skew. **(3) Lazy evaluation** — transformations build a DAG; only an action materialises it, which lets Catalyst rewrite the plan for performance. Locally, scikit-learn is fine for ≤10M rows; beyond that, Spark ML wins. |
| 3 | **What is the role of Spark Streaming in banking systems?** | Banks need sub-second latency on fraud screening, near-real-time aggregations for dashboards, and event-driven workflows for compliance alerts. Spark Structured Streaming provides exactly-once event-time semantics with watermarking and windowed aggregations, on the *same* DataFrame API as the batch pipeline. That means one engineering team owns both surfaces. |
| 4 | **How does data parallelism improve performance?** | Data parallelism partitions the dataset across executors and runs the same operation on each partition concurrently. On Spark, the partition count, partition key, and shuffle behaviour are the three levers: **repartition** by the join/group key to co-locate work, **cache** hot DataFrames to skip recomputation, and **broadcast** small tables to avoid shuffling the large side. Reading `.explain()` plans and watching the SparkUI's stage skew are the day-to-day tools. |
| 5 | **How would you scale this project further in production?** | (1) Replace file-source streaming with Kafka topics behind a schema registry. (2) Add MLflow for experiment tracking and a model registry. (3) Orchestrate the daily batch with Airflow. (4) Move from local-mode Spark to a Databricks or EMR cluster with autoscaling. (5) Wrap the saved `PipelineModel` in a FastAPI online-inference service called by the dialer pre-call. (6) Add a Prometheus/Grafana monitoring layer and a quarterly fairness audit before any model gates real outreach. |

---

## Recording checklist

- [ ] Clean desktop / hide bookmarks bar.
- [ ] Open the notebook with the kernel already started so cells visibly have outputs.
- [ ] Have the architecture diagram on a separate browser tab / slide for §1 and §3.
- [ ] Test microphone level; speak ~20% slower than you think you should.
- [ ] Record one 30-second test, play it back, then record for real.
- [ ] Keep total length **at least 15 minutes** to meet the rubric.
- [ ] Upload to YouTube as *unlisted* or to Google Drive with link-share enabled.
