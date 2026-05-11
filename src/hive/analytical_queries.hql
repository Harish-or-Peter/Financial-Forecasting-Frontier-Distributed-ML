-- =============================================================================
-- HiveQL  —  Bank Marketing Warehouse: 8 analytical queries
-- =============================================================================
-- Production-grade HiveQL that surfaces the marketing team's daily questions:
-- segment-level conversion rates, campaign performance, customer wealth bands,
-- channel effectiveness, prior-outcome carry-over.
--
-- Run with:
--     hive -f src/hive/analytical_queries.hql              (real Hive)
--     spark-sql -f src/hive/analytical_queries.hql          (Spark SQL CLI)
-- =============================================================================

USE bank_warehouse;

-- -----------------------------------------------------------------------------
-- Query 1 — Subscription rate by job category
-- -----------------------------------------------------------------------------
SELECT job,
       COUNT(*)                    AS n_clients,
       SUM(y)                      AS n_subscribed,
       ROUND(100.0 * AVG(y), 2)    AS subscription_rate_pct
FROM   bank_clients
WHERE  job IS NOT NULL
GROUP  BY job
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 2 — Campaign performance by month
-- -----------------------------------------------------------------------------
SELECT month,
       COUNT(*)                    AS n_calls,
       ROUND(AVG(duration), 1)     AS avg_duration_sec,
       ROUND(100.0 * AVG(y), 2)    AS subscription_rate_pct
FROM   bank_clients
GROUP  BY month
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 3 — Balance buckets vs subscription
-- -----------------------------------------------------------------------------
SELECT balance_bucket,
       COUNT(*)                    AS n_clients,
       ROUND(AVG(balance), 0)      AS avg_balance,
       ROUND(100.0 * AVG(y), 2)    AS subscription_rate_pct
FROM   bank_clients
GROUP  BY balance_bucket
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 4 — Education level vs subscription
-- -----------------------------------------------------------------------------
SELECT COALESCE(education, 'unknown') AS education,
       COUNT(*)                       AS n_clients,
       ROUND(100.0 * AVG(y), 2)       AS subscription_rate_pct
FROM   bank_clients
GROUP  BY education
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 5 — Contact-channel effectiveness
-- -----------------------------------------------------------------------------
SELECT COALESCE(contact, 'unknown')   AS contact_type,
       COUNT(*)                       AS n_calls,
       ROUND(AVG(duration), 1)        AS avg_duration_sec,
       ROUND(100.0 * AVG(y), 2)       AS subscription_rate_pct
FROM   bank_clients
GROUP  BY contact
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 6 — Call-duration cohort (subscribers vs non-subscribers)
-- -----------------------------------------------------------------------------
SELECT CASE WHEN y = 1 THEN 'subscribed' ELSE 'not_subscribed' END AS cohort,
       COUNT(*)                AS n_calls,
       ROUND(AVG(duration), 1) AS avg_duration_sec,
       MIN(duration)           AS min_duration_sec,
       MAX(duration)           AS max_duration_sec
FROM   bank_clients
GROUP  BY y;

-- -----------------------------------------------------------------------------
-- Query 7 — Housing × personal-loan combo segments
-- -----------------------------------------------------------------------------
SELECT housing,
       loan,
       COUNT(*)                  AS n_clients,
       ROUND(100.0 * AVG(y), 2)  AS subscription_rate_pct
FROM   bank_clients
GROUP  BY housing, loan
ORDER  BY subscription_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- Query 8 — Top-performing previous-campaign outcomes (window function)
-- -----------------------------------------------------------------------------
SELECT COALESCE(poutcome, 'unknown') AS poutcome,
       COUNT(*)                       AS n_clients,
       ROUND(100.0 * AVG(y), 2)       AS subscription_rate_pct,
       RANK() OVER (ORDER BY AVG(y) DESC) AS rate_rank
FROM   bank_clients
GROUP  BY poutcome;
