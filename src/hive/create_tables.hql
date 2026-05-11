-- =============================================================================
-- HiveQL  —  Bank Marketing Warehouse: schema setup
-- =============================================================================
-- Creates the database and the external table that points at the partitioned
-- Parquet warehouse written by Spark (see notebook Section 4.1 / spark_eda.py).
--
-- Run with:
--     hive -f src/hive/create_tables.hql                  (real Hive)
--     spark-sql -f src/hive/create_tables.hql              (Spark SQL CLI)
-- =============================================================================

CREATE DATABASE IF NOT EXISTS bank_warehouse;
USE bank_warehouse;

DROP TABLE IF EXISTS bank_clients;

CREATE EXTERNAL TABLE bank_clients (
    age                    INT,
    job                    STRING,
    marital                STRING,
    education              STRING,
    `default`              INT,
    balance                INT,
    housing                INT,
    loan                   INT,
    contact                STRING,
    day                    INT,
    duration               INT,
    campaign               INT,
    pdays                  INT,
    previous               INT,
    poutcome               STRING,
    y                      INT,
    never_contacted_before INT,
    age_bucket             STRING,
    balance_bucket         STRING,
    duration_min           DOUBLE
)
PARTITIONED BY (month STRING)
STORED AS PARQUET
LOCATION 'warehouse/bank_parquet';

-- Recover partitions from the directory layout that Spark wrote.
MSCK REPAIR TABLE bank_clients;

-- Sanity-check
SHOW TABLES;
SHOW PARTITIONS bank_clients;
