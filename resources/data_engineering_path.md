# Data Engineering Path (Honest Survey)

A short companion for the [Data Engineer](career_roadmap_guide.md#data-engineer) track. This chapter maps ETL, quality, warehouse modeling, and streaming awareness onto modules you already have. It does **not** replace a dedicated Spark or Kafka course.

## What “done” means here

You can move data on a schedule, check quality, model tables for analytics, and explain when batch is enough versus when streaming is worth the cost. You are not claiming production-scale Spark/Kafka mastery from Module 14 alone.

## Path map

| Step | Focus | Where to study | Honest limit |
|------|--------|----------------|--------------|
| 1 | Python + batch transforms | Module 00, Module 01 | Prefer clear scripts and tests over clever one-liners |
| 2 | SQL + warehouse modeling | **Module 19** | Facts, dimensions, grain, and KPI definitions. OLAP ideas live here |
| 3 | Data quality | [data_validation.md](data_validation.md), Module 20 | Schema checks, null/range rules, basic drift awareness |
| 4 | Serving and packaging | Module 13 | Docker and APIs so pipelines and jobs are runnable |
| 5 | Ops and streaming **survey** | Module 14 | Kafka / Spark names appear so you recognize them. Depth needs separate labs or jobs |
| 6 | Tradeoffs | [System Design for Beginners](../system-design/README.md) | Queues, storage, failure modes. Read before promising “real-time everything” |

## ETL / ELT (minimum practice)

1. Ingest raw files or API extracts into a staging area.
2. Transform into typed tables with documented grain.
3. Load into an analytics-ready schema (warehouse-style).
4. Add at least one automated quality check before “success.”
5. Schedule or document how the job reruns and how failures alert a human.

## Warehouse modeling with Module 19

Use Module 19 for:

- Primary keys, joins, and window functions for metrics
- OLTP vs OLAP mental model
- Dimensional habits (date dimension, fact tables, slowly changing attributes at a basic level)

You do not need every NoSQL engine listed in Module 19. Pick what your project uses.

## Streaming survey (honest limits)

Module 14 may mention Kafka, Spark, or similar tools. Treat that as vocabulary and architecture awareness.

**Good outcome:** You can explain batch vs stream, at-least-once vs exactly-once as ideas, and why most analyst dashboards are fine on a daily batch.

**Not claimed here:** Running a multi-broker Kafka cluster, tuning Spark shuffle, or operating Flink in production.

If a role asks for deep streaming, plan a separate practice project after this path.

## Suggested portfolio spike

| Level | Build | Success look like |
|-------|--------|-------------------|
| Core | Batch ETL into a star-ish schema + validation report | Reproducible run, README with grain and KPIs |
| Stretch | One streaming consumer or cloud queue spike | Short write-up of limits. Not a fake “platform” claim |

## Try next

Open the [Data Engineer](career_roadmap_guide.md#data-engineer) section. Schedule Module 19 and [data_validation.md](data_validation.md) before any Spark deep-dive.
