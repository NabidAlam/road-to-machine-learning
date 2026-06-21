# 30. Design a Real-Time ML Feature Store

This chapter is the one you won't see in most system design courses. Since the rest of this repo is about machine learning, it deserves a closing problem that ties system design directly to ML.

A **feature store** is the central place where machine learning features live. A "feature" is a number (or vector) that describes some entity at some moment, `user.avg_purchase_last_7d`, `merchant.fraud_rate_today`, `product.click_count_last_minute`. Models read these at training time and at serving time, and the values **must match**. If they don't, you get *training-serving skew*. The classic "great in the notebook, terrible in prod" bug.

Real systems: Feast, Tecton, Vertex AI Feature Store, Databricks Feature Store, Pinterest's "Galaxy."

## Clarify

| Question                      | Example answer                                  |
| ----------------------------- | ----------------------------------------------- |
| What entities?                | Users, merchants, products                      |
| Real-time or batch features?  | Both                                            |
| How fresh must features be?   | Streaming features within 1 second; batch daily |
| Serving latency target?       | p99 under 10 ms per request                     |
| Backfill historical features? | Yes, for training datasets                      |
| Point-in-time correctness?    | Yes. Critical for training                     |

## Estimate

- **Entities tracked:** 100M users, 10M merchants, 50M products
- **Features per entity:** 200 (mix of fresh + slow)
- **Online reads per second:** 200K (each scoring call reads 50 features)
- **Streaming updates per second:** 500K events fanned into features
- **Daily batch jobs:** ~500 feature pipelines

Two access patterns dominate: **low-latency lookup** (online) and **point-in-time join** (training). You will need two storage layers.

## High-level design

```
                event sources (clicks, payments, app logs)
                            |
                            v
                     +---------------+
                     |    Kafka      |   raw event stream
                     +-------+-------+
                             |
              +--------------+--------------+
              v                             v
     +------------------+         +-------------------+
     | Stream processor |         |  Batch processor  |
     | (Flink/Spark SS) |         |  (Spark/BigQuery) |
     +--------+---------+         +---------+---------+
              |                             |
              v                             v
     +------------------+         +-------------------+
     |  Online store    |         |  Offline store     |
     |  Redis / Aero    |         |  Parquet on S3 /   |
     |  spike / Dynamo  |         |  BigQuery /        |
     |  KV per entity   |         |  Snowflake         |
     +--------+---------+         +---------+---------+
              ^                             ^
              |                             |
  model server                       training jobs
  (real-time inference)              (build datasets, train models)
```

Two stores, **one feature definition**. Both are populated from the same source events. Same code path or you get skew.

## Deep dive 1: Online vs offline store

**Online store** answers "give me the current features for user 42."

- Storage: Redis, DynamoDB, Aerospike, Cassandra. Anything with sub-10-ms reads.
- Layout: `key = (entity_id, feature_view)`, value = serialized struct.
- Eviction: rarely; you keep recent features warm forever.

**Offline store** answers "give me what features looked like at timestamp T for these 10 million users."

- Storage: columnar on object store (Parquet/Iceberg/Delta), or a data warehouse (BigQuery, Snowflake, Redshift).
- Layout: `(entity_id, event_timestamp, feature1, feature2, ...)`.
- Reads scan partitions by date and entity.

## Deep dive 2: Streaming features

Streaming features change second-to-second. Example: `user.clicks_in_last_5_min`.

```
event = { user_id: 42, type: "click", t: now }
   |
   v
[ Flink job ]
   - keyed by user_id
   - sliding window 5 min, slide 10 s
   - emit count -> sink
   |
   v
   online store: HSET feat:user:42 clicks_5m 17
   offline store: append (user_id=42, event_time=t, clicks_5m=17) to Parquet
```

The same Flink/Spark job writes to **both stores**. Same value, same timestamp. That's how training and serving stay aligned.

## Deep dive 3: Batch features

Batch features run on a schedule. Nightly, hourly.

```
00:00 UTC nightly:
  SELECT user_id,
         AVG(amount) AS avg_purchase_30d,
         COUNT(*)    AS purchases_30d
  FROM   transactions
  WHERE  t >= NOW() - INTERVAL '30 days'
  GROUP  BY user_id
  -> write to online store (Redis HSET)
  -> append to offline store as a new daily partition
```

Schedule with Airflow/Dagster/Argo. Write both stores in the same job.

## Deep dive 4: Point-in-time correctness

The single hardest thing in this design.

A model is trained on labels from January. For each label, you need features **as they were at that moment**, not as they are now. If you blindly join "today's features" to "January's labels," your model leaks the future and gets 99% accuracy in training and 60% in production.

The query you actually want:

```sql
-- for each label, fetch the latest feature row whose event_time <= label_time
SELECT l.label, f.*
FROM labels l
LEFT JOIN LATERAL (
  SELECT *
  FROM features
  WHERE entity_id = l.entity_id
    AND event_time <= l.event_time
  ORDER BY event_time DESC
  LIMIT 1
) f ON TRUE;
```

This is the **point-in-time join**. Feast and Tecton give it to you as a one-liner. If you build your own, you must implement it correctly from day one. It's the single technical reason feature stores exist.

## Deep dive 5: Feature definitions as code

Same definition, two paths. Looks roughly like:

```python
from feature_store import Entity, FeatureView, Field, types

user = Entity(name="user")

user_clicks_5m = FeatureView(
    name="user_clicks_5m",
    entity=user,
    schema=[Field("clicks_5m", types.Int32)],
    online=True,
    source=KafkaSource(topic="clicks", timestamp="t"),
    transformation=window_count("user_id", "5m"),
    ttl="1d",
)
```

The framework generates:
- A Flink job for streaming.
- A Spark job for batch backfill.
- A serving endpoint that reads from Redis with the right schema.

Without this single source of truth, training and serving will drift apart in three months.

## Deep dive 6: Serving the model

The model server gets a request:

```
POST /score
{ "user_id": 42, "merchant_id": 998 }
```

Flow:
1. Look up features for `user:42` and `merchant:998` in the online store. One Redis pipeline call. Under 5 ms.
2. Concatenate features in the order the model expects.
3. Run inference (separate model server. Triton, BentoML, vLLM, whatever).
4. Return prediction + the feature snapshot used (for debugging, drift detection, and audit).

Step 4 is underrated. When the model regresses, you want to know *which feature went weird*.

## Deep dive 7: Monitoring

Watch four things:

1. **Freshness:** is the streaming pipeline keeping up? Lag in seconds per feature view.
2. **Schema drift:** new categories, nulls increasing, value ranges shifting.
3. **Online–offline parity:** sample 1% of online reads, replay through the offline definition, compare. Alert if they diverge by more than a tolerance.
4. **Serving latency:** p99 of feature reads.

If parity breaks, **stop training new models** until it's fixed. Bad parity = bad models, silently.

## Things to remember

- A feature store exists because training-serving skew kills models. Solve that, the rest follows.
- Two stores, one definition. Online for fast lookups, offline for training.
- Point-in-time joins are non-negotiable. Build or buy, but use them.
- The same streaming job writes both stores. Same code, same timestamp.
- Return the feature snapshot with every prediction. You'll thank yourself in debugging.
- Monitor freshness, drift, parity, and latency separately. One number isn't enough.

## Going deeper

- [Feast](https://feast.dev): open-source feature store, great place to read code.
- Tecton's blog: clear writing on point-in-time correctness and orchestration.
- "Feature stores: the missing data layer for ML" by various authors.
- [ML System Design Guide](../resources/ml_system_design_guide.md): broader treatment of serving and monitoring.
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen. Chapter on feature engineering.
- Pinterest's "Galaxy" engineering blog posts.
