# 29. Design a Distributed Message Queue

Kafka, AWS SQS, Pulsar, Google Pub/Sub. Chapter 19 covered queues from a user's perspective. This one is from the inside: how do you build the queue itself, in a way that survives node failures, scales to millions of messages per second, and never loses a message it claimed to accept?

## Clarify

| Question                  | Example answer                             |
| ------------------------- | ------------------------------------------ |
| Queue or pub/sub or both? | Both. Topics with one or many subscribers |
| Order guarantees?         | Per partition, not globally                |
| Delivery guarantee?       | At-least-once; exactly-once optional       |
| Durability?               | Survive disk and node loss                 |
| Message size?             | 1 MB max                                   |
| Retention?                | 7 days default, configurable               |

## Estimate

- **Topics:** 10K
- **Peak ingress:** 5M messages/sec
- **Avg message:** 1 KB → 5 GB/sec → 400 TB/day
- **Consumers:** 100K connected
- **Retention 7 days:** ~3 PB live storage

The numbers force three decisions: partition aggressively, disk-first storage, and zero-copy networking.

## High-level design

```
producers ---write---> +-------------------+
                       |   Broker fleet    |   each broker handles a subset of partitions
                       +-------------------+
                                |
                                v
                    +-----------------------+
                    |  Per-partition log     |   append-only files on disk
                    |  segments + index      |
                    +-----------+-----------+
                                |
                                v
        consumers ---pull--- track per-consumer offsets
```

Three concepts to internalize: **topic, partition, offset.**

- **Topic** = named stream of messages.
- **Partition** = ordered slice of a topic, the unit of parallelism.
- **Offset** = position of a message inside a partition (monotonic int64).

A consumer reads a partition by remembering its offset. That's it.

## Deep dive 1: The log on disk

A partition is just a directory of files.

```
topic-orders-3/
  00000000000000000000.log   <- segment of messages
  00000000000000000000.index <- sparse offset -> byte position
  00000000000004500000.log
  00000000000004500000.index
  ...
```

Writes are **append-only**. The OS page cache buffers writes; periodic `fsync` flushes to disk. Reads use `sendfile()` to push bytes from disk to socket without copying through user space. That's the "zero-copy" that lets Kafka push gigabits per node.

Retention is per-segment. To drop old messages, delete the oldest segment file. No "delete row" exists.

## Deep dive 2: Partitioning

A topic's parallelism is its partition count. More partitions = more throughput.

Producer chooses a partition:

- By **key** (`hash(user_id) % N`) → same key always lands on the same partition → order preserved per key.
- **Round-robin** if no key → maximum spread.

Each partition has a **leader broker**. All writes for that partition go through the leader.

```python
def partition_for(message, num_partitions):
    if message.key is None:
        return random.randrange(num_partitions)
    return hash(message.key) % num_partitions
```

Repartitioning (changing the count) is painful. Keys move. Pick a partition count larger than you currently need.

## Deep dive 3: Replication

Each partition has N replicas (often 3): one leader and N−1 followers.

```
partition orders-3:
   leader   = broker-12
   followers = broker-04, broker-29
```

Followers continuously fetch from the leader, like a Postgres async replica. The leader maintains an **ISR (In-Sync Replicas)** set. Followers caught up within a small lag.

Producer setting `acks`:
- `acks=0`: fire and forget. Fast, lossy.
- `acks=1`: leader has it. Lose the leader before replication and the message is gone.
- `acks=all`: all ISRs have it. Safe.

For real systems, `acks=all` plus `min.insync.replicas=2`. Anything less and you'll learn about it when a broker dies on a Friday night.

## Deep dive 4: Consumer groups and offsets

A **consumer group** is a set of consumers cooperating to read a topic. Each partition is owned by exactly one consumer in the group at a time.

```
topic orders has 12 partitions
consumer group "analytics" has 4 consumers
  -> each consumer owns 3 partitions
```

When a consumer dies, the group **rebalances** and reassigns partitions.

Offsets are stored centrally (in a special internal topic). Each consumer commits its progress periodically. On restart, it resumes where it left off.

This is what gives the queue **horizontal scale on the read side**: add more consumers (up to the partition count), throughput goes up linearly.

## Deep dive 5: Delivery semantics

- **At-most-once:** consumer commits offset *before* processing. Crash = skip.
- **At-least-once (default):** commit *after* processing. Crash = reprocess. Make consumers idempotent (Chapter 09 idempotency keys).
- **Exactly-once:** transactional producer + transactional commit. Higher cost, narrow use cases (financial pipelines).

In practice: at-least-once + idempotent consumers covers 95% of systems.

## Deep dive 6: Dead-letter queues and back-pressure

Some messages are poison. A bug in the consumer crashes on them forever. After N retries, move them to a **dead-letter topic** (Chapter 19). A human investigates.

Back-pressure: if consumers fall behind, the queue grows. Monitor **consumer lag** (latest offset – committed offset). Alert before disks fill.

## Deep dive 7: Cluster coordination

You need a coordinator: who is the leader of partition 3? Which brokers are alive? Where are offsets stored?

Older Kafka used ZooKeeper. Modern Kafka uses an internal Raft (KRaft). Pulsar uses BookKeeper for storage and ZooKeeper for metadata. The exact tech changes; the role does not. Somebody has to track cluster state with strong consistency.

## Things to remember

- Topic → partition → offset. Master that and the rest follows.
- The on-disk log is append-only, with sparse indexes and zero-copy reads.
- Partition = unit of parallelism for both producers and consumers.
- Replication with `acks=all` and `min.insync=2` is the durable default.
- At-least-once + idempotent consumers solves almost every real problem.
- Monitor consumer lag. It's the heart-rate of your pipeline.

## Going deeper

- "Kafka: The Definitive Guide" (Confluent).
- The original LinkedIn Kafka paper.
- Apache Pulsar docs for the tiered storage variant.
- Jay Kreps, "The Log: What every software engineer should know about real-time data's unifying abstraction."
- Chapter 13 (Consistent Hashing) and Chapter 19 (Message Queues) in this course.
