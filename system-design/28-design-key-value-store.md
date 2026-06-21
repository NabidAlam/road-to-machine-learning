# 28. Design a Distributed Key-Value Store

DynamoDB, Cassandra, ScyllaDB, Riak. Single-node Redis is easy. Once you need 100 TB across hundreds of machines, with no downtime even when a rack dies. That's a different problem.

This is the chapter where most of the earlier chapters lock into place: consistent hashing, replication, CAP, sharding, quorum reads.

## Clarify

| Question             | Example answer              |
| -------------------- | --------------------------- |
| Get/put/delete only? | Yes                         |
| Strong or eventual?  | Tunable per request         |
| Replication?         | Multi-region                |
| Range scans?         | Sorted set later; not in v1 |
| Max value size?      | 1 MB                        |
| Durability target?   | 99.999999999% (11 nines)    |

## Estimate

- **Keys:** 10B
- **Average value size:** 1 KB → 10 TB of data
- **Reads per second peak:** 1M
- **Writes per second peak:** 200K

At 10 TB and 1M QPS, a single machine is not in the conversation. Hundreds of nodes, automatic failover, no read-only periods during maintenance.

## High-level design

```
                    [ client ]
                        |
                        v
                +----------------+
                | Coordinator    |  (any node can act as one)
                +-------+--------+
                        |
        consistent hash | -> N replicas for this key
                        v
        +-------+-------+-------+
        v       v       v       v
   [ Node A ][ Node B ][ Node C ][ ... ]    each node owns a slice of the ring
```

Architecture summary, lifted from the Dynamo paper:
- **Consistent hashing** assigns keys to nodes (Chapter 13).
- **Replication factor N** = 3 typical.
- **Tunable quorum (R, W)** controls consistency.
- **Anti-entropy** repairs replicas in the background.

## Deep dive 1: Partitioning

The keyspace is a ring (Chapter 13). Each node owns multiple **virtual nodes** so load is even when machines come and go.

```
hash(key) -> point on ring -> walk clockwise -> first N nodes own this key
```

Add a node: it takes a slice from each existing node. No global rehash. Same when a node dies.

```python
def replicas_for(key, ring, N=3):
    h = hash(key)
    primary_idx = ring.find_clockwise(h)
    return [ring[(primary_idx + i) % len(ring)] for i in range(N)]
```

## Deep dive 2: Replication and quorum

For each key, you have N replicas. Each request specifies:
- **W** = how many replicas must acknowledge a write
- **R** = how many replicas must respond to a read

If `R + W > N`, you get strong consistency (overlap guarantees at least one fresh replica). Otherwise eventual.

| Profile      | Settings (N=3) | Behavior                             |
| ------------ | -------------- | ------------------------------------ |
| Strong reads | R=2, W=2       | Slower but consistent                |
| Fast writes  | R=1, W=1       | Lowest latency, possibly stale reads |
| Read-mostly  | R=1, W=3       | Cheap reads, writes pay the cost     |

Letting the client choose per-request is the trick that made Dynamo influential.

## Deep dive 3: Handling failures

A replica is down. Three patterns:

1. **Hinted handoff.** Coordinator writes to a temporary node and "hints" that this write belongs to the missing one. When the dead node returns, the hint is replayed.
2. **Read repair.** During a read, if replicas disagree, the coordinator pushes the newest value back to the stale ones.
3. **Anti-entropy / Merkle trees.** Background job compares trees of hashes between replicas. Differences become repair work.

You need all three. Each handles a different failure mode.

## Deep dive 4: Concurrent writes

Two clients write key `X` at the same time, with different values, to different coordinators. Who wins?

Options:

- **Last-writer-wins (LWW).** Use a timestamp. Risky. Clocks drift.
- **Vector clocks.** Each replica keeps a counter per writer. Conflicts are detected (you get back both values; app resolves). Used in Dynamo, Riak.
- **CRDTs.** Mathematical structures that merge cleanly. Great for counters, sets, growing lists.

For a generic KV store, vector clocks or CRDTs are the safer default. Application reconciles on read when conflicts surface.

## Deep dive 5: Storage engine

The on-disk format matters at this scale. Two common engines:

- **LSM tree** (Cassandra, RocksDB, Scylla). Writes go to a memtable, flushed to immutable sorted files (SSTables). Compaction merges them. Great for write-heavy.
- **B-tree** (most SQL DBs, some KV). Updates in place. Great for reads, slower for writes.

LSM trees dominate distributed KV stores because writes go fast and replication is naturally append-only.

## Deep dive 6: Multi-region

For global apps, replicate across regions.

- **Active-passive:** writes go to a home region, reads anywhere. Simple, but failover is painful.
- **Active-active:** writes anywhere. CRDTs or LWW reconcile conflicts. DynamoDB Global Tables and Cassandra cross-DC work this way.

Cross-region replication is asynchronous. Be honest with users: a write in Frankfurt might appear in Sydney 200 ms later. That's CAP-AP (Chapter 17).

## Deep dive 7: Client API

Keep it small. Match what you'd build in Python or Go:

```
PUT(key, value, [r, w])    -> success | conflict
GET(key, [r])              -> value | versions[] | not_found
DELETE(key, [r, w])        -> success
```

For ordered access (range scans, time-series), you'll want a wide-column flavor (Chapter 15) with `(partition_key, clustering_key)`.

## Things to remember

- Consistent hashing decides who owns a key; virtual nodes keep load even.
- Tunable R/W lets each call pick consistency vs latency.
- Failures need hinted handoff + read repair + anti-entropy. All three.
- LWW is cheap and lossy; vector clocks and CRDTs are correct and harder.
- LSM trees on disk; B-trees only for read-heavy single-node engines.
- Multi-region is async by default. Strong consistency across regions is expensive.

## Going deeper

- Dynamo paper (Amazon, 2007). The classic.
- Cassandra and ScyllaDB docs on tunable consistency.
- "Designing Data-Intensive Applications," Chapters 5–7 and 9.
- RocksDB wiki for storage engine details.
- Chapter 13 (Consistent Hashing) and Chapter 17 (CAP) in this course.
