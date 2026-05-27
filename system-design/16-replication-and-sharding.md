# 16. Replication and Sharding

One database server can only do so much. Eventually you have more data than fits, more queries than it can handle, or you want redundancy in case the machine dies. Two big tools fix this: replication and sharding. They solve different problems and you often use both.

## Replication: copy the same data to multiple machines

The idea: every write goes to a primary node, and is then copied to one or more replica nodes. Reads can hit any of them.

```
                       writes
                          |
                          v
                     [ primary ]
                       /    \
              writes /        \  writes
                    v          v
                [ replica 1 ] [ replica 2 ]
                    ^          ^
              reads |          | reads
                    |          |
                   user      user
```

Why bother:
- **Read scaling**: spread read traffic across replicas.
- **High availability**: if the primary dies, promote a replica.
- **Geographic latency**: put a replica in Europe for European users.
- **Disaster recovery**: a remote replica survives even if your main data center burns down.

## Synchronous vs asynchronous replication

When the primary writes data, when does the replica get it?

**Synchronous**: primary waits for the replica to confirm before saying "write succeeded".
- Safe: no data loss on failover.
- Slow: every write pays the network round trip to the replica.

**Asynchronous**: primary writes locally and continues. Replica catches up in the background.
- Fast: writes don't wait.
- Risky: if the primary dies before the replica catches up, you lose those writes.

Most production systems use **async** for performance, with rare synchronous setups for the most critical data. Some, like Postgres, support a hybrid where at least one replica is synchronous and others are async.

## Single-leader, multi-leader, leaderless

How many primaries are there?

**Single-leader (primary-replica)**: one writer, many readers. Postgres, MySQL, MongoDB all default to this.

**Multi-leader**: multiple writers, replicate to each other. Used in multi-datacenter setups where you want writes in each region. Needs conflict resolution (what if two regions update the same row?).

**Leaderless**: no special writer. The client writes to multiple nodes, reads from multiple nodes, uses quorum (e.g. write to 3, read from 3 of 5 nodes) to figure out the latest. Used by Cassandra and DynamoDB. Complex but very fault-tolerant.

For most apps, single-leader is what you want. It's the simplest mental model and most databases get it right.

## Replication lag

If replication is async, replicas can be seconds behind. This causes a classic bug:

1. User updates their profile photo.
2. App writes to primary, returns success.
3. App immediately reads from a replica to refresh the UI.
4. Replica hasn't gotten the write yet. User sees the old photo. Confusion.

Three common fixes:

- **Read-after-write consistency**: route reads from a user back to the primary for a few seconds after they wrote.
- **Monotonic reads**: a user always reads from the same replica, so they never go backwards.
- **Read from primary** for the critical bits, replicas for the rest.

Application-level routing logic. Not free.

## A quick example: Postgres streaming replication

In Postgres, you set up a replica by pointing it at the primary and letting it replay the WAL (write-ahead log):

```bash
# On the primary, allow replication
echo "wal_level = replica"      >> postgresql.conf
echo "max_wal_senders = 5"      >> postgresql.conf
echo "host replication ..."     >> pg_hba.conf

# On the replica
pg_basebackup -h primary-host -D /var/lib/postgres -U replica_user -P -R
systemctl start postgresql
```

After that, replica streams changes continuously. You can read from it (`SELECT`), you cannot write to it.

## Sharding: split the data across multiple primaries

Replication doesn't help with one problem: your data is too big for one machine. Or your write traffic is too high for one primary to keep up.

Sharding is partitioning. Split the data so each node owns part of it.

```
                       writes for "users A-M"
                                 |
                                 v
                          [ shard 1 ]
                                                            writes for "users N-Z"
                                                                     |
                                                                     v
                                                              [ shard 2 ]
```

Now you have twice the capacity. Add a third shard, you have triple. In theory, this scales forever.

## How to choose a shard key

The shard key is the column you use to decide which shard a row lives on. Pick wrong and you regret it for years.

Three common strategies:

**Range-based**: rows with IDs 1-1M on shard 1, 1M-2M on shard 2, etc.
- Easy to reason about.
- Range scans are fast.
- Can become unbalanced (the newest range gets all the writes).

**Hash-based**: hash the key, modulo by number of shards.
- Even distribution.
- Range scans are slow (data scattered).
- Adding shards reshuffles everything (this is where consistent hashing from Chapter 13 helps).

**Geographic / categorical**: shard by region, by user type, by tenant.
- Natural for multi-tenant apps.
- Risk of one shard being much bigger than others.

The shard key needs to be in almost every query, otherwise you have to query all shards. "Scatter-gather" queries are slow.

## What gets harder when you shard

Almost everything.

**Joins across shards**: if user 42's data is on shard 1 and user 99's is on shard 2, joining them needs cross-shard coordination. Most apps avoid this by denormalizing or doing the join in the app.

**Transactions across shards**: distributed transactions are slow and error-prone (two-phase commit). Many sharded systems just don't support them. You design around it.

**Unique constraints**: each shard can guarantee uniqueness locally. Globally unique IDs need a separate ID-generation service (UUIDs, snowflake IDs, etc.).

**Rebalancing**: when you add a shard, data has to move. Consistent hashing (Chapter 13) makes this less painful.

## Replication and sharding together

Real systems use both. Each shard is replicated for redundancy, and shards together hold the full data.

```
       Shard 1 (users A-M)              Shard 2 (users N-Z)
            primary                        primary
           /      \                       /      \
      replica    replica              replica    replica
```

If Shard 1's primary dies, its replica gets promoted. If you need more capacity, add Shard 3.

This is roughly how MongoDB, Cassandra, DynamoDB, Google Spanner, CockroachDB, and Vitess (sharded MySQL) all work. Architecture is the same idea, details vary.

## When to actually shard

Late. Sharding is a big complexity tax. Before you shard, exhaust the easier options:

1. **Add indexes**. Slow queries are usually missing indexes, not "the DB is too small".
2. **Add caching**. Redis between app and DB takes a huge bite out of read load.
3. **Vertical scale**. Bigger RAM, bigger CPU. Cheap engineering hours.
4. **Read replicas**. Push reads off the primary.
5. **Move some data out**. Move giant blobs to object storage (Chapter 18). Move logs to a separate store.
6. **Then shard**.

Many companies put off sharding until they're forced to. Some never get there. Instagram famously ran on a single sharded Postgres setup for years.

## A note on managed databases

Modern cloud DBs (Aurora, Cloud SQL, DynamoDB, Cosmos DB) hide a lot of this. AWS Aurora gives you up to 15 read replicas with one click. DynamoDB shards automatically. You pay them money, they handle the boring parts.

If you're running a small to mid-sized product, a managed Postgres with one read replica is probably enough for a long time.

## Things to remember

- Replication = same data on multiple nodes. Helps with reads, HA, and DR.
- Sharding = different data on different nodes. Helps when you outgrow one machine.
- Replication is mostly free with managed databases. Sharding is expensive complexity.
- Pick a shard key that's in every query. Or be ready to scatter-gather.
- Use both together: each shard replicated for safety, shards spread for capacity.

## Going deeper

- *Designing Data-Intensive Applications*, Chapters 5 and 6. Definitive.
- "Scaling Instagram Infrastructure" video (QCon): how Instagram sharded Postgres.
- Vitess docs: https://vitess.io/. The system Slack and YouTube use to shard MySQL.
- "Why You Shouldn't Use a Database" by Ben Johnson (LiteFS): https://fly.io/blog/.
- AWS DynamoDB internals paper: https://www.usenix.org/conference/atc22/presentation/elhemali.
