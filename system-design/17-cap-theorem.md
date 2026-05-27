# 17. CAP Theorem

Once you have data spread across multiple machines, you bump into a fact of life that no engineering can avoid. It's called the CAP theorem. It's small. It's important. People get it wrong constantly.

## The three letters

In a distributed system, you have three properties:

- **C - Consistency**: every read sees the most recent write. Everyone sees the same data.
- **A - Availability**: every request gets a response, even if some nodes are down.
- **P - Partition tolerance**: the system keeps working even when the network between nodes is broken.

The theorem says: **you can have any two, but not all three at once during a network partition**.

## What's a partition

A network partition is when some nodes can't talk to others, even though both are up. Maybe a cable failed. Maybe a region lost connectivity. Maybe the load balancer dropped them. From inside, each side sees the other as "down".

```
            +---------+   X   +---------+
            | Node A  |---x---| Node B  |
            +---------+       +---------+
                  (X = network broken)
```

In a real distributed system, partitions happen. Not "if". "When". They're rare on a single-data-center network. They're common across continents. So P is non-negotiable.

That leaves you choosing between C and A, **during the partition**.

## CP: prefer consistency

If the network is partitioned, refuse to serve queries that might return stale data. Better to return an error than the wrong answer.

```
   user -> Node A (says "I can't reach the others, sorry, error")
```

Used by: banking systems, etcd, ZooKeeper, Spanner (Google's globally-consistent DB).

When this is right: money. Inventory. Locks. Anything where a wrong answer costs you.

## AP: prefer availability

If the network is partitioned, keep serving. Some users might see stale data, but the system stays up.

```
   user -> Node A (says "here's my best guess, it might be stale")
```

Used by: DNS, Cassandra, DynamoDB (default mode), most CDNs, most social media products.

When this is right: a "like" count being briefly wrong is fine. A web page being unreachable is not.

## The CAP triangle (a famous picture)

```
                       C
                      /
                     /
                    /
                  CP --- CA --- AP
                              \
                               \
                                A
                                |
                                P
```

The "CA" corner is what single-machine systems give you. As soon as you go distributed, P is forced on you. The corner becomes a choice between CP and AP.

## A real example

Two database nodes, replicated. User updates their profile photo on Node A. Network breaks before Node B sees it.

```
  Step 1:  user -> A: "set photo to X"
           A confirms.

  Step 2:  network breaks.

  Step 3:  user reads -> B
           Does B return the old photo (AP) or refuse (CP)?
```

- **AP** (Cassandra, Dynamo default): B returns the old photo. User is briefly confused. Some day later, the systems sync. "Eventually consistent."
- **CP** (Spanner, etcd): B refuses to serve a read. User sees an error.

Both are reasonable engineering decisions for different systems.

## PACELC: the more accurate version

CAP only talks about partitions. The Yale CS professor Daniel Abadi pointed out: even when the network is fine, there's still a trade-off. It's called PACELC.

> **If Partition, then Availability or Consistency. Else, Latency or Consistency.**

In other words: even with a healthy network, getting strong consistency costs latency (more nodes to talk to, more rounds of coordination). And to get low latency, you might serve a slightly stale read.

| System                 | P →                                    | E →                                                 |
| ---------------------- | -------------------------------------- | --------------------------------------------------- |
| DynamoDB (default)     | AP                                     | EL (favor latency over consistency in normal times) |
| Cassandra              | AP                                     | EL                                                  |
| Spanner                | CP                                     | EC                                                  |
| MongoDB                | CP-ish                                 | EC                                                  |
| Postgres replica reads | CP (with primary) or AP (with replica) | depends on config                                   |

PACELC is the better mental model. CAP is what people memorize.

## What "eventually consistent" really means

This phrase shows up everywhere. It means: if no new writes happen, every replica will eventually agree. "Eventually" is unbounded. In practice it's milliseconds, sometimes seconds.

```
  Write to Node A: photo = X     (everyone else still says photo = old)
  10 ms later:    Node B has it.
  50 ms later:    Node C has it.
  200 ms later:   Everyone in the cluster agrees.
```

This is fine for likes, view counts, follower counts. Not fine for bank balances.

Some systems offer **read-your-own-writes** consistency: after you write, your own subsequent reads see the new value, even if other users haven't yet. A useful middle ground.

## Strong consistency models

If you want CP, there's a hierarchy of how strict:

- **Linearizable**: the strongest. Reads see writes in real-time order. Every operation looks atomic globally.
- **Sequential consistency**: all nodes see operations in the same order, but maybe not the real-time order.
- **Causal consistency**: operations that depend on each other are ordered. Independent ones can be reordered.

The stronger you go, the slower the system. Linearizable is what etcd and ZooKeeper sell. Causal is what some social networks use to make sure "Alice posted, Bob replied" is never seen in the wrong order.

## How real systems actually work

Most systems are **tunable**. Cassandra lets you choose, per query, how many replicas have to agree:

```
   QUORUM   = strict, slower
   ONE      = fast, weak consistency
   ALL      = strongest, slowest, fragile to failures
```

A typical write/read combo:

```
   write to QUORUM, read from QUORUM
```

QUORUM means a majority of replicas (e.g. 2 out of 3). If both writes and reads use QUORUM, you're guaranteed to read your own writes. This is the practical compromise most apps land on.

## A practical mental model

When you're picking a database or designing a system:

1. **Will partitions happen?** Yes. So you choose C or A.
2. **What does wrong data cost?** If it's money, pick C. If it's a like count, pick A.
3. **What does downtime cost?** If the product can't be down at all, pick A.
4. **Even without partitions, do you care more about latency or strict ordering?** That's the PACELC half.

You almost never have to pick "the world's most consistent database". You usually have to pick "what should this part of the system do under this kind of failure?". A typical app uses Postgres (CP) for the orders table, Redis (AP-ish, cache) for sessions, and an object store (AP) for blobs.

## Things to remember

- CAP: under a network partition, you can have consistency or availability. Not both.
- Partition tolerance isn't optional in a distributed system. Network breaks happen.
- AP = stay up, maybe serve stale data. Used everywhere on the web.
- CP = stop serving if not sure. Used for money and critical state.
- PACELC is the better version: even without partitions, there's a latency vs consistency trade-off.
- "Eventually consistent" means all replicas agree, given enough time and no writes.

## Going deeper

- Eric Brewer's original CAP paper, and his "12 years later" retrospective.
- Daniel Abadi on PACELC: http://www.cs.umd.edu/~abadi/papers/abadi-pacelc.pdf.
- *Designing Data-Intensive Applications*, Chapter 9 (Consistency and Consensus).
- Martin Kleppmann's lectures on distributed systems: https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB.
- Jepsen analyses: https://jepsen.io/. Empirical testing of distributed systems. Eye-opening.
