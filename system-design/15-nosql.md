# 15. NoSQL

"NoSQL" is a confusing label. It just means "not the traditional relational SQL database". That covers a dozen very different things. Calling MongoDB, Redis, and Cassandra all "NoSQL" is like calling a hammer, a screwdriver, and a chainsaw all "hand tools". Technically true, completely unhelpful.

Let's break it apart by what these databases actually do.

## The four flavors

### Key-value

The simplest. A giant hash map: key in, value out. Values are opaque blobs the DB doesn't understand.

- **Examples**: Redis, Memcached, DynamoDB (KV mode).
- **Strengths**: blazingly fast, easy to scale horizontally.
- **Use for**: caching, session storage, simple lookups, rate limiters.

```redis
SET user:42 "{\"name\":\"Ada\",\"plan\":\"pro\"}"
GET user:42
EXPIRE user:42 3600
```

That's the whole API, basically.

### Document

Like key-value, but the value is a structured document (usually JSON). The DB understands the structure and can query inside it.

- **Examples**: MongoDB, Couchbase, Firebase Firestore, AWS DocumentDB.
- **Strengths**: flexible schema, natural fit for object-oriented data, easy onboarding.
- **Use for**: content management, product catalogs, user profiles, anything with nested structure.

```javascript
db.users.insertOne({
  name: "Ada",
  email: "ada@example.com",
  preferences: {
    theme: "dark",
    notifications: ["email", "push"]
  },
  posts: [
    { title: "Hello", views: 42 },
    { title: "Day 2", views: 17 }
  ]
});

db.users.find({ "preferences.theme": "dark" });
```

You can query nested fields directly. Try doing that cleanly in SQL. (You can, with JSONB in Postgres, but it's not as ergonomic.)

### Wide-column

Tables, but with flexible columns. Each row can have different columns. Optimized for huge volumes of writes and analytical queries across many rows.

- **Examples**: Cassandra, ScyllaDB, HBase, Bigtable.
- **Strengths**: enormous scale (petabytes), high write throughput, multi-datacenter from day one.
- **Use for**: time-series data, event logs, user activity tracking, IoT.

Cassandra's data model:

```cql
CREATE TABLE events (
    user_id     UUID,
    event_time  TIMESTAMP,
    event_type  TEXT,
    payload     TEXT,
    PRIMARY KEY (user_id, event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);

INSERT INTO events (user_id, event_time, event_type, payload)
VALUES (uuid(), now(), 'login', '{"ip":"1.2.3.4"}');

SELECT * FROM events WHERE user_id = ? LIMIT 50;
```

Note the schema looks SQL-ish. But the engine underneath is completely different: no joins, no transactions in the SQL sense, partitioning baked in.

### Graph

Data is nodes and edges. Queries traverse relationships.

- **Examples**: Neo4j, Amazon Neptune, ArangoDB.
- **Strengths**: relationship-heavy queries (who knows who, recommendation engines, fraud detection).
- **Use for**: social networks, recommendation systems, knowledge graphs.

Cypher query in Neo4j:

```cypher
MATCH (me:Person {name: "Ada"})-[:FRIEND*2]-(friend_of_friend)
WHERE NOT (me)-[:FRIEND]-(friend_of_friend)
RETURN friend_of_friend.name
LIMIT 10
```

"Find friends-of-friends I'm not yet friends with." Try doing that in SQL with multiple joins on a 10-million-row graph. It's painful. In a graph DB it's natural and fast.

## Why use NoSQL at all

SQL is great. So why does NoSQL exist? Three reasons usually:

### 1. Schema flexibility

In SQL, adding a column to a 1-billion-row table is a migration. In a document DB, you just start writing the new field. Old documents don't have it; new ones do; the app handles both.

This sounds great. It's also dangerous. Schema enforcement saves you from yourself. NoSQL pushes that responsibility to the app code, where it's easy to forget.

### 2. Horizontal scale by default

Postgres can be replicated and sharded, but it takes work. Cassandra and DynamoDB are sharded out of the box. You add nodes, they pick up data. This matters once you're past the single-machine limit.

### 3. Specific data shapes

If you have a graph, a graph DB is a better fit. If you have time series, a time-series DB is. If you have a giant key-value workload, a KV store is. Trying to fit those into SQL works but isn't always optimal.

## What you give up

NoSQL almost always trades something for those gains:

- **Joins**: most NoSQL DBs don't support them. You denormalize and accept duplicated data.
- **Transactions**: many NoSQL stores don't support multi-document transactions, or support them with caveats.
- **Strong consistency**: many are "eventually consistent" by default (we'll see CAP next chapter).
- **Mature query language**: SQL has 50 years of polish. NoSQL query languages vary.

For most apps, SQL gives you all of this and you're fine. Reach for NoSQL when you have a specific reason.

## A common middle ground: SQL with JSON

Postgres has the `JSONB` type. You can store schemaless data inside a SQL row and query it with operators:

```sql
CREATE TABLE products (
    id      SERIAL PRIMARY KEY,
    name    TEXT,
    details JSONB
);

INSERT INTO products (name, details)
VALUES ('Phone', '{"brand": "Acme", "specs": {"ram": "8GB"}}');

SELECT * FROM products
WHERE details->>'brand' = 'Acme'
  AND details->'specs'->>'ram' = '8GB';
```

For many "I want flexibility" cases, this beats reaching for MongoDB. You get ACID, joins, and indexes, plus the schemaless bits where you want them.

## When to pick what (rough guide)

| You're building...                               | Try                             |
| ------------------------------------------------ | ------------------------------- |
| A normal product with user data, orders, content | Postgres                        |
| A cache or session store                         | Redis                           |
| A high-volume event log or metrics store         | Cassandra or ClickHouse         |
| A content / catalog with flexible fields         | Postgres with JSONB, or MongoDB |
| A social graph, recommendations                  | Neo4j or graph extension        |
| Globally distributed key-value at huge scale     | DynamoDB, Spanner               |
| Anything where you're unsure                     | Start with Postgres             |

The number of teams who picked MongoDB on day one and regretted it is large. The number who picked Postgres and regretted it is small. Bias toward the boring answer.

## A real example

Imagine you're building a fitness tracker.

- **User accounts, friends, plans**: Postgres. Relational data, transactions when you upgrade plans.
- **Workout events** (every rep, every heart rate sample): time series, billions of rows. Cassandra or a time-series DB.
- **Session tokens, leaderboard cache**: Redis. Hot reads, simple keys.
- **Profile photos**: object storage (S3, Chapter 18).
- **"People who did this workout also did..."**: graph DB or precomputed in SQL.

That's a polyglot architecture. Most real systems are. The trick is not picking one DB to rule them all, it's picking the right one for each shape of data.

## Things to remember

- "NoSQL" is four different things: key-value, document, wide-column, graph.
- They trade joins, transactions, or consistency for scale or flexibility.
- Postgres with JSONB covers most "I want flexibility" cases.
- Don't reach for NoSQL because it's cool. Reach for it because you have a specific reason.
- Real systems are polyglot: SQL for transactions, KV for cache, time-series for events, object store for files.

## Going deeper

- *Designing Data-Intensive Applications*, Chapter 2 in particular. Best summary of NoSQL trade-offs anywhere.
- "NoSQL Distilled" by Pramod Sadalage and Martin Fowler. Short and clear.
- MongoDB University free courses: https://learn.mongodb.com/.
- Cassandra docs: https://cassandra.apache.org/doc/latest/.
- DynamoDB single-table design (a deep rabbit hole): https://www.alexdebrie.com/posts/dynamodb-single-table/.
