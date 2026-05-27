# 14. SQL

SQL databases are the workhorse of software. If you're starting a project today and have no special reason to do otherwise, you should pick Postgres. The rest of this chapter is about why.

## The relational model in one paragraph

Data lives in **tables**. A table has **columns** (fields) and **rows** (records). Tables relate to each other through **foreign keys**. You query with declarative statements (SQL) that say what you want, not how to get it.

```sql
SELECT u.name, COUNT(p.id) AS post_count
FROM users u
LEFT JOIN posts p ON p.author_id = u.id
GROUP BY u.name
ORDER BY post_count DESC
LIMIT 10;
```

That gets you the top 10 most prolific users. The database figures out how. You don't tell it which indexes to use, what order to read the rows in, or how to combine the tables. That's the engine's job.

## ACID: the promises a SQL DB makes

The defining feature of SQL databases is that they take **ACID** seriously.

- **Atomicity**: a transaction either fully happens or fully doesn't. If you debit one account and credit another, both succeed or both fail.
- **Consistency**: the database moves from one valid state to another. Constraints (foreign keys, unique, NOT NULL) hold.
- **Isolation**: concurrent transactions don't see each other's intermediate state. Two transfers running at the same time don't corrupt each other.
- **Durability**: once a transaction commits, it's on disk. A crash doesn't undo it.

This is why banks use SQL. ACID is "money doesn't vanish".

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

If the server crashes between the two updates, the transaction rolls back. Either both happen on disk or neither does.

## Schema

SQL is strict about schema. You declare what a table looks like upfront:

```sql
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE posts (
    id          SERIAL PRIMARY KEY,
    author_id   INTEGER NOT NULL REFERENCES users(id),
    title       TEXT NOT NULL,
    body        TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_posts_author ON posts(author_id);
```

Why bother:
- The DB enforces invariants. Can't accidentally insert a post with a non-existent author.
- Queries can use indexes the DB knows are safe.
- Anyone reading the schema can understand the data model in 30 seconds.

The cost: schema migrations. Every change ships in a migration file. Adding a column, renaming, dropping. Tools like Liquibase, Flyway, Alembic, or `db-migrate` track them in a version-control style.

## Joins

The killer feature of SQL is the join. You don't denormalize. You keep data in its natural form and stitch it together at query time.

```sql
SELECT
    p.title,
    p.created_at,
    u.name AS author,
    COUNT(c.id) AS comment_count
FROM posts p
JOIN users u ON u.id = p.author_id
LEFT JOIN comments c ON c.post_id = p.id
WHERE p.created_at > NOW() - INTERVAL '7 days'
GROUP BY p.id, u.name
ORDER BY comment_count DESC
LIMIT 20;
```

In a NoSQL store, you'd be doing several round trips and merging in your app. SQL does it in one shot.

## Indexes

An index is a pre-sorted lookup structure (usually a B-tree) that makes one type of query fast.

```sql
CREATE INDEX idx_users_email ON users(email);
```

After this, `SELECT * FROM users WHERE email = '...'` goes from a sequential scan (read every row) to a logarithmic lookup. On a 10 million row table, the difference is ~10 million reads vs ~24.

The trade-off: every write has to update every index. Too many indexes and writes get slow.

Rule of thumb: index columns you filter or join on, not columns you only display.

Use `EXPLAIN` to see how the DB plans a query:

```sql
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'ada@example.com';
```

If the plan says `Seq Scan` on a table with millions of rows, you forgot an index.

## Transactions and isolation levels

Even within ACID, there's a knob: how strictly do transactions hide from each other?

| Level            | Anomalies allowed                    | Notes                  |
| ---------------- | ------------------------------------ | ----------------------  |
| Read uncommitted | Dirty reads                          | Almost no one uses this |
| Read committed   | Non-repeatable reads, phantom reads  | Default in Postgres    |
| Repeatable read  | Phantom reads                        | Default in MySQL       |
| Serializable     | None                                 | Most strict, slowest   |

The looser levels are faster because the DB does less locking. The stricter levels are safer. Most apps run on read committed and rarely think about it. Financial systems often crank to serializable for the critical operations.

```sql
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- ... work ...
COMMIT;
```

## When SQL stops being enough

A single Postgres can comfortably handle:
- A database of hundreds of GB to a few TB.
- Tens of thousands of writes per second.
- Hundreds of thousands of reads per second (with replicas).

Most products never outgrow this. When they do, you reach for:

- **Read replicas**: ship reads to follower databases.
- **Sharding**: partition the data by some key across many primaries.
- **Caching**: don't hit the DB for hot reads (Redis).
- **Specialized stores**: blob storage for files, search indexes for full text, time-series DBs for metrics.

We'll cover replication and sharding in detail in Chapter 16.

## Which SQL?

The mainstream choices:

| Database   | Strengths                                          | Weakness                               |
| ---------- | -------------------------------------------------- | -------------------------------------- |
| Postgres   | Best feature set, JSONB, extensions, sane defaults | Slightly more memory per connection    |
| MySQL      | Huge ecosystem, simple, fast                       | Quirky behaviors, weaker SQL standards |
| SQLite     | Single file, embedded, zero config                 | Single-writer                          |
| SQL Server | Microsoft ecosystem, strong tooling                | Licensing                              |
| Oracle     | Enterprise features                                | Cost                                   |

For most projects in 2026, **Postgres**. SQLite is genuinely amazing for small projects (your phone uses it). MySQL is fine if you have a reason.

## Code: connecting from an app

Python with `psycopg`:

```python
import psycopg

conn = psycopg.connect(
    host="localhost", dbname="myapp", user="me", password="..."
)
with conn.cursor() as cur:
    cur.execute("SELECT id, name FROM users WHERE email = %s", ("ada@example.com",))
    row = cur.fetchone()
    print(row)
```

Never concatenate user input into SQL strings. Use parameters (`%s`) and let the driver handle escaping. That's how SQL injection happens.

For real apps, you'll likely use an ORM (SQLAlchemy, Django ORM, Prisma, ActiveRecord). They give you objects instead of rows but generate SQL behind the scenes. Learn SQL anyway. ORMs leak.

## A few habits that save you later

- **Always have a primary key**. Usually `id SERIAL` or `UUID`.
- **Foreign keys with `ON DELETE` policy**. Decide: CASCADE, SET NULL, RESTRICT.
- **Timestamps on every table**. `created_at`, `updated_at`. You'll want them eventually.
- **Migrations in version control**. Never hand-edit production schemas.
- **Indexes on foreign keys**. Postgres doesn't add them automatically.
- **`EXPLAIN ANALYZE`** before declaring victory on a query.

## Things to remember

- SQL = relational tables + ACID + declarative queries + strict schema.
- Postgres is the sensible default in 2026.
- Joins let you keep data normalized and combine it at query time.
- Indexes speed up reads but slow down writes. Index what you filter on.
- Most products never need to shard. Reach for replicas and caching first.

## Going deeper

- Postgres docs: https://www.postgresql.org/docs/. Treat as a reference.
- *Designing Data-Intensive Applications*, Chapters 2 and 3.
- *SQL Antipatterns* by Bill Karwin. The mistakes you didn't know you were making.
- "Use the Index, Luke!" by Markus Winand: https://use-the-index-luke.com/. Best free book on indexing.
- ByteByteGo's videos on SQL performance.
