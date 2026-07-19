# Phase C: SQL and Relational Design (In-Repo Lessons)

**Maps to:** [Blueprint Phase C](../full_stack_ai_engineer_roadmap.md#phase-c-database-and-sql-mastery)

**Goal:** Model data with keys and relationships, normalize far enough to avoid painful anomalies, and write predictable queries.

---

## Table of Contents

- [Lesson 1: Tables, keys, and relationships](#lesson-1-tables-keys-and-relationships)
- [Lesson 2: Normalization in practice](#lesson-2-normalization-in-practice)
- [Lesson 3: Query building blocks](#lesson-3-query-building-blocks)
- [Lesson 4: Joins and aggregation](#lesson-4-joins-and-aggregation)
- [Lesson 5: Transactions (when they matter)](#lesson-5-transactions-when-they-matter)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1: Tables, keys, and relationships

A **relation** (table) stores tuples (rows) with attributes (columns). A **primary key** uniquely identifies a row. A **foreign key** references another table and enforces integrity at the database boundary.

Cardinality:

- **One-to-many:** `users` → `posts` (`posts.user_id` → `users.id`)
- **Many-to-many:** use a **junction** table (`post_tags` between `posts` and `tags`)

**Exercise:** Draw three tables for a blog: `users`, `posts`, `comments`. Label PKs and FKs.

---

## Lesson 2: Normalization in practice

**1NF:** atomic columns (no repeating groups hidden inside one column).  
**2NF:** remove partial dependencies on part of a composite key.  
**3NF:** remove transitive dependencies (non-key fields should not depend on other non-key fields).

You do not always normalize “perfectly” for analytics warehouses, but for OLTP product databases, **3NF is a strong default**.

**Exercise:** Given a denormalized `orders` row that embeds `customer_name`, split into `customers` and `orders` with a FK.

---

## Lesson 3: Query building blocks

- Filtering: `WHERE`, `AND`, `OR`, `IN`, `BETWEEN`, `LIKE` / `ILIKE`
- Sorting: `ORDER BY`
- Projection: choose columns intentionally (avoid `SELECT *` in hot paths)

**NULL handling:** `COALESCE(column, default)` for display defaults; remember `NULL` comparisons are tricky (`IS NULL`).

**Exercise:** Write a query that lists the 10 newest posts with author name using a `JOIN`.

---

## Lesson 4: Joins and aggregation

- **INNER JOIN:** only matching rows
- **LEFT JOIN:** keep left rows even if no match (watch for `NULL` columns)
- **GROUP BY** + aggregates (`COUNT`, `SUM`, `AVG`) with **`HAVING`** to filter aggregated results

**Pagination:** `LIMIT` / `OFFSET` is simple; keyset pagination is often better at scale (Phase E/D revisit).

**Exercise:** Count posts per user; only include users with at least 2 posts.

---

## Lesson 5: Transactions (when they matter)

Use a transaction when multiple writes must succeed or fail together (money movement, creating parent + children rows).

Conceptual pattern: `BEGIN` → statements → `COMMIT` or `ROLLBACK` on error.

---

## Exercises

1. Create schema for `events(user_id, type, created_at)` and add indexes that match your query patterns.
2. Write a migration-style SQL file that adds a nullable column safely, backfills, then sets `NOT NULL`.
3. Identify one query in your app that could cause an N+1 pattern and rewrite it with a join.

---

## Next step

[Phase D: Prisma & Next.js](phase-d-prisma-nextjs.md)
