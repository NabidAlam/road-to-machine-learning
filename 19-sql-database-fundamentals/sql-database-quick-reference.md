# SQL Quick Reference

Quick SQL reference guide.

## Basic Queries

```sql
SELECT * FROM table;
SELECT col1, col2 FROM table WHERE condition;
INSERT INTO table VALUES (val1, val2);
UPDATE table SET col = val WHERE condition;
DELETE FROM table WHERE condition;
```

## Joins

```sql
-- INNER JOIN
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id;

-- LEFT JOIN
SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id;
```

## Aggregations

```sql
SELECT col, COUNT(*), AVG(num) 
FROM table 
GROUP BY col 
HAVING COUNT(*) > 5;
```

---

**Try next:** Do ten queries on a public dataset today: filters, joins, group by, window.

