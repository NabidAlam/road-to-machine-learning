# Advanced SQL Topics

Advanced SQL techniques for data science.

## Table of Contents

- [Advanced Window Functions](#advanced-window-functions)
- [Query Optimization](#query-optimization)
- [Indexes](#indexes)
- [Transactions](#transactions)
- [Common Pitfalls](#common-pitfalls)

---

## Advanced Window Functions

### Frames

```sql
-- Rows between
SELECT 
    first_name,
    salary,
    SUM(salary) OVER (
        ORDER BY employee_id 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM employees;
```

### Percentiles

```sql
SELECT 
    first_name,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary) AS percentile_rank
FROM employees;
```

---

## Query Optimization

### EXPLAIN

```sql
EXPLAIN SELECT * FROM employees WHERE department_id = 1;
```

### Indexes

```sql
-- Create index
CREATE INDEX idx_department ON employees(department_id);

-- Composite index
CREATE INDEX idx_name_dept ON employees(first_name, department_id);
```

---

## Key Takeaways

1. **Optimize Queries**: Use indexes and EXPLAIN
2. **Window Functions**: Powerful for analytics
3. **Transactions**: Ensure data consistency

---

**Remember**: Advanced SQL skills make you a better data scientist!

