# SQL and Database Fundamentals Complete Guide

Comprehensive guide to SQL and database management for data science.

## Table of Contents

- [Introduction to Databases](#introduction-to-databases)
- [Database Fundamentals](#database-fundamentals)
- [SQL DDL Commands](#sql-ddl-commands)
- [SQL DML Commands](#sql-dml-commands)
- [SQL Joins](#sql-joins)
- [Subqueries](#subqueries)
- [Window Functions](#window-functions)
- [Advanced SQL Topics](#advanced-sql-topics)
- [SQL with Python](#sql-with-python)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Databases

### What is a Database?

A database is an organized collection of data stored and accessed electronically.

**Why Databases?**
- **Data Persistence**: Store data permanently
- **Data Integrity**: Ensure data consistency
- **Efficient Access**: Fast retrieval and updates
- **Concurrent Access**: Multiple users simultaneously
- **Security**: Access control and permissions

### Types of Databases

1. **Relational Databases (SQL)**
   - Structured data in tables
   - Examples: MySQL, PostgreSQL, SQLite
   - Use SQL (Structured Query Language)

2. **NoSQL Databases**
   - Flexible schema
   - Examples: MongoDB, Cassandra, Redis
   - Different query languages

### Database Management System (DBMS)

Software that manages databases:
- **MySQL**: Popular open-source RDBMS
- **PostgreSQL**: Advanced open-source RDBMS
- **SQLite**: Lightweight, file-based
- **SQL Server**: Microsoft's RDBMS
- **Oracle**: Enterprise RDBMS

---

## Database Fundamentals

### Key Concepts

**Tables**: Collections of related data organized in rows and columns
**Rows (Records)**: Individual data entries
**Columns (Fields)**: Data attributes
**Primary Key**: Unique identifier for each row
**Foreign Key**: Reference to another table's primary key
**Index**: Improves query performance

### Database Relationships

1. **One-to-One**: One record in Table A relates to one record in Table B
2. **One-to-Many**: One record in Table A relates to many records in Table B
3. **Many-to-Many**: Many records in Table A relate to many records in Table B

### CRUD Operations

- **Create**: INSERT data
- **Read**: SELECT data
- **Update**: UPDATE data
- **Delete**: DELETE data

---

## SQL DDL Commands

DDL (Data Definition Language) - Define database structure.

### CREATE DATABASE

```sql
-- Create a new database
CREATE DATABASE company_db;

-- Use the database
USE company_db;
```

### CREATE TABLE

```sql
-- Create a table
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10, 2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### Data Types

**Numeric:**
- `INT`: Integer
- `DECIMAL(p, s)`: Fixed-point number
- `FLOAT`: Floating-point number

**String:**
- `VARCHAR(n)`: Variable-length string
- `CHAR(n)`: Fixed-length string
- `TEXT`: Long text

**Date/Time:**
- `DATE`: Date (YYYY-MM-DD)
- `TIME`: Time (HH:MM:SS)
- `DATETIME`: Date and time
- `TIMESTAMP`: Auto-updating timestamp

**Other:**
- `BOOLEAN`: True/False
- `BLOB`: Binary large object

### ALTER TABLE

```sql
-- Add a column
ALTER TABLE employees ADD COLUMN phone VARCHAR(20);

-- Modify a column
ALTER TABLE employees MODIFY COLUMN salary DECIMAL(12, 2);

-- Drop a column
ALTER TABLE employees DROP COLUMN phone;

-- Rename a column
ALTER TABLE employees RENAME COLUMN salary TO annual_salary;
```

### DROP TABLE

```sql
-- Drop a table
DROP TABLE employees;

-- Drop if exists (safe)
DROP TABLE IF EXISTS employees;
```

---

## SQL DML Commands

DML (Data Manipulation Language) - Manipulate data.

### INSERT

```sql
-- Insert single row
INSERT INTO employees (first_name, last_name, email, hire_date, salary)
VALUES ('John', 'Doe', 'john.doe@example.com', '2024-01-15', 75000.00);

-- Insert multiple rows
INSERT INTO employees (first_name, last_name, email, hire_date, salary)
VALUES 
    ('Jane', 'Smith', 'jane.smith@example.com', '2024-02-01', 80000.00),
    ('Bob', 'Johnson', 'bob.johnson@example.com', '2024-02-15', 70000.00);

-- Insert from another table
INSERT INTO employees_backup
SELECT * FROM employees WHERE hire_date < '2020-01-01';
```

### SELECT

```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Select with alias
SELECT 
    first_name AS fname,
    last_name AS lname,
    salary AS annual_salary
FROM employees;

-- Select distinct values
SELECT DISTINCT department_id FROM employees;

-- Select with WHERE clause
SELECT * FROM employees WHERE salary > 70000;

-- Select with multiple conditions
SELECT * FROM employees 
WHERE salary > 70000 AND department_id = 1;

-- Select with IN
SELECT * FROM employees 
WHERE department_id IN (1, 2, 3);

-- Select with LIKE (pattern matching)
SELECT * FROM employees 
WHERE email LIKE '%@example.com';

-- Select with BETWEEN
SELECT * FROM employees 
WHERE salary BETWEEN 60000 AND 80000;

-- Select with IS NULL
SELECT * FROM employees WHERE email IS NULL;

-- Select with ORDER BY
SELECT * FROM employees 
ORDER BY salary DESC;

-- Select with LIMIT
SELECT * FROM employees 
ORDER BY salary DESC 
LIMIT 10;
```

### UPDATE

```sql
-- Update single row
UPDATE employees 
SET salary = 80000 
WHERE employee_id = 1;

-- Update multiple columns
UPDATE employees 
SET salary = 85000, email = 'new.email@example.com'
WHERE employee_id = 1;

-- Update multiple rows
UPDATE employees 
SET salary = salary * 1.1 
WHERE department_id = 1;
```

### DELETE

```sql
-- Delete specific rows
DELETE FROM employees WHERE employee_id = 1;

-- Delete with condition
DELETE FROM employees WHERE salary < 50000;

-- Delete all rows (be careful!)
DELETE FROM employees;

-- Truncate (faster, resets auto-increment)
TRUNCATE TABLE employees;
```

---

## SQL Functions

### Aggregate Functions

```sql
-- COUNT
SELECT COUNT(*) FROM employees;
SELECT COUNT(DISTINCT department_id) FROM employees;

-- SUM
SELECT SUM(salary) AS total_salary FROM employees;

-- AVG
SELECT AVG(salary) AS average_salary FROM employees;

-- MIN and MAX
SELECT MIN(salary) AS min_salary, MAX(salary) AS max_salary 
FROM employees;

-- GROUP BY
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;

-- HAVING (filter groups)
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 70000;
```

### String Functions

```sql
-- CONCAT
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;

-- UPPER and LOWER
SELECT UPPER(first_name) FROM employees;
SELECT LOWER(email) FROM employees;

-- SUBSTRING
SELECT SUBSTRING(email, 1, 5) FROM employees;

-- LENGTH
SELECT LENGTH(first_name) FROM employees;

-- TRIM
SELECT TRIM('  hello  ') AS trimmed;
```

### Date Functions

```sql
-- Current date/time
SELECT NOW();
SELECT CURDATE();
SELECT CURTIME();

-- Date arithmetic
SELECT DATE_ADD(hire_date, INTERVAL 1 YEAR) FROM employees;
SELECT DATEDIFF(CURDATE(), hire_date) AS days_employed FROM employees;

-- Extract date parts
SELECT YEAR(hire_date), MONTH(hire_date), DAY(hire_date) FROM employees;
```

---

## SQL Joins

Joins combine data from multiple tables.

### INNER JOIN

Returns only matching rows from both tables.

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;
```

### LEFT JOIN

Returns all rows from left table, matching rows from right table.

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;
```

### RIGHT JOIN

Returns all rows from right table, matching rows from left table.

```sql
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;
```

### FULL OUTER JOIN

Returns all rows from both tables (MySQL doesn't support, use UNION).

```sql
-- MySQL workaround
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
UNION
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;
```

### CROSS JOIN

Cartesian product of both tables.

```sql
SELECT e.first_name, d.department_name
FROM employees e
CROSS JOIN departments d;
```

### SELF JOIN

Join a table with itself.

```sql
-- Find employees and their managers
SELECT 
    e1.first_name AS employee,
    e2.first_name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id;
```

---

## Subqueries

Queries within queries.

### Scalar Subquery

Returns single value.

```sql
-- Employees with salary above average
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### Row Subquery

Returns single row.

```sql
-- Employee with highest salary
SELECT * FROM employees
WHERE (salary, department_id) = (
    SELECT MAX(salary), department_id 
    FROM employees 
    GROUP BY department_id 
    LIMIT 1
);
```

### Column Subquery

Returns single column.

```sql
-- Employees in departments with more than 5 employees
SELECT * FROM employees
WHERE department_id IN (
    SELECT department_id 
    FROM employees 
    GROUP BY department_id 
    HAVING COUNT(*) > 5
);
```

### Correlated Subquery

References outer query.

```sql
-- Employees with salary above department average
SELECT e1.* FROM employees e1
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees e2 
    WHERE e2.department_id = e1.department_id
);
```

### EXISTS

Check if subquery returns any rows.

```sql
-- Departments with employees
SELECT * FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e 
    WHERE e.department_id = d.department_id
);
```

---

## Window Functions

Perform calculations across rows related to current row.

### ROW_NUMBER

Assigns sequential numbers.

```sql
SELECT 
    first_name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS salary_rank
FROM employees;
```

### RANK and DENSE_RANK

```sql
SELECT 
    first_name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees;
```

### PARTITION BY

Window functions with grouping.

```sql
-- Rank within each department
SELECT 
    first_name,
    department_id,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;
```

### LAG and LEAD

Access previous/next row.

```sql
SELECT 
    first_name,
    salary,
    LAG(salary, 1) OVER (ORDER BY salary) AS prev_salary,
    LEAD(salary, 1) OVER (ORDER BY salary) AS next_salary
FROM employees;
```

### Aggregate Window Functions

```sql
-- Running total
SELECT 
    first_name,
    salary,
    SUM(salary) OVER (ORDER BY employee_id) AS running_total
FROM employees;

-- Average by department
SELECT 
    first_name,
    department_id,
    salary,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
FROM employees;
```

---

## Advanced SQL Topics

### Common Table Expressions (CTE)

Temporary named result set.

```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 80000
)
SELECT * FROM high_earners;
```

### Recursive CTE

```sql
WITH RECURSIVE manager_hierarchy AS (
    -- Base case
    SELECT employee_id, first_name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.employee_id, e.first_name, e.manager_id, mh.level + 1
    FROM employees e
    INNER JOIN manager_hierarchy mh ON e.manager_id = mh.employee_id
)
SELECT * FROM manager_hierarchy;
```

### Views

Virtual tables based on query results.

```sql
-- Create view
CREATE VIEW employee_summary AS
SELECT 
    department_id,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;

-- Use view
SELECT * FROM employee_summary;
```

### Stored Procedures

Reusable SQL code.

```sql
DELIMITER //
CREATE PROCEDURE GetEmployeeByDepartment(IN dept_id INT)
BEGIN
    SELECT * FROM employees WHERE department_id = dept_id;
END //
DELIMITER ;

-- Call procedure
CALL GetEmployeeByDepartment(1);
```

---

## SQL with Python

### Using SQLite

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('company.db')

# Create cursor
cursor = conn.cursor()

# Execute query
cursor.execute("SELECT * FROM employees")
results = cursor.fetchall()

# Using pandas
df = pd.read_sql("SELECT * FROM employees", conn)

# Close connection
conn.close()
```

### Using SQLAlchemy

```python
from sqlalchemy import create_engine
import pandas as pd

# Create engine
engine = create_engine('mysql+pymysql://user:password@localhost/dbname')

# Read from database
df = pd.read_sql("SELECT * FROM employees", engine)

# Write to database
df.to_sql('employees_backup', engine, if_exists='replace', index=False)
```

---

## Practice Exercises

### Exercise 1: Basic Queries

1. Select all employees from department 1
2. Find employees with salary above 70000
3. Count employees in each department

### Exercise 2: Joins

1. List employees with their department names
2. Find departments with no employees
3. List employees and their managers

### Exercise 3: Aggregations

1. Average salary by department
2. Highest paid employee in each department
3. Total salary cost by department

---

## Key Takeaways

1. **SQL is Essential**: Critical skill for data science
2. **Practice Regularly**: Write queries frequently
3. **Understand Joins**: Master different join types
4. **Window Functions**: Powerful for analytics
5. **Optimize Queries**: Use indexes, avoid SELECT *

---

**Remember**: SQL is the foundation of data manipulation. Master it to excel in data science!

