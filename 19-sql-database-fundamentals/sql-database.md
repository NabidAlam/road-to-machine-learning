# SQL and Database Fundamentals Complete Guide

Comprehensive guide to SQL and database management for data science. This guide takes you from absolute beginner to advanced SQL user with detailed explanations, examples, and real-world applications.

## Table of Contents

- [Introduction to Databases](#introduction-to-databases)
- [Database Fundamentals](#database-fundamentals)
- [SQL DDL Commands](#sql-ddl-commands)
- [SQL DML Commands](#sql-dml-commands)
- [SQL Functions](#sql-functions)
- [SQL Joins](#sql-joins)
- [Subqueries](#subqueries)
- [Window Functions](#window-functions)
- [Advanced SQL Topics](#advanced-sql-topics)
- [SQL with Python](#sql-with-python)
- [Data Cleaning with SQL](#data-cleaning-with-sql)
- [Common Mistakes and Best Practices](#common-mistakes-and-best-practices)
- [Practice Exercises](#practice-exercises)
- [Additional Resources](#additional-resources)

---

## Introduction to Databases

### What is a Database?

A database is an organized collection of data stored and accessed electronically. Think of it as a digital filing cabinet where data is stored in a structured format that makes it easy to find, update, and manage.

**Real-World Analogy:**
- **Spreadsheet**: Like a single table in a database
- **Database**: Like a collection of related spreadsheets (tables) with connections between them
- **DBMS**: The software that manages the database (like Excel manages spreadsheets)

**Why Databases?**
- **Data Persistence**: Store data permanently (unlike variables in memory)
- **Data Integrity**: Ensure data consistency and prevent errors
- **Efficient Access**: Fast retrieval and updates using indexes
- **Concurrent Access**: Multiple users can access data simultaneously
- **Security**: Access control and permissions to protect sensitive data
- **Scalability**: Handle large amounts of data efficiently
- **Relationships**: Connect related data across multiple tables

### Understanding Data Storage

**Without Database (File-based):**
```
customer_data.txt
order_data.txt
product_data.txt
```
Problems: No relationships, duplicate data, hard to query, no consistency

**With Database:**
```
customers table
orders table
products table
order_items table
```
Benefits: Relationships, no duplication, easy queries, data consistency

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

Understanding relationships is crucial for database design. Relationships define how tables connect to each other.

#### 1. One-to-One (1:1)
One record in Table A relates to exactly one record in Table B.

**Example:**
- Each employee has exactly one employee profile
- Each user has exactly one login credential

```sql
-- Employee table
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    name VARCHAR(100)
);

-- Employee profile table (one-to-one)
CREATE TABLE employee_profiles (
    profile_id INT PRIMARY KEY,
    employee_id INT UNIQUE,  -- UNIQUE ensures one-to-one
    bio TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);
```

**Visual:**
```
Employee 1 ──── Employee Profile 1
Employee 2 ──── Employee Profile 2
```

#### 2. One-to-Many (1:N)
One record in Table A relates to many records in Table B.

**Example:**
- One customer can have many orders
- One department can have many employees

```sql
-- Customers table (one)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100)
);

-- Orders table (many)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,  -- Foreign key (many side)
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

**Visual:**
```
Customer 1 ──── Order 1
            └── Order 2
            └── Order 3
Customer 2 ──── Order 4
```

#### 3. Many-to-Many (M:N)
Many records in Table A relate to many records in Table B. Requires a junction table.

**Example:**
- Students can enroll in many courses
- Courses can have many students

```sql
-- Students table
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100)
);

-- Courses table
CREATE TABLE courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(100)
);

-- Junction table (many-to-many)
CREATE TABLE enrollments (
    enrollment_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    enrollment_date DATE,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    UNIQUE(student_id, course_id)  -- Prevent duplicate enrollments
);
```

**Visual:**
```
Student 1 ──── Course 1
         └─── Course 2
Student 2 ──── Course 1
         └─── Course 3
```

### CRUD Operations

CRUD stands for the four basic operations you can perform on data:

- **Create**: INSERT data into tables
- **Read**: SELECT data from tables
- **Update**: UPDATE existing data
- **Delete**: DELETE data from tables

These operations form the foundation of all database interactions.

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

## Data Cleaning with SQL

SQL is powerful for data cleaning and preparation. Here are common techniques:

### Handling NULL Values

```sql
-- Check for NULL values
SELECT COUNT(*) FROM employees WHERE email IS NULL;

-- Replace NULL with default value
SELECT 
    first_name,
    COALESCE(email, 'no-email@example.com') AS email,
    COALESCE(salary, 0) AS salary
FROM employees;

-- Filter out NULL values
SELECT * FROM employees WHERE email IS NOT NULL;
```

### Removing Duplicates

```sql
-- Find duplicates
SELECT email, COUNT(*) as count
FROM employees
GROUP BY email
HAVING COUNT(*) > 1;

-- Remove duplicates (keep one)
DELETE e1 FROM employees e1
INNER JOIN employees e2 
WHERE e1.employee_id > e2.employee_id 
AND e1.email = e2.email;

-- Or use DISTINCT
SELECT DISTINCT email FROM employees;
```

### String Cleaning

```sql
-- Trim whitespace
SELECT TRIM(first_name) FROM employees;

-- Remove special characters (MySQL)
SELECT REGEXP_REPLACE(email, '[^a-zA-Z0-9@.]', '') FROM employees;

-- Standardize case
SELECT UPPER(first_name), LOWER(email) FROM employees;

-- Extract substring
SELECT SUBSTRING(email, 1, LOCATE('@', email) - 1) AS username FROM employees;
```

### Data Type Conversion

```sql
-- Convert string to number
SELECT CAST('123' AS UNSIGNED) AS number;
SELECT CONVERT('123', UNSIGNED) AS number;

-- Convert number to string
SELECT CAST(salary AS CHAR) AS salary_str FROM employees;

-- Date conversion
SELECT STR_TO_DATE('2024-01-15', '%Y-%m-%d') AS date_value;
SELECT DATE_FORMAT(hire_date, '%Y-%m-%d') AS formatted_date FROM employees;
```

### Handling Outliers

```sql
-- Find outliers using IQR method
WITH stats AS (
    SELECT 
        AVG(salary) AS mean_salary,
        STDDEV(salary) AS std_salary
    FROM employees
)
SELECT * FROM employees, stats
WHERE salary < mean_salary - 3 * std_salary
   OR salary > mean_salary + 3 * std_salary;

-- Remove outliers
DELETE FROM employees
WHERE salary < (SELECT AVG(salary) - 3 * STDDEV(salary) FROM employees)
   OR salary > (SELECT AVG(salary) + 3 * STDDEV(salary) FROM employees);
```

### Data Validation

```sql
-- Validate email format (basic)
SELECT * FROM employees
WHERE email NOT LIKE '%@%.%';

-- Validate date range
SELECT * FROM employees
WHERE hire_date < '1900-01-01' OR hire_date > CURDATE();

-- Validate numeric range
SELECT * FROM employees
WHERE salary < 0 OR salary > 1000000;
```

## Common Mistakes and Best Practices

### Common Mistakes

1. **Using SELECT *** in production
   ```sql
   -- Bad
   SELECT * FROM employees;
   
   -- Good
   SELECT employee_id, first_name, last_name FROM employees;
   ```

2. **Not using WHERE with UPDATE/DELETE**
   ```sql
   -- Dangerous! Updates all rows
   UPDATE employees SET salary = 50000;
   
   -- Safe
   UPDATE employees SET salary = 50000 WHERE employee_id = 1;
   ```

3. **Ignoring NULL values**
   ```sql
   -- Wrong: NULL != NULL in SQL
   WHERE email = NULL;  -- Always false!
   
   -- Correct
   WHERE email IS NULL;
   ```

4. **Not using indexes on foreign keys**
   ```sql
   -- Always index foreign keys
   CREATE INDEX idx_department_id ON employees(department_id);
   ```

5. **Cartesian products (missing JOIN condition)**
   ```sql
   -- Bad: Creates cartesian product
   SELECT * FROM employees, departments;
   
   -- Good
   SELECT * FROM employees e
   JOIN departments d ON e.department_id = d.department_id;
   ```

### Best Practices

1. **Use meaningful aliases**
   ```sql
   SELECT e.first_name, d.department_name
   FROM employees e
   JOIN departments d ON e.department_id = d.department_id;
   ```

2. **Format queries for readability**
   ```sql
   SELECT 
       e.first_name,
       e.last_name,
       d.department_name,
       e.salary
   FROM employees e
   INNER JOIN departments d 
       ON e.department_id = d.department_id
   WHERE e.salary > 70000
   ORDER BY e.salary DESC;
   ```

3. **Use transactions for multiple operations**
   ```sql
   START TRANSACTION;
   UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
   UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;
   COMMIT;  -- Or ROLLBACK if error
   ```

4. **Use prepared statements (in applications)**
   ```python
   # Python example
   cursor.execute("SELECT * FROM employees WHERE employee_id = %s", (emp_id,))
   ```

5. **Backup before major changes**
   ```sql
   CREATE TABLE employees_backup AS SELECT * FROM employees;
   ```

## Practice Exercises

### Exercise 1: Basic Queries

**Task 1:** Select all employees from department 1
```sql
SELECT * FROM employees WHERE department_id = 1;
```

**Task 2:** Find employees with salary above 70000
```sql
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > 70000
ORDER BY salary DESC;
```

**Task 3:** Count employees in each department
```sql
SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id
ORDER BY employee_count DESC;
```

### Exercise 2: Joins

**Task 1:** List employees with their department names
```sql
SELECT 
    e.first_name,
    e.last_name,
    d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;
```

**Task 2:** Find departments with no employees
```sql
SELECT d.department_name
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
WHERE e.employee_id IS NULL;
```

**Task 3:** List employees and their managers
```sql
SELECT 
    e1.first_name AS employee,
    e1.last_name AS employee_last,
    e2.first_name AS manager,
    e2.last_name AS manager_last
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id;
```

### Exercise 3: Aggregations

**Task 1:** Average salary by department
```sql
SELECT 
    d.department_name,
    AVG(e.salary) AS avg_salary,
    COUNT(e.employee_id) AS employee_count
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name
ORDER BY avg_salary DESC;
```

**Task 2:** Highest paid employee in each department
```sql
SELECT 
    d.department_name,
    e.first_name,
    e.last_name,
    e.salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE (e.department_id, e.salary) IN (
    SELECT department_id, MAX(salary)
    FROM employees
    GROUP BY department_id
);
```

**Task 3:** Total salary cost by department
```sql
SELECT 
    d.department_name,
    SUM(e.salary) AS total_salary_cost,
    COUNT(e.employee_id) AS employee_count,
    AVG(e.salary) AS avg_salary
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name
ORDER BY total_salary_cost DESC;
```

### Exercise 4: Window Functions

**Task:** Find employees ranked by salary within their department
```sql
SELECT 
    first_name,
    last_name,
    department_id,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank,
    ROUND(AVG(salary) OVER (PARTITION BY department_id), 2) AS dept_avg_salary
FROM employees
ORDER BY department_id, salary DESC;
```

### Exercise 5: Complex Query

**Task:** Find departments where average salary is above company average
```sql
WITH dept_stats AS (
    SELECT 
        department_id,
        AVG(salary) AS dept_avg_salary
    FROM employees
    GROUP BY department_id
),
company_avg AS (
    SELECT AVG(salary) AS company_avg_salary
    FROM employees
)
SELECT 
    d.department_name,
    ds.dept_avg_salary,
    ca.company_avg_salary,
    (ds.dept_avg_salary - ca.company_avg_salary) AS difference
FROM dept_stats ds
INNER JOIN departments d ON ds.department_id = d.department_id
CROSS JOIN company_avg ca
WHERE ds.dept_avg_salary > ca.company_avg_salary
ORDER BY difference DESC;
```

## Additional Resources

### Online Learning Platforms

1. **SQLBolt** (https://sqlbolt.com/) - Interactive SQL tutorials
2. **Mode Analytics SQL Tutorial** (https://mode.com/sql-tutorial/) - Comprehensive SQL guide
3. **W3Schools SQL** (https://www.w3schools.com/sql/) - SQL reference and examples
4. **SQLZoo** (https://sqlzoo.net/) - Practice SQL with real datasets
5. **LeetCode Database Problems** (https://leetcode.com/problemset/database/) - SQL interview practice

### Documentation

1. **MySQL Documentation** (https://dev.mysql.com/doc/) - Official MySQL reference
2. **PostgreSQL Documentation** (https://www.postgresql.org/docs/) - PostgreSQL reference
3. **SQLite Documentation** (https://www.sqlite.org/docs.html) - SQLite reference

### Practice Datasets

1. **Sakila Sample Database** - MySQL sample database for practice
2. **Northwind Database** - Classic sample database
3. **Chinook Database** - SQLite sample database

### Books

1. "SQL in 10 Minutes" by Ben Forta - Quick reference guide
2. "Learning SQL" by Alan Beaulieu - Comprehensive SQL learning
3. "SQL Cookbook" by Anthony Molinaro - Advanced SQL techniques

---

## Key Takeaways

1. **SQL is Essential**: Critical skill for data science and analytics
2. **Practice Regularly**: Write queries frequently to build muscle memory
3. **Understand Joins**: Master different join types - they're fundamental
4. **Window Functions**: Powerful for analytics and ranking operations
5. **Optimize Queries**: Use indexes, avoid SELECT *, understand query execution
6. **Data Cleaning**: SQL is excellent for data preparation and cleaning
7. **Think in Sets**: SQL works with sets of data, not individual rows
8. **Read Documentation**: Each database has its own quirks and functions

---

**Remember**: SQL is the foundation of data manipulation. Master it to excel in data science! Start with simple queries and gradually build complexity. Practice with real datasets and don't be afraid to experiment.

