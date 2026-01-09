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

### What is RDBMS?

**RDBMS (Relational Database Management System)** is software that manages relational databases. It stores data in tables (relations) and uses SQL to manage and query data.

**Key Features:**
- **Tables**: Data organized in rows and columns
- **Relationships**: Tables connected by keys
- **ACID Properties**: Atomicity, Consistency, Isolation, Durability
- **SQL**: Standard language for querying

**Popular RDBMS:**
- **MySQL**: Open-source, widely used
- **PostgreSQL**: Advanced open-source
- **SQL Server**: Microsoft's enterprise solution
- **Oracle**: Enterprise-grade
- **SQLite**: Lightweight, embedded

### What is SQL?

**SQL (Structured Query Language)** is the standard language for managing relational databases.

**SQL Categories:**
- **DDL (Data Definition Language)**: CREATE, ALTER, DROP
- **DML (Data Manipulation Language)**: SELECT, INSERT, UPDATE, DELETE
- **DCL (Data Control Language)**: GRANT, REVOKE
- **TCL (Transaction Control Language)**: COMMIT, ROLLBACK

**SQL Standards:**
- ANSI SQL (standard)
- Each RDBMS has extensions (MySQL, PostgreSQL, etc.)

### Installation & Setup

**MySQL Installation:**

**Windows:**
1. Download MySQL Installer from [mysql.com](https://dev.mysql.com/downloads/installer/)
2. Run installer
3. Choose "Developer Default" or "Server only"
4. Configure root password
5. Complete installation

**macOS:**
```bash
# Using Homebrew
brew install mysql
brew services start mysql
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

**PostgreSQL Installation:**

**Windows:**
1. Download from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run installer
3. Set postgres user password
4. Complete installation

**macOS:**
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**SQL Server Installation:**

**Windows:**
1. Download SQL Server Express (free) from [microsoft.com](https://www.microsoft.com/en-us/sql-server/sql-server-downloads)
2. Run installer
3. Choose installation type
4. Configure instance
5. Complete installation

**Basic Client UI Overview:**

**MySQL Workbench:**
- Visual database design
- SQL query editor
- Server administration
- Data modeling

**pgAdmin (PostgreSQL):**
- Database management
- Query tool
- Server administration
- Backup/restore

**SQL Server Management Studio (SSMS):**
- Database management
- Query editor
- Server administration
- Performance monitoring

**Command Line:**
```bash
# MySQL
mysql -u root -p

# PostgreSQL
psql -U postgres

# SQL Server
sqlcmd -S localhost -U sa -P password
```

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

### OLAP vs OLTP: Understanding Database Types

Understanding the difference between OLAP and OLTP is crucial for data analysis and database design.

#### OLTP (Online Transaction Processing)

**Purpose**: Handle day-to-day transactional operations

**Characteristics**:
- **High transaction volume**: Many small, frequent transactions
- **Fast response time**: Milliseconds for queries
- **Normalized data**: Optimized for writes and updates
- **Current data**: Real-time, up-to-date information
- **Small queries**: Simple SELECT, INSERT, UPDATE, DELETE operations
- **ACID compliance**: Ensures data consistency

**Examples**:
- E-commerce order processing
- Banking transactions (deposits, withdrawals)
- Inventory management systems
- Customer relationship management (CRM)
- Online booking systems

**Database Design**:
```sql
-- OLTP: Normalized structure (minimal redundancy)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
```

#### OLAP (Online Analytical Processing)

**Purpose**: Support complex analytical queries and reporting

**Characteristics**:
- **Low transaction volume**: Fewer, but complex queries
- **Slower response time**: Seconds to minutes acceptable
- **Denormalized data**: Optimized for reads and aggregations
- **Historical data**: Time-series, aggregated data
- **Large queries**: Complex JOINs, aggregations, GROUP BY
- **Read-heavy**: Primarily SELECT operations

**Examples**:
- Business intelligence dashboards
- Data warehouses
- Sales reporting and analytics
- Financial reporting
- Trend analysis

**Database Design**:
```sql
-- OLAP: Denormalized structure (star schema)
CREATE TABLE fact_sales (
    sale_id INT PRIMARY KEY,
    date_id INT,
    product_id INT,
    customer_id INT,
    store_id INT,
    quantity INT,
    revenue DECIMAL(10,2),
    profit DECIMAL(10,2)
);

CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    date DATE,
    year INT,
    quarter INT,
    month INT,
    day_of_week VARCHAR(10)
);

CREATE TABLE dim_product (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    brand VARCHAR(50)
);
```

#### Key Differences

| Aspect | OLTP | OLAP |
|--------|------|------|
| **Purpose** | Transaction processing | Data analysis |
| **Users** | Clerks, operators | Analysts, managers |
| **Data** | Current, detailed | Historical, aggregated |
| **Design** | Normalized | Denormalized (star/snowflake) |
| **Queries** | Simple, fast | Complex, analytical |
| **Operations** | INSERT, UPDATE, DELETE | SELECT (mostly) |
| **Size** | Small to medium | Large to very large |
| **Response Time** | Milliseconds | Seconds to minutes |
| **Example** | Order entry system | Sales reporting system |

#### When to Use Each

**Use OLTP when**:
- You need to process transactions in real-time
- Data must be current and accurate
- You need fast write operations
- You're building operational systems
- Examples: E-commerce, banking, inventory

**Use OLAP when**:
- You need to analyze historical data
- You're building reports and dashboards
- You need complex aggregations
- You're doing business intelligence
- Examples: Data warehouses, BI tools, analytics platforms

#### ETL Process: OLTP to OLAP

In practice, data flows from OLTP to OLAP:

```
OLTP Database (Operational)
    ↓
ETL Process (Extract, Transform, Load)
    ↓
OLAP Database (Data Warehouse)
    ↓
BI Tools (Power BI, Tableau, etc.)
```

**ETL Process**:
1. **Extract**: Pull data from OLTP systems
2. **Transform**: Clean, aggregate, denormalize
3. **Load**: Store in OLAP data warehouse

**Example ETL Pipeline**:
```python
# Extract from OLTP
import pandas as pd
import sqlalchemy

# Connect to OLTP database
oltp_engine = sqlalchemy.create_engine('postgresql://oltp_db')

# Extract daily sales
query = """
    SELECT 
        DATE(order_date) as sale_date,
        product_id,
        customer_id,
        SUM(quantity) as total_quantity,
        SUM(amount) as total_revenue
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 day'
    GROUP BY DATE(order_date), product_id, customer_id
"""

df = pd.read_sql(query, oltp_engine)

# Transform: Add dimensions
df['year'] = pd.to_datetime(df['sale_date']).dt.year
df['month'] = pd.to_datetime(df['sale_date']).dt.month
df['quarter'] = pd.to_datetime(df['sale_date']).dt.quarter

# Load into OLAP data warehouse
olap_engine = sqlalchemy.create_engine('postgresql://olap_db')
df.to_sql('fact_sales', olap_engine, if_exists='append', index=False)
```

#### Resources

- [OLTP vs OLAP Explained (GeeksforGeeks)](https://www.geeksforgeeks.org/difference-between-oltp-and-olap/)
- [Data Warehouse Concepts (Oracle)](https://docs.oracle.com/cd/B10501_01/server.920/a96520/concept.htm)
- [Star Schema Design (Kimball Group)](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/)

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

**Basic Syntax:**
```sql
CREATE TABLE table_name (
    column1 datatype constraint,
    column2 datatype constraint,
    ...
);
```

**Example:**
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

**Defining Columns:**
```sql
CREATE TABLE products (
    product_id INT,
    product_name VARCHAR(100),
    price DECIMAL(10, 2),
    quantity INT
);
```

### Constraints

**PRIMARY KEY:**
Uniquely identifies each row.

```sql
-- Single column primary key
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    ...
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

**FOREIGN KEY:**
References another table's primary key.

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- With options
FOREIGN KEY (customer_id) 
REFERENCES customers(customer_id)
ON DELETE CASCADE      -- Delete related rows
ON UPDATE CASCADE;     -- Update related rows
```

**NOT NULL:**
Column cannot contain NULL values.

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100)  -- Can be NULL
);
```

**UNIQUE:**
Column values must be unique.

```sql
-- Single column
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    ...
);

-- Multiple columns (composite unique)
CREATE TABLE user_logins (
    user_id INT,
    login_date DATE,
    UNIQUE(user_id, login_date)
);
```

**DEFAULT:**
Sets default value for column.

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    order_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**CHECK:**
Validates column values.

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    salary DECIMAL(10, 2) CHECK (salary > 0),
    age INT CHECK (age >= 18 AND age <= 65)
);
```

### Data Types

**Numeric:**
- `INT` or `INTEGER`: Integer (-2,147,483,648 to 2,147,483,647)
- `BIGINT`: Large integer
- `SMALLINT`: Small integer
- `TINYINT`: Very small integer
- `DECIMAL(p, s)`: Fixed-point number (precision, scale)
  - Example: `DECIMAL(10, 2)` = 12345678.90
- `NUMERIC(p, s)`: Same as DECIMAL
- `FLOAT`: Floating-point number
- `DOUBLE`: Double precision floating-point
- `REAL`: Single precision floating-point

**String:**
- `VARCHAR(n)`: Variable-length string (max n characters)
  - Example: `VARCHAR(50)` for names
- `CHAR(n)`: Fixed-length string (always n characters)
  - Example: `CHAR(2)` for state codes
- `TEXT`: Long text (unlimited length)
- `LONGTEXT`: Very long text (MySQL)

**Date/Time:**
- `DATE`: Date (YYYY-MM-DD)
  - Example: '2024-01-15'
- `TIME`: Time (HH:MM:SS)
  - Example: '14:30:00'
- `DATETIME`: Date and time
  - Example: '2024-01-15 14:30:00'
- `TIMESTAMP`: Auto-updating timestamp
- `YEAR`: Year value (MySQL)

**Other:**
- `BOOLEAN` or `BOOL`: True/False (stored as TINYINT in MySQL)
- `BLOB`: Binary large object
- `JSON`: JSON data type (MySQL 5.7+, PostgreSQL 9.4+)

**Choosing Data Types:**
- Use smallest appropriate type (saves space)
- Use VARCHAR for variable-length text
- Use DECIMAL for money/precise calculations
- Use DATE/DATETIME for dates (not strings)

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

### INSERT - Create (C)

**Insert Single Row:**
```sql
-- Specify columns
INSERT INTO employees (first_name, last_name, email, hire_date, salary)
VALUES ('John', 'Doe', 'john.doe@example.com', '2024-01-15', 75000.00);

-- All columns (in order)
INSERT INTO employees
VALUES (1, 'John', 'Doe', 'john.doe@example.com', '2024-01-15', 75000.00, 1);
```

**Insert Multiple Rows:**
```sql
INSERT INTO employees (first_name, last_name, email, hire_date, salary)
VALUES 
    ('Jane', 'Smith', 'jane.smith@example.com', '2024-02-01', 80000.00),
    ('Bob', 'Johnson', 'bob.johnson@example.com', '2024-02-15', 70000.00),
    ('Alice', 'Williams', 'alice.williams@example.com', '2024-03-01', 75000.00);
```

**Insert from Another Table:**
```sql
-- Copy all columns
INSERT INTO employees_backup
SELECT * FROM employees WHERE hire_date < '2020-01-01';

-- Copy specific columns
INSERT INTO employees_backup (first_name, last_name, email)
SELECT first_name, last_name, email 
FROM employees 
WHERE department_id = 1;
```

**Insert with Default Values:**
```sql
-- Use DEFAULT keyword
INSERT INTO orders (customer_id, order_date, status)
VALUES (1, DEFAULT, DEFAULT);

-- Omit columns with defaults
INSERT INTO orders (customer_id)
VALUES (1);
```

### SELECT - Basic Queries

**SELECT Statement:**
The SELECT statement retrieves data from tables.

**Basic Syntax:**
```sql
SELECT column1, column2, ...
FROM table_name;
```

**FROM Clause:**
Specifies the table(s) to query.

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
```

### Filtering Data - WHERE Clause

**WHERE Clause:**
Filters rows based on conditions.

**Comparison Operators:**
```sql
-- Equal to
SELECT * FROM employees WHERE salary = 70000;

-- Greater than
SELECT * FROM employees WHERE salary > 70000;

-- Less than
SELECT * FROM employees WHERE salary < 70000;

-- Not equal to (!= or <>)
SELECT * FROM employees WHERE salary != 70000;
SELECT * FROM employees WHERE salary <> 70000;

-- Greater than or equal
SELECT * FROM employees WHERE salary >= 70000;

-- Less than or equal
SELECT * FROM employees WHERE salary <= 70000;
```

**Logical Operators:**
```sql
-- AND: Both conditions must be true
SELECT * FROM employees 
WHERE salary > 70000 AND department_id = 1;

-- OR: Either condition can be true
SELECT * FROM employees 
WHERE salary > 70000 OR department_id = 1;

-- NOT: Negates condition
SELECT * FROM employees 
WHERE NOT salary > 70000;

-- Combining operators
SELECT * FROM employees 
WHERE (salary > 70000 OR salary < 50000) 
AND department_id IN (1, 2);
```

**Special Operators:**

**BETWEEN:**
```sql
-- Inclusive range
SELECT * FROM employees 
WHERE salary BETWEEN 60000 AND 80000;

-- Equivalent to
SELECT * FROM employees 
WHERE salary >= 60000 AND salary <= 80000;
```

**IN:**
```sql
-- Match any value in list
SELECT * FROM employees 
WHERE department_id IN (1, 2, 3);

-- Equivalent to
SELECT * FROM employees 
WHERE department_id = 1 
   OR department_id = 2 
   OR department_id = 3;
```

**LIKE (Pattern Matching):**
```sql
-- % matches any sequence of characters
SELECT * FROM employees 
WHERE email LIKE '%@example.com';  -- Ends with @example.com

SELECT * FROM employees 
WHERE first_name LIKE 'John%';     -- Starts with John

SELECT * FROM employees 
WHERE email LIKE '%@%.%';          -- Contains @ and .

-- _ matches single character
SELECT * FROM employees 
WHERE first_name LIKE 'J_n';       -- J, any char, n (e.g., Jan, Jon)

-- Escape special characters
SELECT * FROM employees 
WHERE email LIKE '%\_%' ESCAPE '\'; -- Contains underscore
```

**IS NULL / IS NOT NULL:**
```sql
-- Check for NULL values
SELECT * FROM employees WHERE email IS NULL;

-- Check for non-NULL values
SELECT * FROM employees WHERE email IS NOT NULL;

-- Important: NULL != NULL in SQL
-- Wrong:
WHERE email = NULL;  -- Always false!

-- Correct:
WHERE email IS NULL;
```

### Sorting Results - ORDER BY

**ORDER BY Clause:**
Sorts result set.

```sql
-- Ascending (default)
SELECT * FROM employees 
ORDER BY salary ASC;

-- Descending
SELECT * FROM employees 
ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees 
ORDER BY department_id ASC, salary DESC;

-- Using column position
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY 3 DESC;  -- Sort by 3rd column (salary)
```

### Limiting Results

**LIMIT (MySQL/PostgreSQL):**
```sql
-- Limit number of rows
SELECT * FROM employees 
ORDER BY salary DESC 
LIMIT 10;

-- Limit with offset
SELECT * FROM employees 
ORDER BY salary DESC 
LIMIT 10 OFFSET 20;  -- Skip 20, return 10

-- Shorthand (MySQL)
SELECT * FROM employees 
ORDER BY salary DESC 
LIMIT 20, 10;  -- Offset 20, limit 10
```

**TOP (SQL Server):**
```sql
-- Top N rows
SELECT TOP 10 * FROM employees 
ORDER BY salary DESC;

-- Top N percent
SELECT TOP 10 PERCENT * FROM employees 
ORDER BY salary DESC;

-- With ties
SELECT TOP 10 WITH TIES * FROM employees 
ORDER BY salary DESC;
```

**FETCH (SQL Standard):**
```sql
-- PostgreSQL, SQL Server 2012+
SELECT * FROM employees 
ORDER BY salary DESC 
OFFSET 20 ROWS 
FETCH NEXT 10 ROWS ONLY;
```

### UPDATE - Update (U)

**Update Single Row:**
```sql
UPDATE employees 
SET salary = 80000 
WHERE employee_id = 1;
```

**Update Multiple Columns:**
```sql
UPDATE employees 
SET salary = 85000, 
    email = 'new.email@example.com',
    hire_date = '2024-01-20'
WHERE employee_id = 1;
```

**Update Multiple Rows:**
```sql
-- Update all employees in department 1
UPDATE employees 
SET salary = salary * 1.1 
WHERE department_id = 1;

-- Update based on condition
UPDATE employees 
SET salary = CASE 
    WHEN salary < 50000 THEN salary * 1.15
    WHEN salary < 70000 THEN salary * 1.10
    ELSE salary * 1.05
END
WHERE department_id = 1;
```

**Update with Subquery:**
```sql
-- Update based on another table
UPDATE employees e
SET salary = (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
)
WHERE department_id = 1;
```

**Important:** Always use WHERE clause! Without it, all rows are updated.

### DELETE - Delete (D)

**Delete Specific Rows:**
```sql
-- Delete single row
DELETE FROM employees WHERE employee_id = 1;

-- Delete with condition
DELETE FROM employees WHERE salary < 50000;

-- Delete with multiple conditions
DELETE FROM employees 
WHERE department_id = 1 AND salary < 50000;
```

**Delete All Rows:**
```sql
-- Delete all rows (be careful!)
DELETE FROM employees;

-- Truncate (faster, resets auto-increment, cannot rollback)
TRUNCATE TABLE employees;
```

**Delete with Subquery:**
```sql
-- Delete employees in departments with no active projects
DELETE FROM employees
WHERE department_id IN (
    SELECT department_id 
    FROM departments 
    WHERE active_projects = 0
);
```

**Important:** Always use WHERE clause! Without it, all rows are deleted.

### Transaction Control

**COMMIT:**
Saves all changes made in the current transaction.

```sql
START TRANSACTION;
INSERT INTO employees (first_name, last_name) VALUES ('John', 'Doe');
UPDATE employees SET salary = 80000 WHERE employee_id = 1;
COMMIT;  -- Save changes
```

**ROLLBACK:**
Undoes all changes made in the current transaction.

```sql
START TRANSACTION;
INSERT INTO employees (first_name, last_name) VALUES ('John', 'Doe');
UPDATE employees SET salary = 80000 WHERE employee_id = 1;
ROLLBACK;  -- Undo changes
```

**Example:**
```sql
-- Transfer money between accounts
START TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- Check for errors
-- If successful:
COMMIT;
-- If error:
ROLLBACK;
```

**Auto-commit:**
- Most databases auto-commit each statement
- Use transactions for multiple related operations
- Ensures data consistency

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

**CONCAT:**
Concatenate strings.

```sql
-- MySQL
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;

-- SQL Server, PostgreSQL
SELECT first_name || ' ' || last_name AS full_name FROM employees;

-- Multiple strings
SELECT CONCAT(first_name, ' ', last_name, ' - ', email) AS info FROM employees;
```

**LENGTH:**
Get string length.

```sql
-- MySQL
SELECT LENGTH(first_name) FROM employees;

-- SQL Server
SELECT LEN(first_name) FROM employees;

-- PostgreSQL
SELECT LENGTH(first_name) FROM employees;
```

**UPPER and LOWER:**
Change case.

```sql
SELECT UPPER(first_name) FROM employees;
SELECT LOWER(email) FROM employees;
```

**SUBSTRING:**
Extract substring.

```sql
-- MySQL, PostgreSQL
SELECT SUBSTRING(email, 1, 5) FROM employees;
SELECT SUBSTRING(email FROM 1 FOR 5) FROM employees;  -- Standard SQL

-- SQL Server
SELECT SUBSTRING(email, 1, 5) FROM employees;

-- Extract domain from email
SELECT SUBSTRING(email, LOCATE('@', email) + 1) AS domain FROM employees;
```

**Other String Functions:**
```sql
-- TRIM: Remove leading/trailing spaces
SELECT TRIM('  hello  ') AS trimmed;
SELECT LTRIM('  hello') AS left_trimmed;
SELECT RTRIM('hello  ') AS right_trimmed;

-- REPLACE: Replace substring
SELECT REPLACE(email, '@example.com', '@company.com') FROM employees;

-- REVERSE: Reverse string
SELECT REVERSE(first_name) FROM employees;

-- LEFT and RIGHT: Extract from sides
SELECT LEFT(email, 5) FROM employees;
SELECT RIGHT(email, 10) FROM employees;
```

### Date/Time Functions

**Current Date/Time:**
```sql
-- MySQL
SELECT NOW();        -- Current date and time
SELECT CURDATE();    -- Current date
SELECT CURTIME();    -- Current time

-- PostgreSQL, SQL Server
SELECT CURRENT_TIMESTAMP;  -- Current date and time
SELECT CURRENT_DATE;       -- Current date
SELECT CURRENT_TIME;       -- Current time
```

**Date Arithmetic:**
```sql
-- MySQL
SELECT DATE_ADD(hire_date, INTERVAL 1 YEAR) FROM employees;
SELECT DATE_SUB(hire_date, INTERVAL 1 MONTH) FROM employees;
SELECT DATEDIFF(CURDATE(), hire_date) AS days_employed FROM employees;

-- PostgreSQL
SELECT hire_date + INTERVAL '1 year' FROM employees;
SELECT hire_date - INTERVAL '1 month' FROM employees;
SELECT CURRENT_DATE - hire_date AS days_employed FROM employees;

-- SQL Server
SELECT DATEADD(YEAR, 1, hire_date) FROM employees;
SELECT DATEDIFF(DAY, hire_date, GETDATE()) AS days_employed FROM employees;
```

**Extract Date Parts:**
```sql
-- MySQL, PostgreSQL
SELECT YEAR(hire_date), MONTH(hire_date), DAY(hire_date) FROM employees;
SELECT DAYOFWEEK(hire_date) FROM employees;  -- MySQL
SELECT EXTRACT(YEAR FROM hire_date) FROM employees;  -- Standard SQL

-- SQL Server
SELECT YEAR(hire_date), MONTH(hire_date), DAY(hire_date) FROM employees;
SELECT DATEPART(YEAR, hire_date) FROM employees;
```

**DATE_FORMAT (MySQL):**
```sql
-- Format dates
SELECT DATE_FORMAT(hire_date, '%Y-%m-%d') AS formatted_date FROM employees;
SELECT DATE_FORMAT(hire_date, '%M %d, %Y') AS formatted_date FROM employees;
SELECT DATE_FORMAT(hire_date, '%W, %M %d') AS formatted_date FROM employees;

-- Common formats
-- %Y: 4-digit year
-- %y: 2-digit year
-- %m: Month (01-12)
-- %M: Month name (January-December)
-- %d: Day (01-31)
-- %W: Weekday name (Sunday-Saturday)
-- %H: Hour (00-23)
-- %i: Minutes (00-59)
-- %s: Seconds (00-59)
```

**TO_CHAR (PostgreSQL):**
```sql
SELECT TO_CHAR(hire_date, 'YYYY-MM-DD') AS formatted_date FROM employees;
SELECT TO_CHAR(hire_date, 'Month DD, YYYY') AS formatted_date FROM employees;
```

**FORMAT (SQL Server):**
```sql
SELECT FORMAT(hire_date, 'yyyy-MM-dd') AS formatted_date FROM employees;
SELECT FORMAT(hire_date, 'MMMM dd, yyyy') AS formatted_date FROM employees;
```

### Numeric Functions

**ROUND:**
Round to specified decimal places.

```sql
-- Round to 2 decimal places
SELECT ROUND(salary, 2) FROM employees;

-- Round to nearest integer
SELECT ROUND(salary) FROM employees;

-- Round to nearest 100
SELECT ROUND(salary, -2) FROM employees;
```

**MOD:**
Modulo (remainder after division).

```sql
-- MySQL
SELECT MOD(10, 3);  -- Returns 1

-- PostgreSQL, SQL Server
SELECT 10 % 3;  -- Returns 1

-- Example: Find even employee IDs
SELECT * FROM employees WHERE MOD(employee_id, 2) = 0;
```

**Other Numeric Functions:**
```sql
-- ABS: Absolute value
SELECT ABS(-10);  -- Returns 10

-- CEIL/CEILING: Round up
SELECT CEIL(10.1);  -- Returns 11

-- FLOOR: Round down
SELECT FLOOR(10.9);  -- Returns 10

-- POWER: Raise to power
SELECT POWER(2, 3);  -- Returns 8

-- SQRT: Square root
SELECT SQRT(16);  -- Returns 4
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

Join a table with itself (useful for hierarchical data).

```sql
-- Find employees and their managers
SELECT 
    e1.first_name AS employee,
    e2.first_name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id;

-- Find employees in same department
SELECT 
    e1.first_name AS employee1,
    e2.first_name AS employee2,
    e1.department_id
FROM employees e1
INNER JOIN employees e2 
    ON e1.department_id = e2.department_id 
    AND e1.employee_id < e2.employee_id;  -- Avoid duplicates
```

### CROSS JOIN

Cartesian product of both tables (every row from first table paired with every row from second table).

```sql
-- Explicit CROSS JOIN
SELECT e.first_name, d.department_name
FROM employees e
CROSS JOIN departments d;

-- Implicit CROSS JOIN (comma)
SELECT e.first_name, d.department_name
FROM employees e, departments d;

-- Use case: Generate all combinations
-- Example: All employees × All products
SELECT e.first_name, p.product_name
FROM employees e
CROSS JOIN products p;
```

**Warning:** CROSS JOIN can produce very large result sets. Use with caution!

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

Returns single column (used with IN, ANY, ALL).

**Subqueries in WHERE Clause:**

**Using IN:**
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

**Using ANY:**
```sql
-- Employees with salary greater than any salary in department 1
SELECT * FROM employees
WHERE salary > ANY (
    SELECT salary 
    FROM employees 
    WHERE department_id = 1
);

-- Equivalent to
SELECT * FROM employees
WHERE salary > (
    SELECT MIN(salary) 
    FROM employees 
    WHERE department_id = 1
);
```

**Using ALL:**
```sql
-- Employees with salary greater than all salaries in department 1
SELECT * FROM employees
WHERE salary > ALL (
    SELECT salary 
    FROM employees 
    WHERE department_id = 1
);

-- Equivalent to
SELECT * FROM employees
WHERE salary > (
    SELECT MAX(salary) 
    FROM employees 
    WHERE department_id = 1
);
```

**Subqueries in FROM Clause (Derived Tables):**
```sql
-- Use subquery as table
SELECT 
    dept_stats.department_id,
    dept_stats.avg_salary,
    d.department_name
FROM (
    SELECT 
        department_id,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
) AS dept_stats
INNER JOIN departments d ON dept_stats.department_id = d.department_id;
```

**Subqueries in SELECT Clause (Scalar Subqueries):**
```sql
-- Single value subquery
SELECT 
    first_name,
    salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    salary - (SELECT AVG(salary) FROM employees) AS difference
FROM employees;
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

### Common Table Expressions (CTE) - WITH Clause

Temporary named result set (simplifies complex queries).

**Basic CTE:**
```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 80000
)
SELECT * FROM high_earners;

-- Multiple CTEs
WITH 
    high_earners AS (
        SELECT * FROM employees WHERE salary > 80000
    ),
    dept_stats AS (
        SELECT department_id, AVG(salary) AS avg_salary
        FROM employees
        GROUP BY department_id
    )
SELECT 
    h.first_name,
    h.salary,
    d.avg_salary
FROM high_earners h
INNER JOIN dept_stats d ON h.department_id = d.department_id;
```

**Use Cases:**
- Simplify complex queries
- Reuse subqueries
- Improve readability
- Replace derived tables

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

### Set Operators

**UNION:**
Combines result sets (removes duplicates).

```sql
-- Combine employees from two tables
SELECT first_name, last_name FROM employees_2023
UNION
SELECT first_name, last_name FROM employees_2024;

-- All columns must match in order and type
SELECT employee_id, first_name FROM employees
UNION
SELECT customer_id, first_name FROM customers;
```

**UNION ALL:**
Combines result sets (keeps duplicates).

```sql
-- Faster than UNION (no duplicate removal)
SELECT first_name, last_name FROM employees_2023
UNION ALL
SELECT first_name, last_name FROM employees_2024;
```

**INTERSECT:**
Returns common rows (PostgreSQL, SQL Server).

```sql
-- Employees in both tables
SELECT employee_id FROM employees_2023
INTERSECT
SELECT employee_id FROM employees_2024;
```

**EXCEPT (MINUS in Oracle):**
Returns rows in first query but not in second.

```sql
-- Employees only in 2023
SELECT employee_id FROM employees_2023
EXCEPT
SELECT employee_id FROM employees_2024;
```

### Views - CREATE VIEW

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

-- Update view (if updatable)
UPDATE employee_summary SET avg_salary = 75000 WHERE department_id = 1;

-- Drop view
DROP VIEW employee_summary;
```

**Benefits:**
- Simplify complex queries
- Security (hide sensitive columns)
- Reusability
- Consistency

### Stored Procedures

Reusable SQL code with parameters.

**CREATE PROCEDURE:**
```sql
-- MySQL
DELIMITER //
CREATE PROCEDURE GetEmployeeByDepartment(IN dept_id INT)
BEGIN
    SELECT * FROM employees WHERE department_id = dept_id;
END //
DELIMITER ;

-- Call procedure
CALL GetEmployeeByDepartment(1);
```

**Parameters:**
- **IN**: Input parameter (default)
- **OUT**: Output parameter
- **INOUT**: Both input and output

```sql
-- MySQL example with OUT parameter
DELIMITER //
CREATE PROCEDURE GetDepartmentStats(
    IN dept_id INT,
    OUT emp_count INT,
    OUT avg_salary DECIMAL(10,2)
)
BEGIN
    SELECT COUNT(*), AVG(salary)
    INTO emp_count, avg_salary
    FROM employees
    WHERE department_id = dept_id;
END //
DELIMITER ;

-- Call
CALL GetDepartmentStats(1, @count, @avg);
SELECT @count, @avg;
```

**Control Flow:**
```sql
-- IF/ELSE
DELIMITER //
CREATE PROCEDURE UpdateSalary(IN emp_id INT, IN new_salary DECIMAL)
BEGIN
    IF new_salary > 0 THEN
        UPDATE employees SET salary = new_salary WHERE employee_id = emp_id;
    ELSE
        SELECT 'Salary must be positive' AS error;
    END IF;
END //
DELIMITER ;

-- CASE
DELIMITER //
CREATE PROCEDURE GetEmployeeStatus(IN emp_id INT)
BEGIN
    SELECT 
        first_name,
        CASE 
            WHEN salary > 80000 THEN 'High'
            WHEN salary > 50000 THEN 'Medium'
            ELSE 'Low'
        END AS salary_status
    FROM employees
    WHERE employee_id = emp_id;
END //
DELIMITER ;

-- Looping (MySQL)
DELIMITER //
CREATE PROCEDURE InsertTestEmployees(IN num INT)
BEGIN
    DECLARE i INT DEFAULT 1;
    WHILE i <= num DO
        INSERT INTO employees (first_name, last_name)
        VALUES (CONCAT('Test', i), CONCAT('User', i));
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;
```

**PostgreSQL Example:**
```sql
CREATE OR REPLACE PROCEDURE get_employee_by_department(dept_id INT)
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT * FROM employees WHERE department_id = dept_id;
END;
$$;

-- Call
CALL get_employee_by_department(1);
```

**SQL Server Example:**
```sql
CREATE PROCEDURE GetEmployeeByDepartment
    @dept_id INT
AS
BEGIN
    SELECT * FROM employees WHERE department_id = @dept_id;
END;

-- Call
EXEC GetEmployeeByDepartment @dept_id = 1;
```

### User-Defined Functions (UDFs)

**Scalar Functions:**
Return single value.

```sql
-- MySQL
DELIMITER //
CREATE FUNCTION CalculateBonus(salary DECIMAL(10,2))
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
    RETURN salary * 0.1;
END //
DELIMITER ;

-- Use
SELECT first_name, salary, CalculateBonus(salary) AS bonus FROM employees;
```

**Table-Valued Functions (SQL Server, PostgreSQL):**
Return table.

```sql
-- SQL Server
CREATE FUNCTION GetEmployeesByDepartment(@dept_id INT)
RETURNS TABLE
AS
RETURN
    SELECT * FROM employees WHERE department_id = @dept_id;

-- Use
SELECT * FROM GetEmployeesByDepartment(1);
```

**PostgreSQL:**
```sql
CREATE FUNCTION get_employees_by_department(dept_id INT)
RETURNS TABLE (
    employee_id INT,
    first_name VARCHAR(50),
    salary DECIMAL(10,2)
)
AS $$
BEGIN
    RETURN QUERY
    SELECT e.employee_id, e.first_name, e.salary
    FROM employees e
    WHERE e.department_id = dept_id;
END;
$$ LANGUAGE plpgsql;

-- Use
SELECT * FROM get_employees_by_department(1);
```

### Indexes

**Purpose and Importance:**
- Speed up queries
- Improve performance
- Essential for large tables
- Automatically created for PRIMARY KEY and UNIQUE

**CREATE INDEX:**
```sql
-- Single column index
CREATE INDEX idx_department_id ON employees(department_id);

-- Composite index
CREATE INDEX idx_dept_salary ON employees(department_id, salary);

-- Unique index
CREATE UNIQUE INDEX idx_email ON employees(email);
```

**DROP INDEX:**
```sql
-- MySQL
DROP INDEX idx_department_id ON employees;

-- PostgreSQL, SQL Server
DROP INDEX idx_department_id;
```

**Clustered vs Non-Clustered:**

**Clustered Index:**
- Physical order of rows matches index order
- Only one per table (usually PRIMARY KEY)
- Faster for range queries

**Non-Clustered Index:**
- Separate structure pointing to data
- Multiple per table
- Faster for lookups

```sql
-- SQL Server: Clustered index
CREATE CLUSTERED INDEX idx_employee_id ON employees(employee_id);

-- Non-clustered index (default)
CREATE NONCLUSTERED INDEX idx_department_id ON employees(department_id);
```

**When to Create Indexes:**
- Foreign keys (always)
- Frequently queried columns
- Columns in WHERE clauses
- Columns in JOIN conditions
- Columns in ORDER BY

**When NOT to Create Indexes:**
- Small tables
- Frequently updated columns
- Columns with few unique values
- Too many indexes (slows INSERT/UPDATE)

### Triggers

**Concept:**
Automated actions executed when specific events occur.

**CREATE TRIGGER:**
```sql
-- MySQL: BEFORE INSERT
DELIMITER //
CREATE TRIGGER before_employee_insert
BEFORE INSERT ON employees
FOR EACH ROW
BEGIN
    IF NEW.salary < 0 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Salary cannot be negative';
    END IF;
END //
DELIMITER ;

-- MySQL: AFTER UPDATE
DELIMITER //
CREATE TRIGGER after_employee_update
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO employee_audit (employee_id, old_salary, new_salary, change_date)
    VALUES (NEW.employee_id, OLD.salary, NEW.salary, NOW());
END //
DELIMITER ;
```

**Event Types:**
- **BEFORE INSERT**: Before row is inserted
- **AFTER INSERT**: After row is inserted
- **BEFORE UPDATE**: Before row is updated
- **AFTER UPDATE**: After row is updated
- **BEFORE DELETE**: Before row is deleted
- **AFTER DELETE**: After row is deleted

**PostgreSQL Example:**
```sql
CREATE OR REPLACE FUNCTION log_employee_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO employee_audit (employee_id, old_salary, new_salary, change_date)
    VALUES (NEW.employee_id, OLD.salary, NEW.salary, NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_employee_update
AFTER UPDATE ON employees
FOR EACH ROW
EXECUTE FUNCTION log_employee_changes();
```

**SQL Server Example:**
```sql
CREATE TRIGGER after_employee_update
ON employees
AFTER UPDATE
AS
BEGIN
    INSERT INTO employee_audit (employee_id, old_salary, new_salary, change_date)
    SELECT 
        i.employee_id,
        d.salary AS old_salary,
        i.salary AS new_salary,
        GETDATE()
    FROM inserted i
    INNER JOIN deleted d ON i.employee_id = d.employee_id;
END;
```

**Example Use Cases:**

**1. Auditing:**
```sql
-- Track all changes to employee table
CREATE TRIGGER audit_employee_changes
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, action, old_data, new_data, timestamp)
    VALUES ('employees', 'UPDATE', OLD.employee_id, NEW.employee_id, NOW());
END;
```

**2. Enforcing Business Rules:**
```sql
-- Prevent salary decrease
CREATE TRIGGER prevent_salary_decrease
BEFORE UPDATE ON employees
FOR EACH ROW
BEGIN
    IF NEW.salary < OLD.salary THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cannot decrease salary';
    END IF;
END;
```

**3. Auto-calculations:**
```sql
-- Update total when order items change
CREATE TRIGGER update_order_total
AFTER INSERT ON order_items
FOR EACH ROW
BEGIN
    UPDATE orders
    SET total = (
        SELECT SUM(quantity * price)
        FROM order_items
        WHERE order_id = NEW.order_id
    )
    WHERE order_id = NEW.order_id;
END;
```

**4. Data Validation:**
```sql
-- Validate email format
CREATE TRIGGER validate_email
BEFORE INSERT ON employees
FOR EACH ROW
BEGIN
    IF NEW.email NOT LIKE '%@%.%' THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Invalid email format';
    END IF;
END;
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

