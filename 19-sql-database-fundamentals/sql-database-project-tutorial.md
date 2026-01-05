# SQL Project Tutorial

Step-by-step SQL project walkthrough.

## Project: Employee Database Analysis

### Step 1: Create Database

```sql
CREATE DATABASE company_db;
USE company_db;
```

### Step 2: Create Tables

```sql
CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50)
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    salary DECIMAL(10,2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### Step 3: Analyze Data

```sql
-- Average salary by department
SELECT d.department_name, AVG(e.salary) AS avg_salary
FROM employees e
JOIN departments d ON e.department_id = d.department_id
GROUP BY d.department_name;
```

---

**Congratulations!** You've completed a SQL project!

