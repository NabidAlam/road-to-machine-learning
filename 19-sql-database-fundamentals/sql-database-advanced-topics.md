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

## NoSQL Databases

### Introduction to NoSQL

NoSQL (Not Only SQL) databases are non-relational databases designed for specific data models and have flexible schemas. They're often used for:
- Large-scale applications
- Real-time data
- Unstructured data
- High-performance requirements

### When to Use NoSQL vs SQL

**Use SQL when:**
- Structured data with relationships
- ACID transactions required
- Complex queries and joins
- Data consistency is critical
- Traditional business applications

**Use NoSQL when:**
- Unstructured or semi-structured data
- High scalability needed
- Fast read/write performance
- Flexible schema requirements
- Big data applications

### Types of NoSQL Databases

#### 1. Document Databases (MongoDB)

Store data as documents (JSON-like structures).

**MongoDB Example:**
```python
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
collection = db['users']

# Insert document
user = {
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30,
    'address': {
        'street': '123 Main St',
        'city': 'New York'
    }
}
collection.insert_one(user)

# Query documents
users = collection.find({'age': {'$gt': 25}})
for user in users:
    print(user)

# Update document
collection.update_one(
    {'name': 'John Doe'},
    {'$set': {'age': 31}}
)
```

**Use Cases:**
- Content management
- User profiles
- Catalogs
- Real-time analytics

#### 2. Key-Value Stores (Redis)

Simple key-value pairs, extremely fast.

**Redis Example:**
```python
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Set value
r.set('user:1', 'John Doe')
r.set('user:1:email', 'john@example.com')

# Get value
name = r.get('user:1')
print(name.decode('utf-8'))

# Set with expiration
r.setex('session:abc123', 3600, 'user_data')

# Lists
r.lpush('tasks', 'task1', 'task2', 'task3')
tasks = r.lrange('tasks', 0, -1)
```

**Use Cases:**
- Caching
- Session storage
- Real-time leaderboards
- Message queues

#### 3. Column-Family Stores (Cassandra)

Store data in columns grouped by column families.

**Cassandra Example:**
```python
from cassandra.cluster import Cluster

# Connect to Cassandra
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('mykeyspace')

# Insert data
session.execute(
    "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
    (1, 'John Doe', 'john@example.com')
)

# Query data
rows = session.execute("SELECT * FROM users WHERE id = ?", (1,))
for row in rows:
    print(row.name, row.email)
```

**Use Cases:**
- Time-series data
- IoT applications
- High write throughput
- Distributed systems

#### 4. Graph Databases (Neo4j)

Store data as nodes and relationships.

**Neo4j Example:**
```python
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def create_person(tx, name):
    tx.run("CREATE (p:Person {name: $name})", name=name)

def create_friendship(tx, name1, name2):
    tx.run("""
        MATCH (a:Person {name: $name1})
        MATCH (b:Person {name: $name2})
        CREATE (a)-[:FRIENDS_WITH]->(b)
    """, name1=name1, name2=name2)

with driver.session() as session:
    session.write_transaction(create_person, "Alice")
    session.write_transaction(create_person, "Bob")
    session.write_transaction(create_friendship, "Alice", "Bob")
```

**Use Cases:**
- Social networks
- Recommendation systems
- Fraud detection
- Knowledge graphs

### NoSQL with Python

**MongoDB:**
```bash
pip install pymongo
```

**Redis:**
```bash
pip install redis
```

**Cassandra:**
```bash
pip install cassandra-driver
```

**Neo4j:**
```bash
pip install neo4j
```

### Comparison Table

| Database | Type | Best For | Python Library |
|----------|------|----------|----------------|
| **MongoDB** | Document | Flexible schemas | pymongo |
| **Redis** | Key-Value | Caching, sessions | redis |
| **Cassandra** | Column | Time-series, IoT | cassandra-driver |
| **Neo4j** | Graph | Relationships | neo4j |

### Choosing the Right Database

**Questions to Ask:**
1. What is the data structure?
2. What are the access patterns?
3. What is the scale requirement?
4. What consistency level is needed?
5. What is the query complexity?

---

## Key Takeaways

1. **Optimize Queries**: Use indexes and EXPLAIN
2. **Window Functions**: Powerful for analytics
3. **Transactions**: Ensure data consistency
4. **NoSQL**: Choose based on use case and data structure

---

**Remember**: Advanced SQL skills make you a better data scientist! NoSQL complements SQL for specific use cases.

