# Advanced Data Wrangling - Complete Guide

Comprehensive guide to advanced data manipulation techniques: reshaping, pivoting, transforming, and optimizing data operations.

## Table of Contents

- [Introduction](#introduction)
- [Reshaping Data](#reshaping-data)
- [Pivot Tables](#pivot-tables)
- [Melt and Unpivot](#melt-and-unpivot)
- [Stack and Unstack](#stack-and-unstack)
- [Advanced Grouping](#advanced-grouping)
- [Performance Optimization](#performance-optimization)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is Data Wrangling?

**Data Wrangling** (also called data munging) is the process of transforming and mapping data from one format to another to make it more appropriate for analysis.

### Why Advanced Wrangling Matters

- **Reshape data**: Convert between wide and long formats
- **Aggregate efficiently**: Create summary tables
- **Handle complex structures**: Work with hierarchical data
- **Optimize performance**: Process large datasets efficiently

---

## Reshaping Data

### Wide vs Long Format

**Wide Format**: Each variable has its own column
```python
# Wide format
df_wide = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Q1': [100, 150],
    'Q2': [120, 160],
    'Q3': [130, 170],
    'Q4': [140, 180]
})
```

**Long Format**: One column for variable names, one for values
```python
# Long format
df_long = pd.DataFrame({
    'Name': ['Alice', 'Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 'Bob'],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4'],
    'Sales': [100, 120, 130, 140, 150, 160, 170, 180]
})
```

---

## Pivot Tables

### Creating Pivot Tables

```python
# Sample data
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'Product': ['A', 'B', 'A', 'B'] * 3,
    'Region': ['North', 'North', 'South', 'South'] * 3,
    'Sales': [100, 150, 200, 250, 120, 180, 220, 280, 110, 160, 210, 270]
})

# Basic pivot table
pivot = df.pivot_table(
    values='Sales',
    index='Date',
    columns='Product',
    aggfunc='sum'
)
print(pivot)

# Multiple aggregations
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc=['sum', 'mean', 'count']
)
print(pivot)

# With margins (totals)
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)
print(pivot)
```

### Advanced Pivot Operations

```python
# Multiple value columns
df = pd.DataFrame({
    'Date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180],
    'Quantity': [10, 15, 12, 18]
})

pivot = df.pivot_table(
    values=['Sales', 'Quantity'],
    index='Date',
    columns='Product',
    aggfunc='sum'
)
print(pivot)

# Custom aggregation functions
pivot = df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc=[np.sum, np.mean, lambda x: x.max() - x.min()]  # sum, mean, range
)
```

---

## Melt and Unpivot

### Melt: Wide to Long

```python
# Wide format
df_wide = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Math': [85, 90, 75],
    'Science': [88, 92, 80],
    'English': [82, 85, 78]
})

# Melt to long format
df_long = df_wide.melt(
    id_vars='Name',
    value_vars=['Math', 'Science', 'English'],
    var_name='Subject',
    value_name='Score'
)
print(df_long)
# Output:
#       Name  Subject  Score
# 0    Alice     Math     85
# 1      Bob     Math     90
# 2  Charlie     Math     75
# 3    Alice  Science     88
# ...

# Melt all columns except id_vars
df_long = df_wide.melt(
    id_vars='Name',
    var_name='Subject',
    value_name='Score'
)
```

### Unpivot Multiple Columns

```python
# Wide format with multiple value columns
df_wide = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Q1_Sales': [100, 150],
    'Q1_Profit': [20, 30],
    'Q2_Sales': [120, 160],
    'Q2_Profit': [24, 32]
})

# Melt with multiple value columns
df_long = df_wide.melt(
    id_vars='Name',
    value_vars=['Q1_Sales', 'Q1_Profit', 'Q2_Sales', 'Q2_Profit'],
    var_name='Metric',
    value_name='Value'
)

# Or split metric into quarter and type
df_long['Quarter'] = df_long['Metric'].str[:2]
df_long['Type'] = df_long['Metric'].str[3:]
df_long = df_long.drop('Metric', axis=1)
print(df_long)
```

---

## Stack and Unstack

### Stack: Columns to Rows

```python
# Create hierarchical columns
df = pd.DataFrame({
    ('Sales', 'Q1'): [100, 150],
    ('Sales', 'Q2'): [120, 160],
    ('Profit', 'Q1'): [20, 30],
    ('Profit', 'Q2'): [24, 32]
}, index=['Product A', 'Product B'])

# Stack
stacked = df.stack()
print(stacked)
# Output:
# Product A  Sales   Q1    100
#                   Q2    120
#            Profit  Q1     20
#                   Q2     24
# Product B  Sales   Q1    150
# ...

# Stack specific level
stacked = df.stack(level=0)  # Stack first level
```

### Unstack: Rows to Columns

```python
# From stacked format back to wide
unstacked = stacked.unstack()
print(unstacked)

# Unstack specific level
unstacked = stacked.unstack(level=0)
```

---

## Advanced Grouping

### Multiple Grouping Levels

```python
df = pd.DataFrame({
    'Region': ['North', 'North', 'South', 'South'] * 3,
    'Product': ['A', 'B', 'A', 'B'] * 3,
    'Date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'Sales': [100, 150, 200, 250, 120, 180, 220, 280, 110, 160, 210, 270]
})

# Group by multiple columns
grouped = df.groupby(['Region', 'Product'])['Sales'].agg(['sum', 'mean', 'count'])
print(grouped)

# Group by with custom functions
def range_func(x):
    return x.max() - x.min()

grouped = df.groupby(['Region', 'Product'])['Sales'].agg(['sum', range_func])
```

### Transform and Apply

```python
# Transform: Return same shape as input
df['Sales_mean_by_region'] = df.groupby('Region')['Sales'].transform('mean')
df['Sales_normalized'] = df.groupby('Region')['Sales'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Apply: Custom function
def custom_agg(group):
    return pd.Series({
        'total': group['Sales'].sum(),
        'avg': group['Sales'].mean(),
        'count': len(group)
    })

result = df.groupby('Region').apply(custom_agg)
print(result)
```

### Named Aggregation

```python
# Named aggregations (clearer column names)
result = df.groupby('Region').agg(
    total_sales=('Sales', 'sum'),
    avg_sales=('Sales', 'mean'),
    sales_count=('Sales', 'count')
)
print(result)
```

---

## Performance Optimization

### Vectorization

```python
# Slow: Using apply with loops
def slow_method(df):
    result = []
    for idx, row in df.iterrows():
        result.append(row['col1'] * 2 + row['col2'])
    return result

# Fast: Vectorized operations
def fast_method(df):
    return df['col1'] * 2 + df['col2']

# Even faster: NumPy operations
def faster_method(df):
    return np.array(df['col1']) * 2 + np.array(df['col2'])
```

### Efficient Filtering

```python
# Slow: Multiple filters
result = df[df['col1'] > 10]
result = result[result['col2'] < 20]
result = result[result['col3'] == 'value']

# Fast: Combined conditions
result = df[(df['col1'] > 10) & (df['col2'] < 20) & (df['col3'] == 'value')]
```

### Chunk Processing

```python
# Process large files in chunks
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = chunk.groupby('category').sum()
    results.append(processed)

# Combine results
final_result = pd.concat(results, ignore_index=True)
```

### Categorical Data Types

```python
# Convert to category for memory efficiency
df['category'] = df['category'].astype('category')

# Benefits:
# - Less memory usage
# - Faster operations
# - Maintains order
```

---

## Practice Exercises

### Exercise 1: Reshape Sales Data

Convert quarterly sales data from wide to long format and back.

```python
# Your solution here
```

### Exercise 2: Create Pivot Table

Create a pivot table showing average sales by region and product.

```python
# Your solution here
```

### Exercise 3: Advanced Grouping

Group data by multiple levels and calculate custom aggregations.

```python
# Your solution here
```

---

## Resources

### Documentation

- [Pandas Reshaping](https://pandas.pydata.org/docs/user_guide/reshaping.html)
- [Pandas GroupBy](https://pandas.pydata.org/docs/user_guide/groupby.html)

### Books

- **"Python for Data Analysis"** by Wes McKinney
  - Comprehensive guide to pandas operations

---

## Key Takeaways

1. **Reshape When Needed**: Convert between wide and long formats
2. **Pivot Tables**: Powerful for summarization
3. **GroupBy is Essential**: Master grouping operations
4. **Optimize Performance**: Use vectorization and efficient methods
5. **Practice**: Work with real datasets to master these techniques

---

**Remember**: Advanced data wrangling skills are crucial for real-world data science. Practice reshaping and transforming data!

