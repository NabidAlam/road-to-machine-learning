# Python for Data Science Quick Reference

Quick lookup guide for NumPy, Pandas, Matplotlib, and data manipulation operations.

## Table of Contents

- [NumPy Quick Reference](#numpy-quick-reference)
- [Pandas Quick Reference](#pandas-quick-reference)
- [Visualization Quick Reference](#visualization-quick-reference)
- [Common Data Manipulation Patterns](#common-data-manipulation-patterns)
- [File I/O Quick Reference](#file-io-quick-reference)

---

## NumPy Quick Reference

### Array Creation

```python
import numpy as np

np.array([1, 2, 3])              # From list
np.zeros((3, 4))                 # Zeros
np.ones((3, 4))                  # Ones
np.full((3, 4), 5)               # Fill value
np.arange(0, 10, 2)               # Range
np.linspace(0, 1, 10)            # Linear space
np.random.rand(3, 4)             # Random [0, 1)
np.random.randn(3, 4)            # Random normal
np.random.randint(0, 10, (3, 4)) # Random integers
np.eye(3)                         # Identity matrix
```

### Array Operations

```python
arr.shape                        # Dimensions
arr.size                         # Total elements
arr.dtype                        # Data type
arr.reshape(3, 4)                # Reshape
arr.flatten()                    # Flatten
arr.T                            # Transpose
np.concatenate([arr1, arr2])     # Concatenate
np.vstack([arr1, arr2])          # Vertical stack
np.hstack([arr1, arr2])          # Horizontal stack
```

### Mathematical Operations

```python
np.sum(arr)                      # Sum
np.mean(arr)                     # Mean
np.std(arr)                      # Standard deviation
np.min(arr)                      # Minimum
np.max(arr)                      # Maximum
np.argmin(arr)                   # Index of min
np.argmax(arr)                   # Index of max
np.sqrt(arr)                     # Square root
np.exp(arr)                      # Exponential
np.log(arr)                      # Natural log
np.abs(arr)                      # Absolute value
```

### Linear Algebra

```python
np.dot(A, B)                     # Dot product
A @ B                            # Matrix multiplication
np.linalg.inv(A)                 # Inverse
np.linalg.det(A)                 # Determinant
np.linalg.eig(A)                 # Eigenvalues/vectors
np.linalg.svd(A)                 # SVD
np.linalg.norm(A)                # Norm
np.linalg.solve(A, b)            # Solve Ax=b
```

---

## Pandas Quick Reference

### DataFrame Creation

```python
import pandas as pd

pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})  # From dict
pd.read_csv('file.csv')                         # From CSV
pd.read_excel('file.xlsx')                      # From Excel
pd.read_json('file.json')                       # From JSON
pd.read_sql(query, connection)                  # From SQL
```

### Data Selection

```python
df['col']                        # Select column
df[['col1', 'col2']]            # Select multiple columns
df.loc[row_index]               # Select row by label
df.iloc[row_index]              # Select row by position
df.loc[row, 'col']              # Select cell
df[df['col'] > 5]               # Filter rows
df.query('col > 5')             # Query syntax
```

### Data Manipulation

```python
df.dropna()                     # Remove NaN
df.fillna(value)                # Fill NaN
df.drop_duplicates()            # Remove duplicates
df.sort_values('col')           # Sort
df.groupby('col').mean()        # Group and aggregate
df.merge(df2, on='key')         # Merge
df.join(df2)                    # Join
df.concat([df1, df2])           # Concatenate
df.pivot_table(values='val', index='idx', columns='col')  # Pivot
```

### Common Operations

```python
df.head()                        # First 5 rows
df.tail()                        # Last 5 rows
df.info()                        # Info
df.describe()                    # Statistics
df.value_counts()               # Value counts
df.nunique()                     # Unique values count
df.isnull().sum()               # Count nulls
df.corr()                        # Correlation matrix
df.cov()                         # Covariance matrix
```

---

## Visualization Quick Reference

### Matplotlib

```python
import matplotlib.pyplot as plt

plt.plot(x, y)                   # Line plot
plt.scatter(x, y)               # Scatter plot
plt.bar(x, y)                   # Bar chart
plt.hist(data, bins=30)         # Histogram
plt.boxplot(data)               # Box plot
plt.pie(sizes, labels=labels)   # Pie chart
plt.xlabel('X Label')           # X-axis label
plt.ylabel('Y Label')           # Y-axis label
plt.title('Title')              # Title
plt.legend()                     # Legend
plt.grid(True)                  # Grid
plt.show()                       # Show plot
plt.savefig('plot.png')         # Save plot
```

### Seaborn

```python
import seaborn as sns

sns.scatterplot(x='x', y='y', data=df)      # Scatter
sns.lineplot(x='x', y='y', data=df)        # Line
sns.barplot(x='x', y='y', data=df)         # Bar
sns.histplot(data=df, x='col')              # Histogram
sns.boxplot(x='x', y='y', data=df)         # Box plot
sns.violinplot(x='x', y='y', data=df)      # Violin
sns.heatmap(corr_matrix, annot=True)       # Heatmap
sns.pairplot(df)                           # Pair plot
sns.countplot(x='col', data=df)            # Count plot
```

### Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

px.scatter(df, x='x', y='y')                # Scatter
px.line(df, x='x', y='y')                   # Line
px.bar(df, x='x', y='y')                    # Bar
px.histogram(df, x='col')                   # Histogram
px.box(df, x='x', y='y')                    # Box
px.heatmap(corr_matrix)                     # Heatmap
fig.show()                                  # Show
fig.write_html('plot.html')                 # Save
```

---

## Common Data Manipulation Patterns

### Data Cleaning

```python
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()                           # Drop rows with NaN
df = df.fillna(0)                          # Fill with value
df = df.fillna(df.mean())                  # Fill with mean

# Remove outliers
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]
```

### Feature Engineering

```python
# Create new columns
df['new_col'] = df['col1'] + df['col2']
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Categorical encoding
df = pd.get_dummies(df, columns=['cat_col'])

# Binning
df['binned'] = pd.cut(df['col'], bins=5)
```

### Grouping and Aggregation

```python
# Basic grouping
df.groupby('col').mean()
df.groupby('col').agg({'col1': 'mean', 'col2': 'sum'})

# Multiple aggregations
df.groupby('col').agg({
    'col1': ['mean', 'std', 'min', 'max'],
    'col2': 'sum'
})

# Custom function
df.groupby('col').apply(lambda x: x['val'].sum())
```

### Merging

```python
# Inner join
pd.merge(df1, df2, on='key')

# Left/Right/Outer join
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, on='key', how='right')
pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])
```

---

## File I/O Quick Reference

### Reading Files

```python
# CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';')      # Custom separator
df = pd.read_csv('file.csv', header=None)  # No header

# Excel
df = pd.read_excel('file.xlsx')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('file.json')
df = pd.read_json('file.json', orient='records')

# Parquet
df = pd.read_parquet('file.parquet')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)
```

### Writing Files

```python
# CSV
df.to_csv('file.csv', index=False)

# Excel
df.to_excel('file.xlsx', index=False)
df.to_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df.to_json('file.json', orient='records')

# Parquet
df.to_parquet('file.parquet')

# SQL
df.to_sql('table_name', connection, if_exists='replace')
```

---

## Performance Tips

### Vectorization

```python
# Slow: Loop
result = []
for x in df['col']:
    result.append(x * 2)

# Fast: Vectorized
result = df['col'] * 2
```

### Efficient Dtypes

```python
# Check memory
df.memory_usage(deep=True)

# Convert to efficient types
df['int_col'] = df['int_col'].astype('int32')
df['float_col'] = df['float_col'].astype('float32')
df['cat_col'] = df['cat_col'].astype('category')
```

### Chunk Processing

```python
# Process large files in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

---

## Common Errors and Solutions

### KeyError

```python
# Check if column exists
if 'col' in df.columns:
    value = df['col']
```

### SettingWithCopyWarning

```python
# Use .copy() or .loc
df_new = df.copy()
df_new['col'] = value

# Or
df.loc[:, 'col'] = value
```

### Memory Error

```python
# Process in chunks
chunks = []
for chunk in pd.read_csv('file.csv', chunksize=10000):
    processed = process(chunk)
    chunks.append(processed)
df = pd.concat(chunks)
```

---

**Remember**: This is a quick reference. For detailed explanations, see the main guides!

