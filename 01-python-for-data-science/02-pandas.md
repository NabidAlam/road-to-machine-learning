# Pandas Complete Guide

Comprehensive guide to Pandas - the most important library for data manipulation and analysis in Python.

## Table of Contents

- [Introduction](#introduction)
- [Series and DataFrames](#series-and-dataframes)
- [Reading and Writing Data](#reading-and-writing-data)
- [Data Selection and Filtering](#data-selection-and-filtering)
- [Data Cleaning](#data-cleaning)
- [Grouping and Aggregation](#grouping-and-aggregation)
- [Merging and Joining](#merging-and-joining)
- [Time Series Operations](#time-series-operations)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is Pandas?

Pandas is a library for data manipulation and analysis. It provides:
- **DataFrame**: 2D labeled data structure (like Excel spreadsheet)
- **Series**: 1D labeled array
- **Powerful tools**: For cleaning, transforming, and analyzing data

### Why Pandas?

- **Easy data manipulation**: Load, clean, transform data easily
- **Handles missing data**: Built-in functions for dealing with NaN
- **Time series**: Excellent support for time-based data
- **Integration**: Works seamlessly with NumPy, Matplotlib, Scikit-learn

### Installation

```python
pip install pandas
```

```python
import pandas as pd
import numpy as np
print(pd.__version__)
```

---

## Series and DataFrames

### Series

A Series is a one-dimensional labeled array.

```python
# Creating Series
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# Output:
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# Output:
# a    10
# b    20
# c    30
# dtype: int64

# From dictionary
s = pd.Series({'Alice': 25, 'Bob': 30, 'Charlie': 35})
print(s)
```

### DataFrame

A DataFrame is a 2D labeled data structure with columns of potentially different types.

```python
# Creating DataFrame from dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)
print(df)
# Output:
#       Name  Age      City  Salary
# 0    Alice   25  New York   50000
# 1      Bob   30    London   60000
# 2  Charlie   35     Tokyo   70000
# 3    Diana   28     Paris   55000

# From list of lists
data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df = pd.DataFrame(data, columns=['Name', 'Age'])
print(df)

# From NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df)
```

### DataFrame Properties

```python
df = pd.DataFrame(data)

print(df.shape)        # Output: (4, 4) - (rows, columns)
print(df.size)         # Output: 16 - total elements
print(df.columns)      # Output: Index(['Name', 'Age', 'City', 'Salary'])
print(df.index)        # Output: RangeIndex(start=0, stop=4, step=1)
print(df.dtypes)       # Data types of each column
print(df.info())       # Summary information
print(df.describe())   # Statistical summary
```

---

## Reading and Writing Data

### Reading CSV Files

```python
# Read CSV
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv', 
                 sep=',',           # Separator
                 header=0,          # Row to use as header
                 index_col=0,       # Column to use as index
                 na_values=['NA', 'N/A'],  # Values to treat as NaN
                 nrows=1000)        # Read only first 1000 rows

# Read with specific columns
df = pd.read_csv('data.csv', usecols=['Name', 'Age', 'Salary'])
```

### Reading Other Formats

```python
# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')

# HTML tables
df = pd.read_html('https://example.com/table.html')[0]

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)
```

### Writing Data

```python
# Write to CSV
df.to_csv('output.csv', index=False)

# Write to Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Write to JSON
df.to_json('output.json', orient='records')

# Write to SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

---

## Data Selection and Filtering

### Selecting Columns

```python
df = pd.DataFrame(data)

# Single column (returns Series)
ages = df['Age']
print(ages)

# Multiple columns (returns DataFrame)
subset = df[['Name', 'Age']]
print(subset)

# Using dot notation (only if column name is valid Python identifier)
ages = df.Age
```

### Selecting Rows

```python
# By index
first_row = df.iloc[0]        # First row
first_three = df.iloc[0:3]    # First 3 rows

# By label
row = df.loc[0]               # Row with index 0

# First/last n rows
head = df.head(3)             # First 3 rows
tail = df.tail(3)             # Last 3 rows
```

### Filtering

```python
# Boolean indexing
young = df[df['Age'] < 30]
print(young)
# Output:
#     Name  Age      City  Salary
# 0  Alice   25  New York   50000
# 3  Diana   28     Paris   55000

# Multiple conditions
filtered = df[(df['Age'] > 25) & (df['Salary'] > 55000)]
print(filtered)

# Using query method
filtered = df.query('Age > 25 and Salary > 55000')
print(filtered)

# isin() method
cities = ['New York', 'London']
filtered = df[df['City'].isin(cities)]
print(filtered)
```

### iloc vs loc

```python
# iloc: Integer position-based indexing
df.iloc[0, 1]        # Row 0, Column 1
df.iloc[0:3, 1:3]    # Rows 0-2, Columns 1-2

# loc: Label-based indexing
df.loc[0, 'Age']     # Row with index 0, column 'Age'
df.loc[0:2, 'Name':'City']  # Rows 0-2, columns Name to City
```

---

## Data Cleaning

### Handling Missing Values

```python
# Check for missing values
print(df.isnull())           # Boolean DataFrame
print(df.isnull().sum())     # Count missing per column
print(df.isnull().any())     # True if any missing in column

# Drop missing values
df_clean = df.dropna()              # Drop rows with any NaN
df_clean = df.dropna(axis=1)        # Drop columns with any NaN
df_clean = df.dropna(subset=['Age'])  # Drop rows where Age is NaN

# Fill missing values
df_filled = df.fillna(0)            # Fill with 0
df_filled = df.fillna(df.mean())    # Fill with mean
df_filled = df['Age'].fillna(df['Age'].mean())  # Fill specific column

# Forward fill / Backward fill
df_ffill = df.fillna(method='ffill')  # Forward fill
df_bfill = df.fillna(method='bfill')  # Backward fill
```

### Handling Duplicates

```python
# Check for duplicates
print(df.duplicated())      # Boolean Series
print(df.duplicated().sum()) # Count duplicates

# Drop duplicates
df_unique = df.drop_duplicates()              # Drop all duplicates
df_unique = df.drop_duplicates(subset=['Name'])  # Drop based on column
```

### Data Types

```python
# Check data types
print(df.dtypes)

# Convert data types
df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(float)

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert to category (saves memory)
df['City'] = df['City'].astype('category')
```

### String Operations

```python
# String methods (on string columns)
df['Name'].str.upper()           # Convert to uppercase
df['Name'].str.lower()           # Convert to lowercase
df['Name'].str.strip()           # Remove whitespace
df['Name'].str.replace(' ', '_') # Replace characters
df['Name'].str.contains('Alice') # Check if contains substring
df['Name'].str.split(' ')       # Split string
```

---

## Grouping and Aggregation

### GroupBy

```python
# Group by column
grouped = df.groupby('City')

# Apply aggregation functions
city_stats = grouped.agg({
    'Age': 'mean',
    'Salary': ['mean', 'sum', 'count']
})
print(city_stats)

# Common aggregations
grouped.mean()      # Mean of each group
grouped.sum()       # Sum of each group
grouped.count()     # Count in each group
grouped.size()      # Size of each group
grouped.min()       # Minimum
grouped.max()       # Maximum
grouped.std()       # Standard deviation
```

### Custom Aggregations

```python
# Multiple aggregations
agg_dict = {
    'Age': ['mean', 'min', 'max'],
    'Salary': ['sum', 'mean']
}
result = df.groupby('City').agg(agg_dict)
print(result)

# Custom function
def range_func(x):
    return x.max() - x.min()

result = df.groupby('City')['Age'].agg(range_func)
print(result)
```

### Pivot Tables

```python
# Create pivot table
pivot = df.pivot_table(
    values='Salary',
    index='City',
    columns='Age',
    aggfunc='mean'
)
print(pivot)

# With multiple aggregations
pivot = df.pivot_table(
    values=['Age', 'Salary'],
    index='City',
    aggfunc={'Age': 'mean', 'Salary': 'sum'}
)
```

---

## Merging and Joining

### Merge (SQL-like joins)

```python
# Sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 4],
    'Salary': [50000, 60000, 70000]
})

# Inner join (default)
merged = pd.merge(df1, df2, on='ID', how='inner')
print(merged)
# Output:
#    ID     Name  Salary
# 0   1    Alice   50000
# 1   2      Bob   60000

# Left join
merged = pd.merge(df1, df2, on='ID', how='left')
print(merged)

# Right join
merged = pd.merge(df1, df2, on='ID', how='right')
print(merged)

# Outer join
merged = pd.merge(df1, df2, on='ID', how='outer')
print(merged)
```

### Concatenation

```python
# Concatenate DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Vertical concatenation
result = pd.concat([df1, df2], axis=0)
print(result)

# Horizontal concatenation
result = pd.concat([df1, df2], axis=1)
print(result)

# With keys
result = pd.concat([df1, df2], keys=['first', 'second'])
print(result)
```

---

## Time Series Operations

### Working with Dates

```python
# Create date range
dates = pd.date_range('2024-01-01', periods=10, freq='D')
print(dates)

# Set date as index
df['Date'] = pd.date_range('2024-01-01', periods=len(df))
df = df.set_index('Date')
print(df)

# Resampling
df_daily = df.resample('D').mean()      # Daily
df_weekly = df.resample('W').mean()     # Weekly
df_monthly = df.resample('M').mean()    # Monthly

# Time-based filtering
df_jan = df[df.index.month == 1]       # January data
df_2024 = df[df.index.year == 2024]     # 2024 data
```

### Time Series Operations

```python
# Shift values
df['Previous'] = df['Value'].shift(1)   # Previous value
df['Next'] = df['Value'].shift(-1)      # Next value

# Rolling window
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()
df['Rolling_Std'] = df['Value'].rolling(window=7).std()

# Expanding window
df['Expanding_Mean'] = df['Value'].expanding().mean()
```

---

## Practice Exercises

### Exercise 1: Data Loading and Inspection

**Task:** Load a CSV file, inspect its structure, and display basic statistics.

**Solution:**
```python
df = pd.read_csv('data.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nStatistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
```

### Exercise 2: Data Filtering

**Task:** Filter data where age > 30 and salary > 60000, then select only Name and City columns.

**Solution:**
```python
filtered = df[(df['Age'] > 30) & (df['Salary'] > 60000)]
result = filtered[['Name', 'City']]
print(result)
```

### Exercise 3: Grouping and Aggregation

**Task:** Group by City and calculate average age and total salary for each city.

**Solution:**
```python
result = df.groupby('City').agg({
    'Age': 'mean',
    'Salary': 'sum'
})
print(result)
```

### Exercise 4: Data Cleaning

**Task:** Handle missing values by filling numeric columns with mean and dropping rows with missing categorical data.

**Solution:**
```python
# Fill numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Drop rows with missing categorical data
categorical_cols = df.select_dtypes(include=['object']).columns
df = df.dropna(subset=categorical_cols)
```

### Exercise 5: Merging DataFrames

**Task:** Merge two DataFrames on a common key and handle missing values.

**Solution:**
```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Value': [10, 20, 30]})

merged = pd.merge(df1, df2, on='ID', how='outer')
merged = merged.fillna(0)  # Fill missing with 0
print(merged)
```

---

## Key Takeaways

1. **DataFrames are powerful** - Think of them as Excel spreadsheets on steroids
2. **Boolean indexing** - Powerful way to filter data
3. **GroupBy** - Essential for data analysis
4. **Handle missing data** - Always check and clean your data
5. **Practice** - Work with real datasets to master Pandas

---

## Common Patterns

### Pattern 1: Data Exploration

```python
# Complete data exploration
def explore_data(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary statistics:\n", df.describe())
    print("\nFirst few rows:\n", df.head())
    return df.info()

explore_data(df)
```

### Pattern 2: Data Cleaning Pipeline

```python
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Remove outliers (example: values beyond 3 standard deviations)
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    return df
```

---

## Next Steps

- Practice with real datasets (Kaggle, UCI Repository)
- Work through the exercises
- Move to [03-visualization.md](03-visualization.md) to learn data visualization

**Remember**: Pandas is your primary tool for data manipulation - master it!

