# Pandas Complete Guide

Comprehensive guide to Pandas - the most important library for data manipulation and analysis in Python.

## Table of Contents

- [Introduction](#introduction)
- [Series and DataFrames](#series-and-dataframes)
- [DataFrame Basics and Inspection](#dataframe-basics-and-inspection)
- [Reading and Writing Data](#reading-and-writing-data)
- [Data Selection and Filtering](#data-selection-and-filtering)
- [Data Cleaning](#data-cleaning)
- [Sorting and Basic Statistics](#sorting-and-basic-statistics)
- [Grouping and Aggregation](#grouping-and-aggregation)
- [Merging and Joining](#merging-and-joining)
- [Time Series Operations](#time-series-operations)
- [Advanced Tricks and Performance](#advanced-tricks-and-performance)
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

## DataFrame Basics and Inspection

### Inspecting Data

Quickly understand your dataset using inspection functions:

```python
df = pd.DataFrame(data)

# First few rows
print(df.head())       # First 5 rows (default)
print(df.head(3))     # First 3 rows

# Last few rows
print(df.tail())      # Last 5 rows (default)
print(df.tail(3))     # Last 3 rows

# Dataset structure
print(df.shape)       # Output: (4, 4) - (rows, columns)
print(df.size)        # Output: 16 - total elements
print(df.ndim)        # Output: 2 - number of dimensions

# Data types
print(df.dtypes)      # Data type of each column
# Output:
# Name      object
# Age        int64
# City      object
# Salary     int64

# Summary information
print(df.info())      # Detailed info about DataFrame
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 4 columns):
#  Name      4 non-null object
#  Age       4 non-null int64
#  ...

# Statistical summary (numeric columns only)
print(df.describe())
# Output:
#              Age        Salary
# count   4.000000      4.000000
# mean   29.500000  58750.000000
# std     4.041452   8164.965809
# min    25.000000  50000.000000
# 25%    27.250000  53750.000000
# 50%    29.000000  57500.000000
# 75%    32.250000  62500.000000
# max    35.000000  70000.000000
```

### Index & Columns Overview

Explore DataFrame components:

```python
# View index
print(df.index)        # Output: RangeIndex(start=0, stop=4, step=1)
print(df.index.tolist())  # [0, 1, 2, 3]

# View columns
print(df.columns)      # Output: Index(['Name', 'Age', 'City', 'Salary'])
print(df.columns.tolist())  # ['Name', 'Age', 'City', 'Salary']

# Underlying NumPy data
print(df.values)      # Returns NumPy array
# Output:
# [['Alice' 25 'New York' 50000]
#  ['Bob' 30 'London' 60000]
#  ...]

# Check if index/columns are unique
print(df.index.is_unique)    # True
print(df.columns.is_unique)  # True
```

### Renaming Columns

```python
# Rename single column
df = df.rename(columns={'Age': 'Years'})
print(df.columns)  # ['Name', 'Years', 'City', 'Salary']

# Rename multiple columns
df = df.rename(columns={
    'Name': 'Full Name',
    'City': 'Location',
    'Salary': 'Income'
})

# Rename using mapper function
df = df.rename(columns=str.lower)  # Convert to lowercase
df = df.rename(columns=str.upper)  # Convert to uppercase

# In-place renaming
df.rename(columns={'Age': 'Years'}, inplace=True)
```

### Setting and Resetting Index

```python
# Set a column as index
df_indexed = df.set_index('Name')
print(df_indexed)
# Output:
#          Age      City  Salary
# Name                          
# Alice    25  New York   50000
# Bob      30    London   60000
# ...

# Reset index (move index back to column)
df_reset = df_indexed.reset_index()
print(df_reset)
# Output:
#      Name  Age      City  Salary
# 0   Alice   25  New York   50000
# 1     Bob   30    London   60000
# ...

# Reset index without keeping old index as column
df_reset = df_indexed.reset_index(drop=True)

# Set multiple columns as index
df_multi = df.set_index(['City', 'Name'])
print(df_multi.index)  # MultiIndex
```

### Basic Attributes Recap

Quick reference for core attributes:

```python
# Structure
df.shape          # (rows, columns)
df.size           # Total elements
df.ndim           # Number of dimensions

# Components
df.columns        # Column names
df.index          # Row index
df.values         # NumPy array

# Data types
df.dtypes         # Data type per column
df.dtypes.value_counts()  # Count of each data type

# Memory usage
df.memory_usage()  # Memory usage per column
df.memory_usage(deep=True)  # Deep memory usage (includes objects)
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

Detect missing entries and handle them appropriately:

```python
# Check for missing values
print(df.isnull())           # Boolean DataFrame (True where NaN)
print(df.isna())             # Same as isnull()
print(df.isnull().sum())     # Count missing per column
print(df.isnull().any())     # True if any missing in column
print(df.isnull().any().any())  # True if any missing in entire DataFrame

# Drop missing values
df_clean = df.dropna()              # Drop rows with any NaN
df_clean = df.dropna(axis=1)        # Drop columns with any NaN
df_clean = df.dropna(subset=['Age'])  # Drop rows where Age is NaN
df_clean = df.dropna(how='all')     # Drop rows where ALL values are NaN
df_clean = df.dropna(thresh=2)      # Keep rows with at least 2 non-NaN values

# Fill missing values
df_filled = df.fillna(0)            # Fill with constant (0)
df_filled = df.fillna({'Age': 0, 'Salary': 50000})  # Fill different columns with different values
df_filled = df.fillna(df.mean())    # Fill with mean (numeric columns only)
df_filled = df['Age'].fillna(df['Age'].mean())  # Fill specific column with its mean
df_filled = df.fillna(df.median())  # Fill with median
df_filled = df.fillna(df.mode().iloc[0])  # Fill with mode

# Forward fill / Backward fill
df_ffill = df.fillna(method='ffill')  # Forward fill (use previous value)
df_bfill = df.fillna(method='bfill')  # Backward fill (use next value)
df_ffill_limit = df.fillna(method='ffill', limit=2)  # Limit forward fill to 2 consecutive NaNs

# Interpolation
df_interp = df.interpolate()  # Linear interpolation
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

### Changing Data Types (astype)

Convert columns to correct types for data compatibility:

```python
# Check data types
print(df.dtypes)

# Convert to integer
df['Age'] = df['Age'].astype(int)
df['Age'] = df['Age'].astype('int64')  # Explicit type

# Convert to float
df['Salary'] = df['Salary'].astype(float)
df['Salary'] = df['Salary'].astype('float64')

# Convert to string
df['Name'] = df['Name'].astype(str)
df['Name'] = df['Name'].astype('string')  # Pandas string type

# Convert to category (saves memory for repeated values)
df['City'] = df['City'].astype('category')
print(df['City'].cat.categories)  # View categories

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # With format

# Convert to boolean
df['IsActive'] = df['IsActive'].astype(bool)

# Convert multiple columns at once
df[['Age', 'Salary']] = df[['Age', 'Salary']].astype(float)

# Handle errors during conversion
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Convert invalid to NaN
df['Age'] = pd.to_numeric(df['Age'], errors='ignore')  # Leave invalid as-is
```

### Replacing Values (replace)

Clean or fix specific data points and typos:

```python
# Replace single value
df['City'] = df['City'].replace('New York', 'NYC')

# Replace multiple values
df['City'] = df['City'].replace(['New York', 'London'], ['NYC', 'LON'])

# Replace using dictionary
df = df.replace({'City': {'New York': 'NYC', 'London': 'LON'}})

# Replace in specific column
df['Age'] = df['Age'].replace(25, 26)

# Replace using mapping dictionary
mapping = {25: 26, 30: 31}
df['Age'] = df['Age'].replace(mapping)

# Replace using regex patterns
df['Name'] = df['Name'].replace(r'^A', 'Alpha', regex=True)  # Replace 'A' at start
df['City'] = df['City'].replace(r'York$', 'City', regex=True)  # Replace 'York' at end

# Replace NaN with specific value
df = df.replace(np.nan, 'Unknown')

# Replace multiple values with one
df['Status'] = df['Status'].replace(['Active', 'A', 'active'], 'ACTIVE')
```

### Dropping Rows & Columns (drop)

Remove unnecessary rows or columns:

```python
# Drop rows by index
df_dropped = df.drop(0)              # Drop row with index 0
df_dropped = df.drop([0, 2])          # Drop rows with indices 0 and 2
df_dropped = df.drop(df.index[0:2])   # Drop first 2 rows

# Drop columns
df_dropped = df.drop('City', axis=1)   # Drop column 'City'
df_dropped = df.drop(['City', 'Age'], axis=1)  # Drop multiple columns
df_dropped = df.drop(columns=['City', 'Age'])  # Alternative syntax

# Drop using axis parameter
df_dropped = df.drop('City', axis=1)  # axis=1 for columns
df_dropped = df.drop(0, axis=0)       # axis=0 for rows (default)

# In-place modification
df.drop('City', axis=1, inplace=True)  # Modify original DataFrame

# Drop rows/columns with errors handling
df_dropped = df.drop('NonExistent', axis=1, errors='ignore')  # Ignore if doesn't exist
```

### Detecting & Removing Duplicates

```python
# Check for duplicates
print(df.duplicated())           # Boolean Series (True for duplicates)
print(df.duplicated().sum())    # Count of duplicate rows

# Check duplicates in specific column
print(df.duplicated(subset=['Name']))

# Keep parameter: 'first' (default), 'last', or False
print(df.duplicated(keep='first'))   # Mark all but first as duplicate
print(df.duplicated(keep='last'))    # Mark all but last as duplicate
print(df.duplicated(keep=False))     # Mark all duplicates

# Drop duplicates
df_unique = df.drop_duplicates()              # Drop all duplicate rows
df_unique = df.drop_duplicates(keep='first')  # Keep first occurrence
df_unique = df.drop_duplicates(keep='last')   # Keep last occurrence
df_unique = df.drop_duplicates(keep=False)    # Drop all duplicates

# Drop duplicates based on specific columns
df_unique = df.drop_duplicates(subset=['Name'])  # Based on Name only
df_unique = df.drop_duplicates(subset=['Name', 'City'])  # Based on multiple columns

# In-place
df.drop_duplicates(inplace=True)
```

### String Operations

Use `.str` accessor for text-based filtering and cleaning:

```python
# Case conversion
df['Name'].str.upper()           # Convert to uppercase
df['Name'].str.lower()           # Convert to lowercase
df['Name'].str.title()           # Title case
df['Name'].str.capitalize()      # Capitalize first letter

# Whitespace handling
df['Name'].str.strip()           # Remove leading/trailing whitespace
df['Name'].str.lstrip()          # Remove leading whitespace
df['Name'].str.rstrip()          # Remove trailing whitespace

# String replacement
df['Name'].str.replace(' ', '_')  # Replace characters
df['Name'].str.replace(r'[0-9]', '', regex=True)  # Remove digits using regex

# String checking (returns boolean Series for filtering)
df['Name'].str.contains('Alice')      # Check if contains substring
df['Name'].str.startswith('A')        # Check if starts with
df['Name'].str.endswith('e')          # Check if ends with
df['Name'].str.match('^A.*')          # Check if matches pattern

# String splitting
df['Name'].str.split(' ')             # Split string into list
df['Name'].str.split(' ', expand=True)  # Split into separate columns
df['Name'].str.split(' ', n=1)        # Split into max n parts

# String extraction
df['Name'].str.extract(r'([A-Z])')    # Extract pattern
df['Name'].str.findall(r'[A-Z]')      # Find all matches

# String length
df['Name'].str.len()                  # Length of each string

# Filtering with string operations
filtered = df[df['Name'].str.contains('Alice')]
filtered = df[df['Name'].str.startswith('A')]
filtered = df[df['Name'].str.endswith('e')]
```

---

## Sorting and Basic Statistics

### Sorting Data

Sort data by one or multiple columns:

```python
# Sort by single column
df_sorted = df.sort_values('Age')              # Ascending (default)
df_sorted = df.sort_values('Age', ascending=False)  # Descending

# Sort by multiple columns
df_sorted = df.sort_values(['City', 'Age'])    # Sort by City, then Age
df_sorted = df.sort_values(['City', 'Age'], ascending=[True, False])  # Different order for each

# Sort by index
df_sorted = df.sort_index()                    # Sort by row index
df_sorted = df.sort_index(axis=1)              # Sort by column index

# In-place sorting
df.sort_values('Age', inplace=True)

# Handle NaN values
df_sorted = df.sort_values('Age', na_position='first')   # NaNs first
df_sorted = df.sort_values('Age', na_position='last')    # NaNs last (default)
```

### Descriptive Statistics

Summarize and understand numeric data:

```python
# Mean (average)
print(df['Age'].mean())         # Mean of Age column
print(df.mean())                # Mean of all numeric columns

# Median
print(df['Age'].median())       # Median of Age column
print(df.median())              # Median of all numeric columns

# Mode
print(df['City'].mode())        # Most frequent value(s)
print(df.mode())                 # Mode for each column

# Standard deviation
print(df['Age'].std())          # Standard deviation
print(df.std())                 # Std for all numeric columns

# Variance
print(df['Age'].var())          # Variance
print(df.var())                 # Variance for all numeric columns

# Min and Max
print(df['Age'].min())          # Minimum value
print(df['Age'].max())          # Maximum value
print(df.min())                 # Min for all columns
print(df.max())                 # Max for all columns

# Comprehensive summary
print(df.describe())            # Summary statistics for numeric columns
print(df.describe(include='all'))  # Include categorical columns

# Additional statistics
print(df['Age'].quantile(0.25))  # 25th percentile (Q1)
print(df['Age'].quantile(0.75))  # 75th percentile (Q3)
print(df['Age'].quantile([0.25, 0.5, 0.75]))  # Multiple percentiles
print(df['Age'].skew())         # Skewness
print(df['Age'].kurtosis())     # Kurtosis
```

### Counting Values

Find frequency or percentage of unique values:

```python
# Count unique values (frequency)
print(df['City'].value_counts())
# Output:
# New York    2
# London      1
# Tokyo       1

# With percentages
print(df['City'].value_counts(normalize=True))
# Output:
# New York    0.500
# London      0.250
# Tokyo       0.250

# Sort by frequency
print(df['City'].value_counts(ascending=True))  # Ascending order

# Count including NaN
print(df['City'].value_counts(dropna=False))    # Include NaN count

# Count unique values
print(df['City'].nunique())     # Number of unique values
print(df.nunique())             # Unique count per column

# Total count
print(df['City'].count())       # Count non-null values
print(df.count())               # Count per column
```

### Correlation & Covariance

Understand relationships between numerical columns:

```python
# Correlation matrix
print(df[['Age', 'Salary']].corr())
# Output:
#            Age    Salary
# Age     1.000000  0.866025
# Salary  0.866025  1.000000

# Correlation with specific method
print(df[['Age', 'Salary']].corr(method='pearson'))   # Pearson (default)
print(df[['Age', 'Salary']].corr(method='spearman'))  # Spearman
print(df[['Age', 'Salary']].corr(method='kendall'))   # Kendall

# Correlation with single column
print(df['Age'].corr(df['Salary']))

# Covariance matrix
print(df[['Age', 'Salary']].cov())
# Output:
#            Age        Salary
# Age     16.333333   28333.333333
# Salary  28333.333333  66666666.666667

# Covariance with single column
print(df['Age'].cov(df['Salary']))
```

**Interpretation:**
- **Correlation**: Ranges from -1 to 1
  - +1: Perfect positive correlation
  - 0: No linear correlation
  - -1: Perfect negative correlation
- **Covariance**: Measures directional relationship (not normalized)

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

Create pivot tables to summarize data (like Excel pivots):

```python
# Basic pivot table
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

# Multiple index levels
pivot = df.pivot_table(
    values='Salary',
    index=['City', 'Name'],
    aggfunc='mean'
)

# Fill missing values
pivot = df.pivot_table(
    values='Salary',
    index='City',
    aggfunc='mean',
    fill_value=0
)

# Margins (totals)
pivot = df.pivot_table(
    values='Salary',
    index='City',
    aggfunc='mean',
    margins=True  # Add row/column totals
)
```

### Crosstab

Analyze frequency distributions across categories:

```python
# Basic crosstab
crosstab = pd.crosstab(df['City'], df['Age'])
print(crosstab)

# With margins
crosstab = pd.crosstab(df['City'], df['Age'], margins=True)

# Normalize (percentages)
crosstab = pd.crosstab(df['City'], df['Age'], normalize=True)  # Overall percentages
crosstab = pd.crosstab(df['City'], df['Age'], normalize='index')  # Row percentages
crosstab = pd.crosstab(df['City'], df['Age'], normalize='columns')  # Column percentages

# Multiple columns
crosstab = pd.crosstab([df['City'], df['Name']], df['Age'])
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

### DateTime Conversion (pd.to_datetime)

Convert string or numeric date columns into datetime objects:

```python
# Convert string to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # With format

# Handle different formats
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Handle errors
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Invalid → NaN
df['Date'] = pd.to_datetime(df['Date'], errors='ignore')  # Invalid → unchanged

# Convert multiple columns
df[['Start', 'End']] = df[['Start', 'End']].apply(pd.to_datetime)
```

### DateTime Indexing & Resampling

```python
# Set datetime column as index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Resample data by different time periods
df_daily = df.resample('D').mean()      # Daily ('D')
df_weekly = df.resample('W').mean()     # Weekly ('W')
df_monthly = df.resample('M').mean()    # Monthly ('M')
df_yearly = df.resample('Y').mean()     # Yearly ('Y')
df_quarterly = df.resample('Q').mean()  # Quarterly ('Q')

# Different aggregation functions
df_monthly_sum = df.resample('M').sum()
df_monthly_count = df.resample('M').count()
df_monthly_max = df.resample('M').max()

# Multiple aggregations
df_monthly = df.resample('M').agg({
    'Sales': 'sum',
    'Revenue': 'mean'
})
```

### Date-Based Filtering & Slicing (loc)

```python
# Filter by specific date
df_specific = df.loc['2024-01-15']

# Filter by month
df_jan = df.loc['2024-01']              # January 2024
df_jan = df.loc['2024-01-01':'2024-01-31']  # Date range

# Filter by year
df_2024 = df.loc['2024']                # All of 2024
df_2024 = df.loc['2024-01-01':'2024-12-31']  # Explicit range

# Filter by date range
df_range = df.loc['2024-01-10':'2024-01-20']  # Custom range

# Filter by year and month
df_jan_2024 = df.loc['2024-01']

# Using datetime components
df_jan = df[df.index.month == 1]        # January (any year)
df_2024 = df[df.index.year == 2024]      # Year 2024
df_weekend = df[df.index.weekday >= 5]   # Weekend (Saturday=5, Sunday=6)
```

### Rolling & Expanding Windows

Calculate moving averages, cumulative sums, or trends over time:

```python
# Rolling window (moving average)
df['Rolling_Mean_7'] = df['Value'].rolling(window=7).mean()      # 7-day moving average
df['Rolling_Mean_30'] = df['Value'].rolling(window=30).mean()    # 30-day moving average
df['Rolling_Std'] = df['Value'].rolling(window=7).std()         # Rolling standard deviation
df['Rolling_Sum'] = df['Value'].rolling(window=7).sum()         # Rolling sum
df['Rolling_Max'] = df['Value'].rolling(window=7).max()         # Rolling maximum

# Rolling with minimum periods
df['Rolling_Mean'] = df['Value'].rolling(window=7, min_periods=3).mean()  # Need at least 3 values

# Expanding window (cumulative)
df['Expanding_Mean'] = df['Value'].expanding().mean()           # Cumulative mean
df['Expanding_Sum'] = df['Value'].expanding().sum()             # Cumulative sum
df['Expanding_Max'] = df['Value'].expanding().max()             # Running maximum
df['Expanding_Min'] = df['Value'].expanding().min()             # Running minimum

# Shift values (lag/lead)
df['Previous'] = df['Value'].shift(1)    # Previous value (lag 1)
df['Next'] = df['Value'].shift(-1)      # Next value (lead 1)
df['Lag_7'] = df['Value'].shift(7)      # 7 periods ago

# Percentage change
df['Pct_Change'] = df['Value'].pct_change()                     # Period-over-period change
df['Pct_Change_7'] = df['Value'].pct_change(periods=7)          # 7-period change
```

### Practical Use Cases

```python
# Example: Sales analysis
sales_df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'Sales': np.random.randint(1000, 5000, 100)
})
sales_df = sales_df.set_index('Date')

# Monthly sales report
monthly_sales = sales_df.resample('M').sum()
print("Monthly Sales:")
print(monthly_sales)

# 7-day moving average for trend
sales_df['MA_7'] = sales_df['Sales'].rolling(7).mean()

# Year-over-year comparison
sales_df['YoY'] = sales_df['Sales'].pct_change(periods=365)
```

---

## Advanced Tricks and Performance

### Applying Custom Functions (apply, map)

Use `apply()` for row- or column-wise operations and `map()` for element-wise transformations.

#### Apply Function

```python
# Apply function to each column
df.apply(np.mean)              # Mean of each column
df.apply(lambda x: x.max() - x.min())  # Range of each column

# Apply function to each row
df.apply(lambda row: row['Age'] + row['Salary'], axis=1)  # Row-wise

# Apply to specific column
df['Age'].apply(lambda x: x * 2)       # Double each age
df['Name'].apply(len)                  # Length of each name

# Apply custom function
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle'
    else:
        return 'Senior'

df['Age_Category'] = df['Age'].apply(categorize_age)

# Apply with multiple arguments
def calculate_bonus(salary, multiplier):
    return salary * multiplier

df['Bonus'] = df['Salary'].apply(lambda x: calculate_bonus(x, 0.1))
```

#### Map Function

Element-wise transformations on Series:

```python
# Map using dictionary
mapping = {'New York': 'NY', 'London': 'LON', 'Tokyo': 'TYO'}
df['City_Code'] = df['City'].map(mapping)

# Map using function
df['Age_Doubled'] = df['Age'].map(lambda x: x * 2)

# Map with default for missing values
df['City_Code'] = df['City'].map(mapping).fillna('Unknown')
```

**Difference:**
- **apply()**: More flexible, can work on entire DataFrame or Series
- **map()**: Faster for simple element-wise transformations on Series

### Lambda Functions

Quickly apply inline transformations:

```python
# Compute totals
df['Total'] = df.apply(lambda row: row['Age'] + row['Salary'], axis=1)

# Normalize values
df['Normalized_Age'] = df['Age'].apply(lambda x: (x - df['Age'].min()) / (df['Age'].max() - df['Age'].min()))

# Categorize grades
df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')

# Conditional updates
df['Status'] = df.apply(lambda row: 'High' if row['Salary'] > 60000 else 'Low', axis=1)

# Multiple conditions
df['Category'] = df.apply(
    lambda row: 'High Earner' if row['Salary'] > 60000 and row['Age'] > 30 else 'Other',
    axis=1
)
```

### Custom Logic Functions

Define user functions for complex operations:

```python
# Complex grading function
def calculate_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(calculate_grade)

# Conditional updates with multiple columns
def determine_status(row):
    if row['Age'] < 30 and row['Salary'] > 50000:
        return 'Young High Earner'
    elif row['Age'] >= 50:
        return 'Senior'
    else:
        return 'Standard'

df['Status'] = df.apply(determine_status, axis=1)

# Data cleaning function
def clean_name(name):
    name = str(name).strip()
    name = name.title()
    return name

df['Name'] = df['Name'].apply(clean_name)
```

### Practical Use Cases

**Automate repetitive transformations:**
```python
# Create derived columns
df['Age_Group'] = df['Age'].apply(lambda x: f"{(x//10)*10}s")
df['Salary_Per_Age'] = df.apply(lambda row: row['Salary'] / row['Age'], axis=1)

# Enhance datasets with dynamic calculations
df['Years_Until_Retirement'] = df['Age'].apply(lambda x: 65 - x if x < 65 else 0)
df['Tax_Bracket'] = df['Salary'].apply(
    lambda x: 'Low' if x < 50000 else 'Medium' if x < 100000 else 'High'
)
```

**Performance Tips:**
```python
# Vectorized operations are faster than apply
# SLOW:
df['New'] = df['Age'].apply(lambda x: x * 2)

# FAST:
df['New'] = df['Age'] * 2

# Use apply only when necessary (complex logic)
# For simple operations, use vectorized operations
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

