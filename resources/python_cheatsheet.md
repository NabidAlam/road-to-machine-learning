# Python Data Science Cheatsheet

Quick reference for common Python syntax and operations used in everyday data science work.

## Table of Contents

- [NumPy](#numpy)
- [Pandas](#pandas)
- [Matplotlib & Seaborn](#matplotlib--seaborn)
- [Scikit-learn](#scikit-learn)
- [File Operations](#file-operations)
- [List & Dictionary Operations](#list--dictionary-operations)
- [String Operations](#string-operations)
- [Date & Time](#date--time)
- [Useful Functions](#useful-functions)

---

## NumPy

### Creating Arrays

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
empty = np.empty((2, 2))
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
random = np.random.rand(3, 3)  # Random 0-1
randint = np.random.randint(0, 10, (3, 3))  # Random integers
```

### Array Operations

```python
# Shape and dimensions
arr.shape
arr.ndim
arr.size
arr.dtype

# Reshaping
arr.reshape(2, 3)
arr.flatten()
arr.ravel()

# Indexing and slicing
arr[0]           # First element
arr[-1]          # Last element
arr[1:4]         # Slice
arr[arr > 5]     # Boolean indexing
arr2d[0, 1]      # 2D indexing
arr2d[:, 1]      # All rows, column 1
arr2d[1, :]      # Row 1, all columns

# Mathematical operations
np.sum(arr)
np.mean(arr)
np.std(arr)
np.var(arr)
np.min(arr)
np.max(arr)
np.median(arr)
np.percentile(arr, 50)

# Element-wise operations
arr + 1
arr * 2
arr ** 2
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.sin(arr)

# Array operations
arr1 + arr2      # Element-wise addition
arr1 * arr2      # Element-wise multiplication
np.dot(arr1, arr2)  # Matrix multiplication
arr1 @ arr2      # Matrix multiplication (Python 3.5+)
```

### Useful NumPy Functions

```python
np.concatenate([arr1, arr2])
np.vstack([arr1, arr2])  # Vertical stack
np.hstack([arr1, arr2])  # Horizontal stack
np.where(condition, x, y)  # Conditional
np.unique(arr)  # Unique values
np.sort(arr)  # Sort
np.argsort(arr)  # Sort indices
```

---

## Pandas

### Creating DataFrames

```python
import pandas as pd

# From dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# From CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', index_col=0)
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# From Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# From JSON
df = pd.read_json('file.json')

# From dictionary of lists
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
```

### Viewing Data

```python
df.head()        # First 5 rows
df.head(10)      # First 10 rows
df.tail()        # Last 5 rows
df.shape         # (rows, columns)
df.info()        # Data types and memory
df.describe()    # Statistical summary
df.columns       # Column names
df.index         # Row indices
df.dtypes        # Data types
df.value_counts()  # Value counts for Series
```

### Selecting Data

```python
# Single column (returns Series)
df['column_name']
df.column_name

# Multiple columns (returns DataFrame)
df[['col1', 'col2']]

# Rows by index
df.iloc[0]           # First row
df.iloc[0:5]        # First 5 rows
df.iloc[0, 1]       # Row 0, column 1
df.iloc[0:5, 1:3]   # Rows 0-4, columns 1-2

# Rows by label
df.loc['index_name']
df.loc['row1':'row5', 'col1':'col3']

# Boolean indexing
df[df['column'] > 5]
df[(df['col1'] > 5) & (df['col2'] < 10)]
df[df['column'].isin([1, 2, 3])]
df[df['column'].str.contains('text')]
```

### Data Manipulation

```python
# Adding/Removing columns
df['new_col'] = values
df.drop('column', axis=1)  # Drop column
df.drop([0, 1], axis=0)     # Drop rows

# Renaming
df.rename(columns={'old': 'new'})
df.rename(index={0: 'first'})

# Sorting
df.sort_values('column')
df.sort_values('column', ascending=False)
df.sort_values(['col1', 'col2'])

# Grouping
df.groupby('column').mean()
df.groupby('column').sum()
df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})

# Merging
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')
pd.concat([df1, df2])  # Concatenate
df1.join(df2)  # Join on index
```

### Handling Missing Data

```python
df.isna()        # Check for NaN
df.isnull()      # Same as isna()
df.notna()       # Check for non-NaN
df.dropna()      # Drop rows with NaN
df.dropna(axis=1)  # Drop columns with NaN
df.fillna(0)     # Fill NaN with 0
df.fillna(df.mean())  # Fill with mean
df['col'].fillna(df['col'].median())
```

### Data Types

```python
df['col'].astype(int)
df['col'].astype(float)
df['col'].astype(str)
df['col'].astype('category')
pd.to_datetime(df['date_col'])
```

### String Operations

```python
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.contains('text')
df['col'].str.split(' ')
df['col'].str.len()
```

### Aggregations

```python
df.sum()
df.mean()
df.median()
df.std()
df.min()
df.max()
df.count()
df.nunique()  # Number of unique values
```

### Saving Data

```python
df.to_csv('output.csv')
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx')
df.to_json('output.json')
df.to_pickle('output.pkl')
```

---

## Matplotlib & Seaborn

### Basic Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot(x, y)
plt.plot(x, y, label='Line', color='blue', linestyle='--', linewidth=2)

# Scatter plot
plt.scatter(x, y, s=50, alpha=0.5, c='red')

# Bar plot
plt.bar(x, y)
plt.barh(x, y)  # Horizontal

# Histogram
plt.hist(data, bins=30)

# Box plot
plt.boxplot(data)

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)

# Styling
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Seaborn

```python
# Distribution plots
sns.distplot(data)
sns.histplot(data, kde=True)

# Scatter plot
sns.scatterplot(x='col1', y='col2', data=df, hue='col3')

# Line plot
sns.lineplot(x='col1', y='col2', data=df)

# Bar plot
sns.barplot(x='col1', y='col2', data=df)

# Box plot
sns.boxplot(x='col1', y='col2', data=df)

# Violin plot
sns.violinplot(x='col1', y='col2', data=df)

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Pair plot
sns.pairplot(df, hue='target')

# Count plot
sns.countplot(x='column', data=df)

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')
```

---

## Scikit-learn

### Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Min-Max scaling
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### Models

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification
model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100)
model = DecisionTreeClassifier()
model = SVC()
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Classification
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities[:, 1])

# Regression
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

---

## File Operations

### Reading Files

```python
# Text file
with open('file.txt', 'r') as f:
    content = f.read()
    lines = f.readlines()

# CSV
import csv
with open('file.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# JSON
import json
with open('file.json', 'r') as f:
    data = json.load(f)
```

### Writing Files

```python
# Text file
with open('file.txt', 'w') as f:
    f.write('content')

# CSV
with open('file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['col1', 'col2'])
    writer.writerow([1, 2])

# JSON
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)
```

---

## List & Dictionary Operations

### Lists

```python
# Creating
lst = [1, 2, 3]
lst = list(range(10))

# Adding
lst.append(4)
lst.extend([5, 6])
lst.insert(0, 0)

# Removing
lst.remove(3)
lst.pop()  # Remove last
lst.pop(0)  # Remove by index
del lst[0]

# Slicing
lst[0:3]      # First 3
lst[-3:]      # Last 3
lst[::2]      # Every 2nd element
lst[::-1]     # Reverse

# List comprehension
[x**2 for x in range(10)]
[x for x in lst if x > 5]
[x**2 if x > 5 else x for x in lst]
```

### Dictionaries

```python
# Creating
d = {'key1': 'value1', 'key2': 'value2'}
d = dict(key1='value1', key2='value2')

# Accessing
d['key1']
d.get('key1', 'default')  # With default

# Adding/Updating
d['key3'] = 'value3'
d.update({'key4': 'value4'})

# Removing
del d['key1']
d.pop('key2')
d.popitem()  # Remove last item

# Iterating
for key, value in d.items():
    print(key, value)

for key in d.keys():
    print(key)

for value in d.values():
    print(value)

# Dictionary comprehension
{key: value**2 for key, value in d.items()}
```

---

## String Operations

```python
# Basic operations
s = "Hello World"
s.lower()
s.upper()
s.strip()
s.split(' ')
s.replace('old', 'new')
s.startswith('H')
s.endswith('d')
s.find('World')
s.count('l')

# Formatting
f"Value: {value}"
"Value: {}".format(value)
"Value: {:.2f}".format(3.14159)

# Checking
s.isdigit()
s.isalpha()
s.isalnum()
```

---

## Date & Time

```python
from datetime import datetime, timedelta

# Current time
now = datetime.now()
today = datetime.today()

# Creating dates
date = datetime(2024, 1, 1)
date = datetime.strptime('2024-01-01', '%Y-%m-%d')

# Formatting
date.strftime('%Y-%m-%d')
date.strftime('%B %d, %Y')

# Operations
date + timedelta(days=7)
date - timedelta(days=7)
(date2 - date1).days
```

### Pandas Date Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# Filtering
df[df['date'] > '2024-01-01']
df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]
```

---

## Useful Functions

### Lambda Functions

```python
# Simple function
square = lambda x: x**2

# With map
squared = list(map(lambda x: x**2, [1, 2, 3]))

# With filter
evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))

# With pandas
df['new_col'] = df['col'].apply(lambda x: x**2)
```

### Zip and Enumerate

```python
# Zip
for x, y in zip(list1, list2):
    print(x, y)

# Enumerate
for i, value in enumerate(list1):
    print(i, value)
```

### Any and All

```python
any([True, False, True])  # True
all([True, True, True])   # True
```

### Sorting

```python
sorted(lst)
sorted(lst, reverse=True)
sorted(lst, key=lambda x: x[1])  # Sort by second element

# In-place
lst.sort()
lst.sort(reverse=True)
```

### Set Operations

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

set1.union(set2)      # {1, 2, 3, 4, 5}
set1.intersection(set2)  # {3}
set1.difference(set2)  # {1, 2}
```

---

## Quick Tips

1. **Always use virtual environments**: `python -m venv env`
2. **Import conventions**:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```
3. **Set random seeds**: `np.random.seed(42)`, `random_state=42`
4. **Use `.copy()`** when modifying DataFrames to avoid SettingWithCopyWarning
5. **Check data types**: Always verify with `df.dtypes` and `df.info()`
6. **Handle missing data early**: Check with `df.isna().sum()`
7. **Save your work**: Use version control and save intermediate results

---

**Print this cheatsheet and keep it handy!** Practice these commands daily to commit them to memory.

