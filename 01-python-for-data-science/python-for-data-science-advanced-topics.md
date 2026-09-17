# Advanced Python for Data Science Topics

Advanced techniques for efficient data manipulation, analysis, and visualization.

## Table of Contents

- [Advanced Pandas Techniques](#advanced-pandas-techniques)
- [Performance Optimization](#performance-optimization)
- [Advanced Visualization Techniques](#advanced-visualization-techniques)
- [Memory Optimization](#memory-optimization)
- [Advanced EDA Techniques](#advanced-eda-techniques)
- [Data Pipeline Design Patterns](#data-pipeline-design-patterns)
- [Integration of Multiple Tools](#integration-of-multiple-tools)

---

## Advanced Pandas Techniques

### Multi-Index DataFrames

```python
import pandas as pd
import numpy as np

# Create multi-index DataFrame
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'number'))
df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)

# Access multi-index
df.loc['A']
df.loc[('A', 1)]
df.xs('A', level='letter')

# Stack and unstack
stacked = df.stack()
unstacked = df.unstack()

# Pivot with multi-index
df_pivot = df.pivot_table(values='value', index='letter', columns='number')
```

### Advanced Grouping

```python
# Start from a known DataFrame for grouping demos
df = pd.DataFrame({
    'col1': ['a', 'a', 'b', 'b'],
    'col2': [1, 2, 1, 2],
    'col3': [10.0, 20.0, 30.0, 40.0],
    'col4': [1, 1, 1, 1],
    'category': ['x', 'x', 'y', 'y'],
    'value': [5.0, 7.0, 9.0, 11.0],
})

# Group by multiple columns
df.groupby(['col1', 'col2']).agg({
    'col3': ['mean', 'std'],
    'col4': 'sum'
})

# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

df.groupby('category').agg({
    'value': ['mean', range_func, lambda x: x.quantile(0.75)]
})

# Transform (same shape as original)
df['group_mean'] = df.groupby('category')['value'].transform('mean')

# Apply custom function to groups
def normalize_group(group):
    return (group - group.mean()) / group.std()

df['normalized'] = df.groupby('category')['value'].transform(normalize_group)
```

### Advanced Merging and Joining

```python
# Sample frames for merge demos
df1 = pd.DataFrame({
    'key': [1, 2, 3],
    'key1': ['a', 'b', 'c'],
    'key2': [10, 20, 30],
    'left_val': [100, 200, 300],
})
df2 = pd.DataFrame({
    'key': [1, 2, 4],
    'key1': ['a', 'b', 'd'],
    'key2': [10, 20, 40],
    'right_val': [9, 8, 7],
})

# Merge with multiple keys
df1.merge(df2, on=['key1', 'key2'], how='inner')

# Merge with different column names
df_left = pd.DataFrame({'employee_id': [1, 2], 'name': ['Ada', 'Bob']})
df_right = pd.DataFrame({'emp_id': [1, 2], 'dept': ['ML', 'Data']})
df_left.merge(df_right, left_on='employee_id', right_on='emp_id')

# Merge with indicator
df1.merge(df2, on='key', how='outer', indicator=True)

# Merge on index
df1.merge(df2, left_index=True, right_index=True)

# Concatenate with keys
pd.concat([df1, df2], keys=['A', 'B'], names=['source', 'index'])
```

### Window Functions

```python
# Rolling window demos need a datetime index and a value series
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'value': np.arange(30, dtype=float),
})

df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['rolling_std'] = df['value'].rolling(window=7).std()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()

# Custom window functions
def custom_window(x):
    return x.iloc[-1] - x.iloc[0]

df['window_diff'] = df['value'].rolling(window=5).apply(custom_window)

# Time-based rolling
df = df.set_index('date')
df['30d_mean'] = df['value'].rolling('30D').mean()
```

---

## Performance Optimization

### Vectorization

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'col1': np.array([1.0, -1.0, 2.0, 0.0]),
    'col2': np.array([3.0, 4.0, 5.0, 6.0]),
})

# Slow: Apply with Python function
def slow_function(row):
    return row['col1'] * 2 + row['col2']

df['result'] = df.apply(slow_function, axis=1)

# Fast: Vectorized operations
df['result'] = df['col1'] * 2 + df['col2']

# Fast: NumPy operations
df['result'] = np.where(df['col1'] > 0, df['col1'] * 2, 0)

# Fast: Pandas built-in methods
df['result'] = df['col1'].mul(2).add(df['col2'])
```

### Efficient Data Types

```python
df = pd.DataFrame({
    'int_col': np.arange(1000, dtype='int64'),
    'float_col': np.random.rand(1000).astype('float64'),
    'category_col': np.random.choice(['a', 'b', 'c'], size=1000),
    'date_col': pd.date_range('2024-01-01', periods=1000, freq='h'),
})

# Check memory usage
df.info(memory_usage='deep')

# Convert to efficient dtypes
df['int_col'] = df['int_col'].astype('int32')  # Instead of int64
df['float_col'] = df['float_col'].astype('float32')  # Instead of float64

# Convert to category for strings
df['category_col'] = df['category_col'].astype('category')

# Convert to datetime
df['date_col'] = pd.to_datetime(df['date_col'])

# Check memory savings
print(f"Before optimize helper: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def optimize_dtypes(frame):
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast='integer')
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast='float')
        elif out[col].dtype == object and out[col].nunique() < 50:
            out[col] = out[col].astype('category')
    return out

df = optimize_dtypes(df)
print(f"After: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Parallel Processing

```python
import numpy as np
import pandas as pd

def process_chunk(chunk):
    # Process chunk
    return chunk.groupby('category').sum(numeric_only=True)

df = pd.DataFrame({
    'category': np.random.choice(['a', 'b', 'c'], size=100),
    'value': np.random.randn(100),
})

# Split into chunks (same pattern as multiprocessing.Pool.map)
chunks = np.array_split(df, 4)
results = list(map(process_chunk, chunks))

# Combine results
final_result = pd.concat(results)
```

### Using Cython/Numba

```python snippet-skip
# Numba for numerical computations
from numba import jit
import numpy as np

@jit(nopython=True)
def fast_computation(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2 + 1
    return result

# Use with pandas
df['result'] = fast_computation(df['value'].values)
```

---

## Advanced Visualization Techniques

### Interactive Visualizations with Plotly

```python snippet-skip
import plotly.graph_objects as go
import plotly.express as px

# Advanced Plotly figure
fig = go.Figure()

# Add multiple traces
fig.add_trace(go.Scatter(x=df['x'], y=df['y1'], name='Series 1'))
fig.add_trace(go.Scatter(x=df['x'], y=df['y2'], name='Series 2'))

# Customize layout
fig.update_layout(
    title='Advanced Plot',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    hovermode='x unified',
    template='plotly_dark'
)

# Add annotations
fig.add_annotation(
    x=5, y=10,
    text="Important Point",
    showarrow=True,
    arrowhead=2
)

fig.show()

# Subplots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4')
)

fig.add_trace(go.Scatter(x=df['x'], y=df['y1']), row=1, col=1)
fig.add_trace(go.Bar(x=df['x'], y=df['y2']), row=1, col=2)
fig.show()
```

### Advanced Matplotlib Customization

```python snippet-skip
import matplotlib.pyplot as plt
import seaborn as sns

# Create figure with custom layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Advanced Visualization', fontsize=16)

# Custom styling
sns.set_style("whitegrid")
sns.set_palette("husl")

# Advanced plot types
axes[0, 0].hexbin(df['x'], df['y'], gridsize=20, cmap='Blues')
axes[0, 1].contour(X, Y, Z, levels=10)
axes[1, 0].violinplot([df['group1'], df['group2']])
axes[1, 1].boxplot([df['group1'], df['group2']], labels=['Group 1', 'Group 2'])

# Add annotations
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

plt.tight_layout()
plt.show()
```

### Custom Color Maps

```python snippet-skip
import matplotlib.colors as mcolors

# Create custom colormap
colors = ['#FF0000', '#00FF00', '#0000FF']
n_bins = 100
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use in plot
plt.scatter(df['x'], df['y'], c=df['value'], cmap=cmap)
plt.colorbar()
plt.show()
```

---

## Memory Optimization

### Chunk Processing

```python snippet-skip
# Process large files in chunks
chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = chunk.groupby('category').sum()
    chunks.append(processed)

# Combine results
final_df = pd.concat(chunks, ignore_index=True)
```

### Sparse DataFrames

```python snippet-skip
# For data with many zeros
from scipy import sparse

# Create sparse matrix
sparse_matrix = sparse.csr_matrix(df.values)

# Convert back to DataFrame
df_sparse = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=df.columns)
```

### Memory-Efficient Data Types

```python
def optimize_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Convert to category if low cardinality
            if df[col].nunique() < 50:
                df[col] = df[col].astype('category')
    
    return df
```

---

## Advanced EDA Techniques

### Automated EDA

```python snippet-skip
import pandas_profiling as pp

# Generate comprehensive EDA report
profile = pp.ProfileReport(df)
profile.to_file("eda_report.html")

# Using ydata_profiling (newer)
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="EDA Report")
profile.to_file("eda_report.html")
```

### Statistical Tests

```python snippet-skip
from scipy import stats

# Normality test
stat, p_value = stats.normaltest(df['value'])
print(f"Normality test: p-value = {p_value}")

# Correlation test
corr, p_value = stats.pearsonr(df['x'], df['y'])
print(f"Correlation: {corr:.3f}, p-value: {p_value:.3f}")

# T-test
stat, p_value = stats.ttest_ind(df['group1'], df['group2'])
print(f"T-test: p-value = {p_value:.3f}")

# ANOVA
f_stat, p_value = stats.f_oneway(df['group1'], df['group2'], df['group3'])
print(f"ANOVA: p-value = {p_value:.3f}")
```

### Outlier Detection

```python snippet-skip
# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
outliers = df[z_scores > 3]

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1)
outlier_labels = iso_forest.fit_predict(df[['value']])
outliers = df[outlier_labels == -1]
```

---

## Data Pipeline Design Patterns

### ETL Pipeline

```python snippet-skip
class ETLPipeline:
    def __init__(self):
        self.data = None
    
    def extract(self, source):
        """Extract data from source"""
        if source.endswith('.csv'):
            self.data = pd.read_csv(source)
        elif source.endswith('.json'):
            self.data = pd.read_json(source)
        return self
    
    def transform(self):
        """Transform data"""
        # Clean data
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        
        # Feature engineering
        self.data['new_feature'] = self.data['col1'] * self.data['col2']
        
        return self
    
    def load(self, destination):
        """Load data to destination"""
        self.data.to_csv(destination, index=False)
        return self

# Usage
pipeline = ETLPipeline()
pipeline.extract('source.csv').transform().load('destination.csv')
```

### Data Validation Pipeline

```python
def validate_data(df, schema):
    """Validate DataFrame against schema"""
    errors = []
    
    # Check columns
    required_cols = schema.get('required_columns', [])
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check data types
    for col, dtype in schema.get('dtypes', {}).items():
        if col in df.columns and df[col].dtype != dtype:
            errors.append(f"Column {col} has wrong dtype: {df[col].dtype} != {dtype}")
    
    # Check value ranges
    for col, range_vals in schema.get('ranges', {}).items():
        if col in df.columns:
            min_val, max_val = range_vals
            if (df[col] < min_val).any() or (df[col] > max_val).any():
                errors.append(f"Column {col} has values outside range [{min_val}, {max_val}]")
    
    return errors

# Usage
schema = {
    'required_columns': ['col1', 'col2'],
    'dtypes': {'col1': 'int64', 'col2': 'float64'},
    'ranges': {'col1': (0, 100), 'col2': (0.0, 1.0)}
}

errors = validate_data(df, schema)
if errors:
    print("Validation errors:", errors)
```

---

## Integration of Multiple Tools

### NumPy + Pandas + Visualization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create data with NumPy
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Convert to Pandas DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Advanced operations
df['y_smooth'] = df['y'].rolling(window=5).mean()
df['y_diff'] = np.diff(df['y'], prepend=df['y'].iloc[0])

# Visualize with Matplotlib/Seaborn
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['x'], df['y'], alpha=0.5, label='Original')
axes[0].plot(df['x'], df['y_smooth'], label='Smoothed')
axes[0].legend()
axes[0].set_title('Time Series')

sns.histplot(df['y'], ax=axes[1])
axes[1].set_title('Distribution')

plt.tight_layout()
plt.show()
```

### Pandas + Plotly + Streamlit

```python snippet-skip
import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('data.csv')

# Streamlit app
st.title('Data Analysis Dashboard')

# Sidebar filters
category = st.sidebar.selectbox('Category', df['category'].unique())
filtered_df = df[df['category'] == category]

# Main content
st.dataframe(filtered_df)

# Interactive Plotly chart
fig = px.scatter(filtered_df, x='x', y='y', color='value', size='size')
st.plotly_chart(fig, use_container_width=True)

# Statistics
st.metric('Mean Value', filtered_df['value'].mean())
st.metric('Std Dev', filtered_df['value'].std())
```

---

## Key Takeaways

1. **Advanced Pandas**: Master multi-index, advanced grouping, and window functions
2. **Performance**: Vectorize operations, use efficient dtypes, parallel processing
3. **Visualization**: Create interactive and publication-quality plots
4. **Memory**: Optimize data types, use chunking, sparse matrices
5. **Pipelines**: Design robust ETL and validation pipelines
6. **Integration**: Combine NumPy, Pandas, and visualization tools effectively

---

**Try next:** Apply one technique from this page to a dataset you already cleaned once.

