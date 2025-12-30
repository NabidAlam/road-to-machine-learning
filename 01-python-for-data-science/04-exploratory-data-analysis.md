# Exploratory Data Analysis (EDA) - Complete Guide

Comprehensive guide to performing systematic exploratory data analysis to understand your data before modeling.

## Table of Contents

- [Introduction](#introduction)
- [EDA Workflow](#eda-workflow)
- [Understanding Your Data](#understanding-your-data)
- [Univariate Analysis](#univariate-analysis)
- [Bivariate Analysis](#bivariate-analysis)
- [Multivariate Analysis](#multivariate-analysis)
- [Data Quality Checks](#data-quality-checks)
- [EDA Best Practices](#eda-best-practices)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is EDA?

**Exploratory Data Analysis (EDA)** is the process of analyzing datasets to summarize their main characteristics, often with visual methods. It helps you:
- Understand data structure and patterns
- Identify outliers and anomalies
- Discover relationships between variables
- Form hypotheses for modeling
- Validate assumptions

### Why EDA is Critical

- **Before Modeling**: Understand what you're working with
- **Feature Engineering**: Discover which features matter
- **Data Quality**: Identify issues early
- **Business Insights**: Find actionable patterns
- **Model Selection**: Guide algorithm choice

### EDA Philosophy

> "The purpose of computing is insight, not numbers." - Richard Hamming

EDA is about asking questions and finding answers in your data.

---

## EDA Workflow

### Systematic Approach

1. **Load and Inspect** - First look at your data
2. **Understand Structure** - Shape, types, missing values
3. **Univariate Analysis** - Individual variables
4. **Bivariate Analysis** - Relationships between pairs
5. **Multivariate Analysis** - Complex relationships
6. **Data Quality** - Check for issues
7. **Summary and Insights** - Document findings

---

## Understanding Your Data

### Initial Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# First look
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nRandom 5 rows:")
print(df.sample(5))
```

### Basic Information

```python
# Comprehensive info
df.info()

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# For categorical columns
print("\nCategorical Summary:")
print(df.describe(include=['object']))

# Memory usage
print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

### Missing Values Analysis

```python
# Missing values count
missing = df.isnull().sum()
missing_percent = 100 * missing / len(df)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
    'Missing Count', ascending=False
)
print("\nMissing Values:")
print(missing_df)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

# Missing values pattern
import missingno as msno
msno.matrix(df)
plt.show()
msno.bar(df)
plt.show()
```

### Duplicate Analysis

```python
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

if duplicates > 0:
    print("\nDuplicate rows:")
    print(df[df.duplicated(keep=False)])
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    print(f"\nAfter removing duplicates: {df_clean.shape}")
```

---

## Univariate Analysis

### Numerical Variables

```python
def analyze_numerical(df, column):
    """
    Comprehensive analysis of a numerical column
    """
    print(f"\n{'='*50}")
    print(f"Analysis of: {column}")
    print(f"{'='*50}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df[column].describe())
    
    # Additional statistics
    print(f"\nMean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")
    print(f"Mode: {df[column].mode().values}")
    print(f"Std Dev: {df[column].std():.2f}")
    print(f"Variance: {df[column].var():.2f}")
    print(f"Skewness: {df[column].skew():.2f}")
    print(f"Kurtosis: {df[column].kurtosis():.2f}")
    
    # Outliers (IQR method)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"\nOutliers (IQR method): {len(outliers)}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df[column].dropna(), bins=30, edgecolor='black')
    axes[0, 0].set_title(f'Histogram of {column}')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(df[column].dropna())
    axes[0, 1].set_title(f'Box Plot of {column}')
    axes[0, 1].set_ylabel(column)
    
    # KDE plot
    axes[1, 0].hist(df[column].dropna(), bins=30, density=True, alpha=0.7, label='Histogram')
    df[column].dropna().plot(kind='kde', ax=axes[1, 0], label='KDE')
    axes[1, 0].set_title(f'Distribution of {column}')
    axes[1, 0].set_xlabel(column)
    axes[1, 0].legend()
    
    # Q-Q plot (check normality)
    from scipy import stats
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q Plot of {column}')
    
    plt.tight_layout()
    plt.show()

# Example
analyze_numerical(df, 'age')
```

### Categorical Variables

```python
def analyze_categorical(df, column):
    """
    Comprehensive analysis of a categorical column
    """
    print(f"\n{'='*50}")
    print(f"Analysis of: {column}")
    print(f"{'='*50}")
    
    # Value counts
    value_counts = df[column].value_counts()
    print("\nValue Counts:")
    print(value_counts)
    
    # Percentages
    print("\nPercentages:")
    print(df[column].value_counts(normalize=True) * 100)
    
    # Unique values
    print(f"\nNumber of unique values: {df[column].nunique()}")
    print(f"Unique values: {df[column].unique()}")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    value_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Bar Plot of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
    axes[1].set_title(f'Pie Chart of {column}')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

# Example
analyze_categorical(df, 'category')
```

---

## Bivariate Analysis

### Numerical vs Numerical

```python
def analyze_numerical_pair(df, col1, col2):
    """
    Analyze relationship between two numerical variables
    """
    # Correlation
    correlation = df[col1].corr(df[col2])
    print(f"\nCorrelation between {col1} and {col2}: {correlation:.3f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot
    axes[0, 0].scatter(df[col1], df[col2], alpha=0.5)
    axes[0, 0].set_xlabel(col1)
    axes[0, 0].set_ylabel(col2)
    axes[0, 0].set_title(f'Scatter Plot: {col1} vs {col2}')
    
    # Add regression line
    z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df[col1], p(df[col1]), "r--", alpha=0.5)
    
    # Hexbin plot (for large datasets)
    axes[0, 1].hexbin(df[col1], df[col2], gridsize=20, cmap='Blues')
    axes[0, 1].set_xlabel(col1)
    axes[0, 1].set_ylabel(col2)
    axes[0, 1].set_title(f'Hexbin Plot: {col1} vs {col2}')
    
    # Joint plot (using seaborn)
    sns.jointplot(data=df, x=col1, y=col2, kind='scatter', ax=axes[1, 0])
    
    # 2D Histogram
    axes[1, 1].hist2d(df[col1], df[col2], bins=20, cmap='Blues')
    axes[1, 1].set_xlabel(col1)
    axes[1, 1].set_ylabel(col2)
    axes[1, 1].set_title(f'2D Histogram: {col1} vs {col2}')
    
    plt.tight_layout()
    plt.show()

# Example
analyze_numerical_pair(df, 'age', 'income')
```

### Numerical vs Categorical

```python
def analyze_numerical_categorical(df, num_col, cat_col):
    """
    Analyze relationship between numerical and categorical variables
    """
    # Group statistics
    print(f"\nStatistics of {num_col} by {cat_col}:")
    print(df.groupby(cat_col)[num_col].describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Box plot
    df.boxplot(column=num_col, by=cat_col, ax=axes[0, 0])
    axes[0, 0].set_title(f'{num_col} by {cat_col}')
    axes[0, 0].set_xlabel(cat_col)
    axes[0, 0].set_ylabel(num_col)
    
    # Violin plot
    sns.violinplot(data=df, x=cat_col, y=num_col, ax=axes[0, 1])
    axes[0, 1].set_title(f'Violin Plot: {num_col} by {cat_col}')
    
    # Histogram by category
    for category in df[cat_col].unique():
        axes[1, 0].hist(df[df[cat_col]==category][num_col], 
                       alpha=0.5, label=category, bins=20)
    axes[1, 0].set_xlabel(num_col)
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Distribution of {num_col} by {cat_col}')
    axes[1, 0].legend()
    
    # Bar plot of means
    means = df.groupby(cat_col)[num_col].mean()
    means.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title(f'Mean {num_col} by {cat_col}')
    axes[1, 1].set_xlabel(cat_col)
    axes[1, 1].set_ylabel(f'Mean {num_col}')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example
analyze_numerical_categorical(df, 'income', 'category')
```

### Categorical vs Categorical

```python
def analyze_categorical_pair(df, col1, col2):
    """
    Analyze relationship between two categorical variables
    """
    # Contingency table
    contingency = pd.crosstab(df[col1], df[col2])
    print(f"\nContingency Table: {col1} vs {col2}")
    print(contingency)
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test:")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Degrees of freedom: {dof}")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Stacked bar chart
    contingency.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title(f'Stacked Bar Chart: {col1} vs {col2}')
    axes[0].set_xlabel(col1)
    axes[0].set_ylabel('Count')
    axes[0].legend(title=col2)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Heatmap
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'Heatmap: {col1} vs {col2}')
    
    plt.tight_layout()
    plt.show()

# Example
analyze_categorical_pair(df, 'category', 'status')
```

---

## Multivariate Analysis

### Correlation Matrix

```python
# Calculate correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Find highly correlated pairs
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print("\nHighly Correlated Pairs (|r| > 0.7):")
for col1, col2, corr in high_corr:
    print(f"{col1} - {col2}: {corr:.3f}")
```

### Pair Plot

```python
# Pair plot for key numerical variables
key_vars = ['var1', 'var2', 'var3', 'var4']  # Select key variables
sns.pairplot(df[key_vars], diag_kind='kde')
plt.show()
```

### Principal Component Analysis (PCA) Visualization

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select numerical columns
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Visualization')
plt.show()
```

---

## Data Quality Checks

### Outlier Detection

```python
def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers

# Example
outliers_iqr = detect_outliers_iqr(df, 'age')
outliers_zscore = detect_outliers_zscore(df, 'age')
print(f"IQR outliers: {len(outliers_iqr)}")
print(f"Z-score outliers: {len(outliers_zscore)}")
```

### Data Consistency Checks

```python
# Check for logical inconsistencies
def check_consistency(df):
    """
    Check for logical inconsistencies in data
    """
    issues = []
    
    # Example: Age should be positive
    if 'age' in df.columns:
        negative_age = df[df['age'] < 0]
        if len(negative_age) > 0:
            issues.append(f"Found {len(negative_age)} rows with negative age")
    
    # Example: Start date should be before end date
    if 'start_date' in df.columns and 'end_date' in df.columns:
        invalid_dates = df[df['start_date'] > df['end_date']]
        if len(invalid_dates) > 0:
            issues.append(f"Found {len(invalid_dates)} rows with start_date > end_date")
    
    return issues

issues = check_consistency(df)
if issues:
    print("Data Consistency Issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No consistency issues found")
```

---

## EDA Best Practices

### 1. Start with Questions

Before analyzing, ask:
- What problem are we solving?
- What do we expect to find?
- What questions need answers?

### 2. Document Everything

```python
# Create EDA report
def create_eda_report(df, output_file='eda_report.txt'):
    """
    Generate comprehensive EDA report
    """
    with open(output_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write(f"   Shape: {df.shape}\n")
        f.write(f"   Columns: {df.columns.tolist()}\n\n")
        
        f.write("2. MISSING VALUES\n")
        missing = df.isnull().sum()
        for col, count in missing[missing > 0].items():
            f.write(f"   {col}: {count} ({100*count/len(df):.2f}%)\n")
        f.write("\n")
        
        f.write("3. DATA TYPES\n")
        for col, dtype in df.dtypes.items():
            f.write(f"   {col}: {dtype}\n")
        f.write("\n")
        
        f.write("4. SUMMARY STATISTICS\n")
        f.write(str(df.describe()))
        f.write("\n\n")
        
        # Add more sections as needed
        
    print(f"EDA report saved to {output_file}")

create_eda_report(df)
```

### 3. Use Visualizations

- Visuals reveal patterns numbers miss
- Use appropriate plot types
- Keep plots clear and labeled

### 4. Iterate

- EDA is iterative
- New questions arise from initial findings
- Deep dive into interesting patterns

### 5. Validate Assumptions

- Check normality for parametric tests
- Verify independence assumptions
- Test for multicollinearity

---

## Practice Exercises

### Exercise 1: Complete EDA

Perform complete EDA on a dataset:
1. Load and inspect
2. Check missing values
3. Analyze each variable
4. Find relationships
5. Detect outliers
6. Create summary report

### Exercise 2: Hypothesis Testing

Form hypotheses and test them:
- "Older customers spend more"
- "Category A has higher sales"
- "There's a correlation between X and Y"

### Exercise 3: Feature Engineering Ideas

Based on EDA, suggest:
- New features to create
- Features to remove
- Transformations needed

---

## Resources

### Libraries

- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Visualization
- **Missingno**: Missing data visualization
- **Scipy**: Statistical tests

### Books

- **"Exploratory Data Analysis"** by John Tukey
- **"Python for Data Analysis"** by Wes McKinney

### Tools

- **Jupyter Notebook**: Interactive EDA
- **Pandas Profiling**: Automated EDA reports

```python
# Install: pip install pandas-profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="EDA Report")
profile.to_file("eda_report.html")
```

---

## Key Takeaways

1. **EDA is Essential**: Never skip this step
2. **Ask Questions**: Start with hypotheses
3. **Visualize**: Use plots to understand data
4. **Document**: Keep notes of findings
5. **Iterate**: EDA is an ongoing process
6. **Validate**: Check assumptions and data quality

---

**Remember**: Good EDA leads to better models. Spend time understanding your data before modeling!

