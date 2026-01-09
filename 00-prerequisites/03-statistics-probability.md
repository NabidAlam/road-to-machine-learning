# Statistics and Probability for Machine Learning

Comprehensive guide to statistical concepts and probability theory essential for understanding and evaluating machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Visualizing Data](#visualizing-data)
- [Descriptive Statistics](#descriptive-statistics)
- [Data Distribution](#data-distribution)
- [Probability Fundamentals](#probability-fundamentals)
- [Probability Distributions](#probability-distributions)
- [Inferential Statistics](#inferential-statistics)
- [Hypothesis Testing](#hypothesis-testing)
- [Applications in ML](#applications-in-ml)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Statistics and Probability Matter

Statistics and probability are the **foundation of machine learning**:

- **Model Evaluation**: Metrics, confidence intervals, significance tests
- **Uncertainty Quantification**: Understanding model predictions
- **Feature Analysis**: Understanding data distributions
- **Model Assumptions**: Many algorithms assume specific distributions
- **A/B Testing**: Comparing model performance

### What You'll Learn

- Descriptive statistics (mean, variance, correlation)
- Probability theory (conditional probability, Bayes' theorem)
- Common distributions (normal, binomial, Poisson)
- Hypothesis testing (t-tests, chi-square)
- Confidence intervals
- How these apply to ML

---

## Visualizing Data

Understanding the importance of visualizing data for analysis and communication. Visualizations help identify patterns, outliers, and relationships that numbers alone cannot reveal.

### One-Way Table

A simple table displaying frequency counts of a single variable.

```python
import pandas as pd

# Example: Category frequencies
data = {'Category': ['A', 'B', 'C', 'D', 'A', 'B', 'A'],
        'Frequency': [3, 2, 1, 1, 3, 2, 3]}

df = pd.DataFrame(data)
frequency_table = df['Category'].value_counts().reset_index()
frequency_table.columns = ['Category', 'Frequency']
print(frequency_table)
# Output:
#   Category  Frequency
# 0        A          3
# 1        B          2
# 2        C          1
# 3        D          1
```

### Two-Way Table (Contingency Table)

A table summarizing the relationship between two categorical variables.

```python
# Example: Relationship between two variables
data = {
    'Variable_A': ['Category1', 'Category1', 'Category2', 'Category2', 'Category1'],
    'Variable_B': ['X', 'Y', 'X', 'Y', 'X']
}

df = pd.DataFrame(data)
two_way_table = pd.crosstab(df['Variable_A'], df['Variable_B'])
print(two_way_table)
# Output:
# Variable_B    X  Y
# Variable_A        
# Category1     2  1
# Category2     1  1
```

### Frequency Table

Shows how often each value in a dataset occurs.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
frequency_table = pd.Series(data).value_counts().sort_index().reset_index()
frequency_table.columns = ['Value', 'Frequency']
print(frequency_table)
# Output:
#    Value  Frequency
# 0      1          1
# 1      2          2
# 2      3          3
# 3      4          4
# 4      5          2
```

### Relative Frequency Table

Displays proportions or percentages instead of raw counts.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
freq = pd.Series(data).value_counts().sort_index()
relative_freq = (freq / len(data) * 100).round(2)
relative_table = pd.DataFrame({
    'Value': relative_freq.index,
    'Relative Frequency (%)': relative_freq.values
})
print(relative_table)
```

### Bar Graphs

Represent categorical data with rectangular bars. Height/length represents frequency.

```python
import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
frequencies = [23, 45, 56, 78]

plt.figure(figsize=(8, 6))
plt.bar(categories, frequencies, color='skyblue', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Bar Graph - Categorical Data')
plt.show()
```

**Use Cases:**
- Comparing categories
- Showing frequency distributions
- Visualizing survey results

### Pie Charts

Show proportions of a whole using slices of a circle.

```python
categories = ['A', 'B', 'C', 'D']
sizes = [30, 25, 20, 25]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart - Proportions')
plt.axis('equal')
plt.show()
```

**Use Cases:**
- Showing composition/parts of a whole
- Percentage distributions
- Market share visualization

### Line Graphs

Display trends over time or continuous data points.

```python
time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
values = [10, 15, 13, 17, 20, 18, 22, 25, 23, 28]

plt.figure(figsize=(10, 6))
plt.plot(time, values, marker='o', linewidth=2, markersize=8)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Line Graph - Trends Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

**Use Cases:**
- Time series data
- Trends over time
- Continuous relationships

### Dot Plots

A simple graph using dots to represent data points along a number line.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

plt.figure(figsize=(10, 2))
plt.plot(data, [1]*len(data), 'o', markersize=10)
plt.xlabel('Value')
plt.yticks([])
plt.title('Dot Plot')
plt.grid(True, axis='x', alpha=0.3)
plt.show()
```

**Use Cases:**
- Small datasets
- Showing individual data points
- Identifying clusters and gaps

### Histogram

A graphical representation using bars to show the frequency of numerical data ranges.

```python
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram - Frequency Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

**Key Features:**
- Bins: Ranges that group data
- Frequency: Height of bars shows count
- Shape: Reveals distribution pattern

**Use Cases:**
- Understanding data distribution
- Identifying skewness
- Detecting outliers

### Box Plot

A graphical representation using quartiles, highlighting median, IQR, and potential outliers.

```python
data = np.random.normal(100, 15, 100)

plt.figure(figsize=(8, 6))
plt.boxplot(data, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
plt.ylabel('Value')
plt.title('Box Plot - Five-Number Summary')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

**Components:**
- **Lower Whisker**: Q1 - 1.5 × IQR (or minimum if no outliers)
- **Box (Q1 to Q3)**: Interquartile Range
- **Median Line**: Q2 (middle value)
- **Upper Whisker**: Q3 + 1.5 × IQR (or maximum if no outliers)
- **Outliers**: Points beyond whiskers (marked individually)

**Use Cases:**
- Comparing distributions
- Identifying outliers
- Showing spread and skewness

### Joint Distribution

A table or graph showing the distribution of two variables simultaneously.

```python
# Joint distribution table
x = np.random.choice(['Low', 'Medium', 'High'], 100)
y = np.random.choice(['A', 'B', 'C'], 100)
joint_dist = pd.crosstab(x, y, normalize='all') * 100
print("Joint Distribution (%):")
print(joint_dist.round(2))

# Joint distribution visualization (heatmap)
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(joint_dist, annot=True, fmt='.2f', cmap='Blues')
plt.title('Joint Distribution Heatmap')
plt.show()
```

**Use Cases:**
- Understanding relationships between two variables
- Identifying patterns in categorical data
- Conditional probability analysis

---

## Descriptive Statistics

### Measures of Central Tendency

#### Mean (Average)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = np.array([10, 20, 30, 40, 50])

# Mean
mean = np.mean(data)
print(f"Mean: {mean}")  # 30.0

# Formula: μ = (1/n) * Σx_i
manual_mean = np.sum(data) / len(data)
print(f"Manual calculation: {manual_mean}")  # 30.0

# Weighted mean
weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
weighted_mean = np.average(data, weights=weights)
print(f"Weighted mean: {weighted_mean}")
```

**Properties:**
- Sensitive to outliers
- Sum of deviations from mean is zero
- Used in: Loss functions (MSE), normalization

#### Median

```python
# Median (middle value)
median = np.median(data)
print(f"Median: {median}")  # 30.0

# For even number of values
data_even = np.array([10, 20, 30, 40])
median_even = np.median(data_even)
print(f"Median (even): {median_even}")  # 25.0 (average of 20 and 30)

# Robust to outliers
data_with_outlier = np.array([10, 20, 30, 40, 50, 1000])
print(f"Mean with outlier: {np.mean(data_with_outlier)}")  # 191.67
print(f"Median with outlier: {np.median(data_with_outlier)}")  # 35.0
```

**Properties:**
- Robust to outliers
- 50th percentile
- Used in: Robust statistics, median-based metrics

#### Mode

```python
# Mode (most frequent value)
data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
mode_result = stats.mode(data)
print(f"Mode: {mode_result.mode[0]}")  # 4
print(f"Count: {mode_result.count[0]}")  # 4

# For continuous data, use histogram
continuous_data = np.random.normal(0, 1, 1000)
hist, bins = np.histogram(continuous_data, bins=50)
mode_bin = bins[np.argmax(hist)]
print(f"Mode (approximate): {mode_bin}")
```

### Measures of Dispersion

#### Variance and Standard Deviation

```python
data = np.array([10, 20, 30, 40, 50])

# Variance
variance = np.var(data)
print(f"Variance: {variance}")  # 200.0

# Formula: σ² = (1/n) * Σ(x_i - μ)²
mean = np.mean(data)
manual_variance = np.mean((data - mean) ** 2)
print(f"Manual variance: {manual_variance}")  # 200.0

# Sample variance (Bessel's correction)
sample_variance = np.var(data, ddof=1)  # Divide by (n-1) instead of n
print(f"Sample variance: {sample_variance}")  # 250.0

# Standard deviation
std_dev = np.std(data)
print(f"Standard deviation: {std_dev}")  # 14.142...

# Standard deviation = √variance
print(f"√variance = {np.sqrt(variance)}")  # 14.142...
```

**Why it matters:**
- **Feature scaling**: Normalize using mean and std
- **Outlier detection**: Values beyond 3 standard deviations
- **Model assumptions**: Many algorithms assume unit variance

#### Range and Quartiles

```python
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Range
data_range = np.max(data) - np.min(data)
print(f"Range: {data_range}")  # 90

# Quartiles
q1 = np.percentile(data, 25)  # First quartile
q2 = np.percentile(data, 50)  # Median (second quartile)
q3 = np.percentile(data, 75)  # Third quartile

print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")

# Interquartile Range (IQR)
iqr = q3 - q1
print(f"IQR: {iqr}")

# Outlier detection using IQR
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")
```

### Correlation and Covariance

#### Covariance

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Covariance
covariance = np.cov(x, y)[0, 1]
print(f"Covariance: {covariance}")  # 5.0

# Manual calculation
mean_x = np.mean(x)
mean_y = np.mean(y)
manual_cov = np.mean((x - mean_x) * (y - mean_y))
print(f"Manual covariance: {manual_cov}")  # 4.0 (population), 5.0 (sample)
```

#### Correlation

```python
# Pearson correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation}")  # 1.0 (perfect positive)

# Formula: r = Cov(X,Y) / (σ_X * σ_Y)
std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)
manual_corr = covariance / (std_x * std_y)
print(f"Manual correlation: {manual_corr}")  # 1.0

# Correlation matrix
data = np.array([x, y])
corr_matrix = np.corrcoef(data)
print(f"Correlation matrix:\n{corr_matrix}")
```

**Interpretation:**
- **+1**: Perfect positive correlation
- **0**: No linear correlation
- **-1**: Perfect negative correlation

**In ML:**
- Feature selection (remove highly correlated features)
- Multicollinearity detection
- Understanding feature relationships

### Changing the Data and Outliers

#### Outliers

Data points that are significantly different from other observations.

```python
# Detecting outliers using IQR method
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])  # 200 is an outlier

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")  # [200]

# Detecting outliers using Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(data))
outliers_z = data[z_scores > 3]
print(f"Outliers (Z-score > 3): {outliers_z}")
```

#### Impact of Outliers

Outliers can significantly skew measures of central tendency and spread:

```python
data_normal = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
data_with_outlier = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000])

print(f"Mean without outlier: {np.mean(data_normal):.2f}")      # 55.00
print(f"Mean with outlier: {np.mean(data_with_outlier):.2f}")   # 140.91

print(f"Median without outlier: {np.median(data_normal):.2f}")  # 55.00
print(f"Median with outlier: {np.median(data_with_outlier):.2f}")  # 60.00

print(f"Std without outlier: {np.std(data_normal):.2f}")        # 28.72
print(f"Std with outlier: {np.std(data_with_outlier):.2f}")     # 285.15
```

**Key Insight:** Median is more robust to outliers than mean.

#### Handling Outliers

```python
# Method 1: Trimming (remove outliers)
data_trimmed = data[(data >= lower_bound) & (data <= upper_bound)]

# Method 2: Capping (replace with bounds)
data_capped = np.clip(data, lower_bound, upper_bound)

# Method 3: Transformation (log, square root)
data_log = np.log1p(data)  # log(1+x) to handle zeros

# Method 4: Using robust statistics (median, IQR)
median = np.median(data)
mad = np.median(np.abs(data - median))  # Median Absolute Deviation
```

---

## Data Distribution

Understanding how data is spread and its patterns is crucial for statistical analysis and ML.

### Measures of Variability

#### Variance

Measures dispersion from the mean.

```python
data = np.array([10, 20, 30, 40, 50])
mean = np.mean(data)

# Variance formula: σ² = Σ(x - μ)² / N
variance = np.mean((data - mean) ** 2)
print(f"Variance: {variance}")  # 200.0

# Using NumPy
variance_np = np.var(data)
print(f"Variance (NumPy): {variance_np}")  # 200.0
```

#### Standard Deviation

Square root of variance. Same units as the data.

```python
# Standard deviation formula: σ = √variance
std_dev = np.sqrt(variance)
print(f"Standard Deviation: {std_dev:.2f}")  # 14.14

# Using NumPy
std_dev_np = np.std(data)
print(f"Standard Deviation (NumPy): {std_dev_np:.2f}")  # 14.14
```

**Interpretation:**
- Low SD: Data points are close to the mean
- High SD: Data points are spread out from the mean

### Frequency and Density Representation

#### Histogram

Bar graph showing frequency distribution.

```python
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins=30, color='steelblue', 
                                 edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram - Frequency Distribution')
plt.show()
```

#### Frequency Polygon

Line graph linking histogram midpoints.

```python
data = np.random.normal(100, 15, 1000)

# Create histogram
counts, bins = np.histogram(data, bins=30)

# Calculate midpoints
midpoints = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10, 6))
plt.plot(midpoints, counts, marker='o', linewidth=2, markersize=4)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Polygon')
plt.grid(True, alpha=0.3)
plt.show()
```

#### Density Curve

Smooth curve approximating data distribution.

```python
from scipy.stats import gaussian_kde

data = np.random.normal(100, 15, 1000)

# Create density curve using kernel density estimation
density = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 100)
y = density(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='darkblue')
plt.fill_between(x, y, alpha=0.3, color='lightblue')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density Curve')
plt.grid(True, alpha=0.3)
plt.show()
```

**Progression:** Histogram → Frequency Polygon → Density Curve

```python
# Complete progression visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
counts, bins = np.histogram(data, bins=30)
axes[0].hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('1. Histogram')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Frequency Polygon
midpoints = (bins[:-1] + bins[1:]) / 2
axes[1].plot(midpoints, counts, marker='o', linewidth=2, markersize=4)
axes[1].set_title('2. Frequency Polygon')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# Density Curve
density = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 100)
y = density(x)
axes[2].plot(x, y, linewidth=2, color='darkblue')
axes[2].fill_between(x, y, alpha=0.3, color='lightblue')
axes[2].set_title('3. Density Curve')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Density')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Types of Distributions

#### Symmetrical Distribution

Balanced around the center. Mean, median, and mode are at the same position.

```python
# Normal distribution is symmetrical
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label='Median')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Symmetrical Distribution')
plt.legend()
plt.show()

print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
# Mean and median are approximately equal
```

#### Right-Skewed Distribution

Tail extends to the right side. Mean > Median > Mode.

```python
# Right-skewed data (e.g., income, house prices)
data = np.random.gamma(2, 2, 1000)  # Gamma distribution is right-skewed

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label='Median')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Right-Skewed Distribution')
plt.legend()
plt.show()

print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
# Mean > Median (pulled right by tail)
```

#### Left-Skewed Distribution

Tail extends to the left side. Mean < Median < Mode.

```python
# Left-skewed data (e.g., exam scores with ceiling effect)
data = 100 - np.random.gamma(2, 2, 1000)  # Inverted gamma

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='--', linewidth=2, label='Median')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Left-Skewed Distribution')
plt.legend()
plt.show()

print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
# Mean < Median (pulled left by tail)
```

### Normal Distribution

Bell-shaped, symmetrical curve centered around the mean.

#### Properties

```python
mu = 100  # Mean
sigma = 15  # Standard deviation
data = np.random.normal(mu, sigma, 10000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, color='steelblue', edgecolor='black', alpha=0.7)

# Overlay theoretical normal curve
from scipy.stats import norm
x = np.linspace(data.min(), data.max(), 100)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r-', linewidth=2, label='Theoretical Normal')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 68-95-99.7 Rule (Empirical Rule)

For a normal distribution:
- **68%** of data within **1 standard deviation** of the mean
- **95%** of data within **2 standard deviations** of the mean
- **99.7%** of data within **3 standard deviations** of the mean

```python
mu = 100
sigma = 15
data = np.random.normal(mu, sigma, 10000)

# Calculate percentages
within_1sd = np.sum((data >= mu - sigma) & (data <= mu + sigma)) / len(data) * 100
within_2sd = np.sum((data >= mu - 2*sigma) & (data <= mu + 2*sigma)) / len(data) * 100
within_3sd = np.sum((data >= mu - 3*sigma) & (data <= mu + 3*sigma)) / len(data) * 100

print(f"Within 1 SD: {within_1sd:.1f}% (expected: 68%)")
print(f"Within 2 SD: {within_2sd:.1f}% (expected: 95%)")
print(f"Within 3 SD: {within_3sd:.1f}% (expected: 99.7%)")
```

#### Z-Score

Standardizes a value. Formula: `z = (X - μ) / σ`

```python
# Calculate Z-score
x = 130
mu = 100
sigma = 15
z_score = (x - mu) / sigma
print(f"Z-score: {z_score:.2f}")  # 2.00 (2 standard deviations above mean)

# Interpretation
if abs(z_score) < 1:
    interpretation = "Within 1 SD (68% of data)"
elif abs(z_score) < 2:
    interpretation = "Within 2 SD (95% of data)"
elif abs(z_score) < 3:
    interpretation = "Within 3 SD (99.7% of data)"
else:
    interpretation = "Outlier (beyond 3 SD)"

print(f"Interpretation: {interpretation}")

# Convert Z-score back to value
x_from_z = mu + z_score * sigma
print(f"Value from Z-score: {x_from_z}")  # 130.0
```

**Use Cases:**
- Standardizing different scales
- Outlier detection
- Comparing values from different distributions

---

## Probability Fundamentals

### Basic Concepts

#### Sample Space and Events

```python
# Example: Rolling a die
sample_space = {1, 2, 3, 4, 5, 6}
total_outcomes = len(sample_space)

# Event: Rolling an even number
even_numbers = {2, 4, 6}
favorable_outcomes = len(even_numbers)

# Probability
P_even = favorable_outcomes / total_outcomes
print(f"P(even) = {P_even}")  # 0.5
```

#### Probability Rules

```python
# 1. Probability ranges from 0 to 1
# 2. Sum of all probabilities = 1
# 3. Complement rule: P(not A) = 1 - P(A)

P_A = 0.3
P_not_A = 1 - P_A
print(f"P(not A) = {P_not_A}")  # 0.7

# 4. Addition rule: P(A or B) = P(A) + P(B) - P(A and B)
P_A = 0.4
P_B = 0.5
P_A_and_B = 0.2
P_A_or_B = P_A + P_B - P_A_and_B
print(f"P(A or B) = {P_A_or_B}")  # 0.7
```

### Conditional Probability

```python
# P(A|B) = P(A and B) / P(B)

# Example: Probability of rain given clouds
P_clouds = 0.5
P_rain_and_clouds = 0.3
P_rain_given_clouds = P_rain_and_clouds / P_clouds
print(f"P(Rain|Clouds) = {P_rain_given_clouds}")  # 0.6

# In ML: Naive Bayes classifier uses conditional probabilities
```

### Bayes' Theorem

```python
# P(A|B) = P(B|A) * P(A) / P(B)

# Example: Medical test
# P(Disease) = 0.01 (1% of population)
# P(Positive|Disease) = 0.99 (99% accurate if disease)
# P(Positive|No Disease) = 0.05 (5% false positive)

P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05

# P(Positive) = P(Positive|Disease)*P(Disease) + P(Positive|No Disease)*P(No Disease)
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_no_disease * (1 - P_disease))

# P(Disease|Positive) using Bayes' theorem
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
print(f"P(Disease|Positive) = {P_disease_given_positive:.4f}")  # ~0.167

# Surprising result: Even with 99% accuracy, only 16.7% chance of disease!
```

**In ML:**
- **Naive Bayes**: Classification using Bayes' theorem
- **Bayesian inference**: Updating beliefs with evidence
- **Bayesian optimization**: Hyperparameter tuning

### Independence

```python
# Events A and B are independent if:
# P(A and B) = P(A) * P(B)
# P(A|B) = P(A)

# Example: Two coin flips
P_heads1 = 0.5
P_heads2 = 0.5
P_both_heads = P_heads1 * P_heads2
print(f"P(Both heads) = {P_both_heads}")  # 0.25

# In ML: Naive Bayes assumes feature independence
```

---

## Probability Distributions

### Discrete Distributions

#### Binomial Distribution

```python
from scipy.stats import binom

# Parameters
n = 10  # Number of trials
p = 0.5  # Probability of success

# Probability mass function
k = 5  # Number of successes
prob = binom.pmf(k, n, p)
print(f"P(X = 5) = {prob:.4f}")  # Probability of exactly 5 successes

# Cumulative distribution
prob_at_most_5 = binom.cdf(5, n, p)
print(f"P(X ≤ 5) = {prob_at_most_5:.4f}")

# Generate samples
samples = binom.rvs(n, p, size=1000)
print(f"Mean: {np.mean(samples):.2f}")  # Should be close to n*p = 5
print(f"Variance: {np.var(samples):.2f}")  # Should be close to n*p*(1-p) = 2.5
```

**In ML:**
- Binary classification (success/failure)
- A/B testing

#### Poisson Distribution

```python
from scipy.stats import poisson

# Parameter: λ (lambda) - average rate
lam = 3  # Average 3 events per time period

# Probability of exactly k events
k = 5
prob = poisson.pmf(k, lam)
print(f"P(X = 5) = {prob:.4f}")

# Generate samples
samples = poisson.rvs(lam, size=1000)
print(f"Mean: {np.mean(samples):.2f}")  # Should be close to λ
print(f"Variance: {np.var(samples):.2f}")  # Should also be close to λ
```

**In ML:**
- Count data (number of events)
- Rare event modeling

### Continuous Distributions

#### Normal (Gaussian) Distribution

```python
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters
mu = 0    # Mean
sigma = 1 # Standard deviation

# Probability density function
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x, mu, sigma)

plt.plot(x, pdf, label=f'Normal(μ={mu}, σ={sigma})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Cumulative distribution function
prob_less_than_1 = norm.cdf(1, mu, sigma)
print(f"P(X < 1) = {prob_less_than_1:.4f}")  # ~0.8413

# Percentiles
percentile_95 = norm.ppf(0.95, mu, sigma)
print(f"95th percentile: {percentile_95:.2f}")  # ~1.64

# Generate samples
samples = np.random.normal(mu, sigma, 1000)
print(f"Sample mean: {np.mean(samples):.2f}")
print(f"Sample std: {np.std(samples):.2f}")
```

**Properties:**
- **68-95-99.7 Rule**: 
  - 68% within 1σ
  - 95% within 2σ
  - 99.7% within 3σ

**In ML:**
- Assumption in many algorithms
- Feature normalization
- Error terms in regression
- Prior distributions in Bayesian methods

#### Standard Normal Distribution

```python
# Standard normal: μ=0, σ=1
# Any normal can be standardized: z = (x - μ) / σ

x = 5
mu = 3
sigma = 2
z = (x - mu) / sigma
print(f"Standardized value: {z}")  # 1.0 (1 standard deviation above mean)
```

---

## Inferential Statistics

### Sampling

```python
# Population
population = np.random.normal(50, 15, 10000)

# Sample
sample_size = 100
sample = np.random.choice(population, sample_size, replace=False)

print(f"Population mean: {np.mean(population):.2f}")
print(f"Sample mean: {np.mean(sample):.2f}")
print(f"Population std: {np.std(population):.2f}")
print(f"Sample std: {np.std(sample, ddof=1):.2f}")  # Use ddof=1 for sample
```

### Central Limit Theorem

```python
# CLT: Sample means are normally distributed (for large n)

sample_means = []
for _ in range(1000):
    sample = np.random.choice(population, 30, replace=False)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True, alpha=0.7)
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Distribution of Sample Means (CLT)')
plt.axvline(np.mean(population), color='r', linestyle='--', label='Population Mean')
plt.legend()
plt.show()

print(f"Mean of sample means: {np.mean(sample_means):.2f}")
print(f"Std of sample means: {np.std(sample_means):.2f}")
print(f"Expected std (σ/√n): {np.std(population)/np.sqrt(30):.2f}")
```

### Confidence Intervals

```python
from scipy.stats import t

# 95% confidence interval for mean
sample = np.random.normal(50, 15, 100)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)

# t-distribution (for small samples)
alpha = 0.05
t_critical = t.ppf(1 - alpha/2, df=n-1)

margin_error = t_critical * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error

print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Sample mean: {sample_mean:.2f}")

# Interpretation: We're 95% confident the true mean lies in this interval
```

---

## Hypothesis Testing

A statistical method to make inferences about a population using sample data. Helps decide whether to accept or reject a claim (hypothesis).

### Types of Hypotheses

- **Null Hypothesis (H₀)**: Assumes no effect or no difference exists
- **Alternative Hypothesis (H₁)**: Assumes a significant effect or difference exists

### Steps in Hypothesis Testing

1. **Define H₀ and H₁**
2. **Set the significance level (α)**, commonly 0.05
3. **Choose the test statistic** (Z, t, χ², ANOVA)
4. **Compute the test statistic** from sample data
5. **Compare with the critical value** or **p-value**
6. **Reject H₀** if p-value < α; otherwise, **fail to reject H₀**

### Z-Test

Used when population variance is known or sample size > 30.

**Formula:** `Z = (X̄ - μ) / (σ / √n)`

Where:
- X̄ = Sample mean
- μ = Population mean
- σ = Population standard deviation
- n = Sample size

```python
from scipy.stats import norm

# Example: Test if sample mean differs from population mean
# H0: μ = 100
# H1: μ ≠ 100

sample_mean = 105
population_mean = 100
population_std = 15
sample_size = 100
alpha = 0.05

# Calculate Z-statistic
z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
print(f"Z-statistic: {z_stat:.4f}")  # 3.33

# Calculate p-value (two-tailed test)
p_value = 2 * (1 - norm.cdf(abs(z_stat)))
print(f"P-value: {p_value:.4f}")  # 0.0009

# Critical value
z_critical = norm.ppf(1 - alpha/2)
print(f"Critical value: ±{z_critical:.4f}")  # ±1.96

# Decision
if p_value < alpha:
    print("Reject H0: Sample mean is significantly different from population mean")
else:
    print("Fail to reject H0: No significant difference")
```

### t-test

```python
from scipy.stats import ttest_1samp, ttest_ind

# One-sample t-test
# H0: μ = 50
# H1: μ ≠ 50
sample = np.random.normal(52, 10, 30)
t_stat, p_value = ttest_1samp(sample, 50)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: Mean is significantly different from 50")
else:
    print("Fail to reject H0: No significant difference")

# Two-sample t-test
sample1 = np.random.normal(50, 10, 30)
sample2 = np.random.normal(55, 10, 30)
t_stat, p_value = ttest_ind(sample1, sample2)

print(f"\nTwo-sample t-test:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### Chi-Square (χ²) Test

Used for **categorical data** to test independence or goodness of fit.

**Formula:** `χ² = Σ [(O - E)² / E]`

Where:
- O = Observed value
- E = Expected value

```python
from scipy.stats import chi2_contingency, chi2

# Test independence between two categorical variables
observed = np.array([[10, 20, 30],
                      [15, 25, 35]])

chi2_stat, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected values:\n{expected}")

# Manual calculation
chi2_manual = np.sum((observed - expected) ** 2 / expected)
print(f"\nManual calculation: {chi2_manual:.4f}")

# Critical value
alpha = 0.05
chi2_critical = chi2.ppf(1 - alpha, dof)
print(f"Critical value: {chi2_critical:.4f}")

if p_value < alpha:
    print("\nReject H0: Variables are NOT independent")
else:
    print("\nFail to reject H0: Variables are independent")
```

**Types of Chi-Square Tests:**
- **Test of Independence**: Are two categorical variables related?
- **Goodness of Fit**: Does data follow a specific distribution?

### ANOVA (Analysis of Variance)

Used to compare **means of three or more groups**. Tests if at least one group mean differs significantly from others.

**Based on F-statistic:** `F = Variance between groups / Variance within groups`

```python
from scipy.stats import f_oneway

# Example: Compare test scores across three teaching methods
method1 = np.random.normal(75, 10, 30)
method2 = np.random.normal(80, 10, 30)
method3 = np.random.normal(85, 10, 30)

# One-way ANOVA
f_stat, p_value = f_oneway(method1, method2, method3)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: At least one group mean is significantly different")
else:
    print("Fail to reject H0: No significant difference between groups")
```

**ANOVA Assumptions:**
1. Normality: Data in each group is normally distributed
2. Homogeneity of variance: Groups have equal variances
3. Independence: Observations are independent

**Post-hoc Tests:**
If ANOVA is significant, use post-hoc tests (e.g., Tukey's HSD) to identify which groups differ.

```python
from scipy.stats import ttest_ind

# Pairwise comparisons (if ANOVA is significant)
# Note: In practice, use Tukey's HSD or Bonferroni correction
p12 = ttest_ind(method1, method2)[1]
p13 = ttest_ind(method1, method3)[1]
p23 = ttest_ind(method2, method3)[1]

print(f"\nPairwise comparisons (p-values):")
print(f"Method 1 vs Method 2: {p12:.4f}")
print(f"Method 1 vs Method 3: {p13:.4f}")
print(f"Method 2 vs Method 3: {p23:.4f}")
```

**Types of ANOVA:**
- **One-way ANOVA**: One factor with multiple levels
- **Two-way ANOVA**: Two factors
- **Repeated measures ANOVA**: Same subjects measured multiple times

---

## Applications in ML

### Model Evaluation

```python
# Confidence intervals for model accuracy
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier()

# Cross-validation scores
scores = cross_val_score(model, X, y, cv=10)

# 95% confidence interval
mean_score = np.mean(scores)
std_score = np.std(scores)
n = len(scores)
t_critical = t.ppf(0.975, df=n-1)
margin = t_critical * (std_score / np.sqrt(n))

print(f"Mean accuracy: {mean_score:.4f}")
print(f"95% CI: [{mean_score - margin:.4f}, {mean_score + margin:.4f}]")
```

### Feature Analysis

```python
# Check if feature is normally distributed
from scipy.stats import shapiro

feature = np.random.normal(0, 1, 100)
stat, p_value = shapiro(feature)

if p_value > 0.05:
    print("Feature appears normally distributed")
else:
    print("Feature does NOT appear normally distributed")
```

---

## Practice Exercises

### Exercise 1: Descriptive Statistics

Calculate all descriptive statistics for a dataset.

### Exercise 2: Probability Calculations

Solve probability problems using Bayes' theorem.

### Exercise 3: Hypothesis Testing

Perform t-tests to compare model performances.

---

## Resources

### Books

- **"Introduction to Statistical Learning"** by James et al.
- **"Think Stats"** by Allen Downey (free online)

### Online Courses

- [Khan Academy - Statistics](https://www.khanacademy.org/math/statistics-probability)
- [Coursera - Statistics](https://www.coursera.org/learn/statistics)

---

## Key Takeaways

1. **Descriptive stats** help understand data
2. **Probability** quantifies uncertainty
3. **Distributions** model data patterns
4. **Hypothesis testing** validates assumptions
5. **Confidence intervals** quantify uncertainty

---

**Remember**: Statistics helps you understand and evaluate your ML models!

