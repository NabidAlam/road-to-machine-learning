# Statistics and Probability for Machine Learning

Comprehensive guide to statistical concepts and probability theory essential for understanding and evaluating machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Descriptive Statistics](#descriptive-statistics)
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

### Chi-square Test

```python
from scipy.stats import chi2_contingency

# Test independence between two categorical variables
observed = np.array([[10, 20, 30],
                      [15, 25, 35]])

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("Variables are NOT independent")
else:
    print("Variables are independent")
```

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

