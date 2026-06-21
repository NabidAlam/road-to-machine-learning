# Supervised Learning - Regression Complete Guide

This guide covers regression algorithms for predicting continuous values.

## ML for beginners curriculum map (this guide)

Beginner-friendly checklist; each line links to **code** below or to the intro foundations.

- **Linear regression and evaluation metrics** → [Linear regression](#linear-regression), [Evaluation metrics](#evaluation-metrics)
- **Multiple and polynomial regression** → [Linear regression](#linear-regression) (multiple features), [Polynomial regression](#polynomial-regression)
- **Data distribution analysis** (percentiles, histogram, boxplot) → [Data distribution analysis](#data-distribution-analysis)
- **Feature relationships** (scatter, covariance, correlation) → [Feature relationship analysis](#feature-relationship-analysis)
- **End-to-end persistence** (save and load model) → [End-to-end train, evaluate, and persist](#end-to-end-train-evaluate-and-persist)
- **Statistics foundations** → [Introduction to ML. Descriptive stats](../02-introduction-to-ml/introduction-to-ml.md#descriptive-statistics-and-sampling-foundations)

## Table of Contents

- [ML for beginners curriculum map (this guide)](#ml-for-beginners-curriculum-map-this-guide)
- [Introduction to Regression](#introduction-to-regression)
- [Linear Regression](#linear-regression)
- [Polynomial Regression](#polynomial-regression)
- [Regularization](#regularization)
- [Evaluation Metrics](#evaluation-metrics)
- [Data distribution analysis](#data-distribution-analysis)
- [Feature relationship analysis](#feature-relationship-analysis)
- [End-to-end train, evaluate, and persist](#end-to-end-train-evaluate-and-persist)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Regression

### What is Regression?

Regression predicts continuous numerical values. Unlike classification (which predicts categories), regression predicts quantities.

**Examples:**
- House prices
- Temperature
- Stock prices
- Sales revenue
- Age

### When to Use Regression

- **Target variable is continuous** (not categorical)
- **Want to predict a quantity**
- **Relationship between features and target**

---

## Linear Regression

### Simple Linear Regression

Predicts target using a single feature with a linear relationship.

**Formula:**
```
y = mx + b
where:
- y = predicted value
- m = slope (coefficient)
- x = feature value
- b = intercept
```

**Example:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + 1.5 + np.random.randn(100) * 2

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
```

**Output:**
```
Coefficient (slope): 2.48
Intercept: 1.65
MSE: 3.89
R²: 0.92
```

### Multiple Linear Regression

Uses multiple features to predict target.

**Formula:**
```
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

**Example:**
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for interpretation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Feature importance (coefficients)
feature_names = housing.feature_names
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.3f}")
```

### Assumptions of Linear Regression

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

**Checking Assumptions:**
```python
from scipy import stats

# Residuals
residuals = y_test - y_pred

# 1. Check normality of residuals
stat, p_value = stats.shapiro(residuals[:1000])  # Limit for large datasets
print(f"Normality test p-value: {p_value:.4f}")

# 2. Check homoscedasticity (visual)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

---

## Polynomial Regression

### When to Use

When relationship is non-linear but can be captured with polynomial terms.

**Formula:**
```
y = b₀ + b₁x + b₂x² + b₃x³ + ...
```

**Example:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**2 - 2 * X.flatten() + 3 + np.random.randn(100) * 2

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predictions
y_pred = model.predict(X_poly)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (degree=2)')
plt.legend()
plt.show()

# Evaluate
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.2f}")
```

**Warning:** Higher degree polynomials can overfit!

```python
# Compare different degrees
degrees = [1, 2, 5, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    axes[idx//2, idx%2].scatter(X, y, alpha=0.3)
    axes[idx//2, idx%2].plot(X, y_pred, 'r-', linewidth=2)
    axes[idx//2, idx%2].set_title(f'Degree {degree}')
    axes[idx//2, idx%2].set_xlabel('X')
    axes[idx//2, idx%2].set_ylabel('y')

plt.tight_layout()
plt.show()
```

---

## Regularization

### Why Regularization?

Prevents overfitting by penalizing large coefficients.

### Ridge Regression (L2)

Penalizes sum of squared coefficients.

**Formula:**
```
Loss = MSE + α * Σ(coefficients²)
```

**Example:**
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Create Ridge regression
ridge = Ridge(alpha=1.0)  # alpha = regularization strength
ridge.fit(X_train_scaled, y_train)

# Predictions
y_pred = ridge.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Ridge - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Compare coefficients
print("\nLinear Regression coefficients:")
print(model.coef_)
print("\nRidge coefficients (smaller):")
print(ridge.coef_)

# Cross-validation to find best alpha
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    score = cross_val_score(ridge, X_train_scaled, y_train, 
                           cv=5, scoring='neg_mean_squared_error')
    scores.append(-score.mean())

best_alpha = alphas[np.argmin(scores)]
print(f"\nBest alpha: {best_alpha}")
```

### Lasso Regression (L1)

Penalizes sum of absolute coefficients. Can set coefficients to zero (feature selection).

**Formula:**
```
Loss = MSE + α * Σ|coefficients|
```

**Example:**
```python
from sklearn.linear_model import Lasso

# Create Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Predictions
y_pred = lasso.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Lasso - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Feature selection (coefficients set to zero)
print("\nLasso coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    if abs(coef) > 0.001:  # Non-zero coefficients
        print(f"{name}: {coef:.3f}")
    else:
        print(f"{name}: 0.000 (eliminated)")
```

### Elastic Net

Combines Ridge and Lasso.

**Formula:**
```
Loss = MSE + α * (λ * L1 + (1-λ) * L2)
```

**Example:**
```python
from sklearn.linear_model import ElasticNet

# Create Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
elastic.fit(X_train_scaled, y_train)

# Predictions
y_pred = elastic.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Elastic Net - RMSE: {rmse:.2f}, R²: {r2:.2f}")
```

### Comparison

```python
# Compare all methods
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R²': r2}

# Display results
for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.3f}")
```

---

## Evaluation Metrics

### Mean Squared Error (MSE)

Average of squared differences between actual and predicted.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# Formula: MSE = (1/n) * Σ(y_true - y_pred)²
```

### Root Mean Squared Error (RMSE)

Square root of MSE. Same units as target variable.

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# More interpretable than MSE
# If RMSE = 50000, predictions are off by ~$50,000 on average
```

### Mean Absolute Error (MAE)

Average of absolute differences. Less sensitive to outliers.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# Formula: MAE = (1/n) * Σ|y_true - y_pred|
```

### R² Score (Coefficient of Determination)

Proportion of variance explained. Range: -∞ to 1 (1 = perfect).

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.3f}")

# Interpretation:
# R² = 0.85 means model explains 85% of variance
# R² = 1.0 means perfect predictions
# R² < 0 means model worse than baseline (mean)
```

### Adjusted R²

R² adjusted for number of features. Penalizes adding unnecessary features.

```python
def adjusted_r2_score(y_true, y_pred, n_features):
    """Calculate adjusted R²"""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2

adj_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])
print(f"Adjusted R²: {adj_r2:.3f}")
```

### Complete Evaluation

```python
def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Regression Metrics:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    # Residual analysis
    residuals = y_true - y_pred
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.2f}")
    print(f"  Std:  {residuals.std():.2f}")
    print(f"  Min:  {residuals.min():.2f}")
    print(f"  Max:  {residuals.max():.2f}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

metrics = evaluate_regression(y_test, y_pred)
```

---

## Data distribution analysis

Use **percentiles**, **histograms**, and **boxplots** to check skew, outliers, and whether targets or residuals look reasonable before and after modeling.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
# Toy target: house prices in $1000s (right-skewed log-normal style)
prices = rng.lognormal(mean=4.5, sigma=0.35, size=500)

print("Percentiles (5, 25, 50, 75, 95):", np.percentile(prices, [5, 25, 50, 75, 95]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(prices, bins=30, color="steelblue", edgecolor="white")
axes[0].set_title("Histogram of target")
axes[0].set_xlabel("Price ($1000s)")

axes[1].boxplot(prices, vert=True)
axes[1].set_title("Boxplot of target")
plt.tight_layout()
plt.show()
```

---

## Feature relationship analysis

**Scatter plots** show pairs of variables; **covariance** and **correlation** summarize linear association (Pearson correlation is scaled covariance).

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
x = rng.normal(0, 1, 200)
y = 0.7 * x + rng.normal(0, 0.5, 200)  # correlated noise
df = pd.DataFrame({"x": x, "y": y})

print(df.cov())    # covariance matrix
print(df.corr())   # Pearson correlation matrix

import matplotlib.pyplot as plt
plt.scatter(df["x"], df["y"], alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter: feature vs synthetic target")
plt.show()
```

---

## End-to-end train, evaluate, and persist

Minimal **save → load → predict** flow (real deployment adds APIs, monitoring, and versioning—see the [model deployment](../13-model-deployment/README.md) module).

```python
from pathlib import Path
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

artifact = Path("regression_model.joblib")
joblib.dump(model, artifact)
loaded = joblib.load(artifact)
print("Same preds?", np.allclose(loaded.predict(X_test[:5]), model.predict(X_test[:5])))
```

---

## Practice Exercises

### Exercise 1: Simple Linear Regression

**Task:** Create a linear regression model to predict salary from years of experience.

**Solution:**
```python
# Generate data
years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salary = np.array([45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000])

# Reshape for sklearn
X = years.reshape(-1, 1)
y = salary

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluate
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R²: {r2_score(y, y_pred):.3f}")

# Predict for new value
new_years = np.array([[12]])
predicted_salary = model.predict(new_years)
print(f"\nPredicted salary for 12 years: ${predicted_salary[0]:,.2f}")
```

### Exercise 2: Multiple Regression with Regularization

**Task:** Use California housing dataset, compare Linear, Ridge, and Lasso regression.

**Solution:**
```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:12s} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
```

### Exercise 3: Polynomial Regression

**Task:** Fit polynomial regression of different degrees and find optimal degree using cross-validation.

**Solution:**
```python
from sklearn.model_selection import cross_val_score

# Generate non-linear data
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.flatten()**2 - 5 * X.flatten() + 3 + np.random.randn(50) * 5

# Test different degrees
degrees = range(1, 8)
cv_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, 
                           scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Find best degree
best_degree = degrees[np.argmin(cv_scores)]
print(f"Best degree: {best_degree}")

# Train with best degree
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

print(f"R²: {r2_score(y, y_pred):.3f}")
```

---

## Key Takeaways

1. **Linear Regression**: Simple, interpretable, good baseline
2. **Polynomial Regression**: Captures non-linear relationships
3. **Regularization**: Prevents overfitting (Ridge, Lasso, Elastic Net)
4. **Evaluation**: Use multiple metrics (RMSE, MAE, R²)
5. **Feature Scaling**: Important for regularization

---

## Next Steps

- Practice with real datasets
- Experiment with different regularization strengths
- Move to [04-supervised-learning-classification](../04-supervised-learning-classification/README.md) for classification

**Try next:** Start simple (linear), then add complexity only if needed!

