# Regression Quick Reference Guide

Quick reference for regression algorithms, metrics, and best practices.

## Table of Contents

- [Algorithm Selection](#algorithm-selection)
- [Code Snippets](#code-snippets)
- [Evaluation Metrics](#evaluation-metrics)
- [Common Issues & Solutions](#common-issues-solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Algorithm Selection

### Quick Decision Tree

```
Need to predict continuous value?
│
├─ Linear relationship?
│  ├─ YES → Linear Regression
│  └─ NO → Continue
│
├─ Non-linear but simple?
│  ├─ YES → Polynomial Regression
│  └─ NO → Continue
│
├─ Many features, risk of overfitting?
│  ├─ YES → Ridge Regression
│  └─ NO → Continue
│
├─ Need feature selection?
│  ├─ YES → Lasso Regression
│  └─ NO → Continue
│
├─ Need both regularization and feature selection?
│  ├─ YES → Elastic Net
│  └─ NO → Continue
│
└─ Many outliers?
   └─ YES → Robust Regression (Huber, RANSAC)
```

### Algorithm Comparison

| Algorithm | When to Use | Pros | Cons | Code |
|-----------|-------------|------|------|------|
| **Linear Regression** | Linear relationships, baseline | Simple, interpretable, fast | Assumes linearity | `LinearRegression()` |
| **Polynomial Regression** | Non-linear relationships | Captures curves | Can overfit | `PolynomialFeatures()` + `LinearRegression()` |
| **Ridge Regression** | Many features, multicollinearity | Prevents overfitting, stable | Doesn't eliminate features | `Ridge(alpha=1.0)` |
| **Lasso Regression** | Feature selection needed | Eliminates features, sparse | May eliminate important features | `Lasso(alpha=0.1)` |
| **Elastic Net** | Need both Ridge and Lasso benefits | Combines both approaches | More hyperparameters | `ElasticNet(alpha=0.1, l1_ratio=0.5)` |
| **Huber Regression** | Many outliers | Robust to outliers | Slower than linear | `HuberRegressor(epsilon=1.35)` |
| **RANSAC** | Extreme outliers | Very robust | Computationally expensive | `RANSACRegressor()` |

---

## Code Snippets

### Basic Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale (optional for linear regression, required for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

### Ridge Regression

```python
from sklearn.linear_model import Ridge

# Train
ridge = Ridge(alpha=1.0)  # alpha = regularization strength
ridge.fit(X_train_scaled, y_train)

# Predict
y_pred = ridge.predict(X_test_scaled)
```

### Lasso Regression

```python
from sklearn.linear_model import Lasso

# Train
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Check feature selection
selected_features = np.abs(lasso.coef_) > 0.001
print(f"Features used: {selected_features.sum()}")
```

### Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Train
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: 0=Ridge, 1=Lasso
elastic.fit(X_train_scaled, y_train)
```

### Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create pipeline
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Train
poly_reg.fit(X_train, y_train)

# Predict
y_pred = poly_reg.predict(X_test)
```

### Robust Regression (Huber)

```python
from sklearn.linear_model import HuberRegressor

# Train (less sensitive to outliers)
robust = HuberRegressor(epsilon=1.35)  # epsilon controls sensitivity
robust.fit(X_train_scaled, y_train)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Grid search
grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid.fit(X_train_scaled, y_train)

# Best model
best_model = grid.best_estimator_
print(f"Best alpha: {grid.best_params_['alpha']}")
```

---

## Evaluation Metrics

### Quick Reference

| Metric | Formula | When to Use | Interpretation |
|--------|---------|-------------|----------------|
| **MSE** | `(1/n) * Σ(y_true - y_pred)²` | General purpose | Lower is better, penalizes large errors |
| **RMSE** | `√MSE` | Most common | Same units as target, interpretable |
| **MAE** | `(1/n) * Σ\|y_true - y_pred\|` | Outliers matter | Average error, less sensitive to outliers |
| **R²** | `1 - (SS_res / SS_tot)` | Overall fit | 1 = perfect, 0 = baseline, <0 = worse than mean |
| **Adjusted R²** | `1 - (1-R²) * (n-1)/(n-p-1)` | Many features | Penalizes unnecessary features |

### Code

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Adjusted R²
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")
print(f"Adj R²: {adj_r2:.3f}")
```

---

## Common Issues & Solutions

### Issue 1: Overfitting

**Symptoms:**
- High training R², low test R²
- Large gap between train and test performance

**Solutions:**
- Use regularization (Ridge, Lasso)
- Reduce polynomial degree
- Get more training data
- Reduce number of features

### Issue 2: Underfitting

**Symptoms:**
- Low training R², low test R²
- Both similar but both low

**Solutions:**
- Add more features
- Use polynomial features
- Increase model complexity
- Remove regularization

### Issue 3: Multicollinearity

**Symptoms:**
- High VIF (>10)
- Unstable coefficients
- Large standard errors

**Solutions:**
- Remove highly correlated features
- Use Ridge regression
- Use PCA

### Issue 4: Outliers

**Symptoms:**
- Extreme residuals
- Poor model fit
- High RMSE

**Solutions:**
- Remove outliers (IQR method)
- Use robust regression (Huber, RANSAC)
- Transform target (log transform)

### Issue 5: Non-linear Relationships

**Symptoms:**
- Curved residual plot
- Low R² despite good features

**Solutions:**
- Use polynomial features
- Transform features (log, sqrt)
- Use non-linear models (tree-based)

### Issue 6: Heteroscedasticity

**Symptoms:**
- Funnel shape in residual plot
- Non-constant variance

**Solutions:**
- Transform target variable (log)
- Use weighted least squares
- Use robust standard errors

---

## Best Practices Checklist

### Before Training

- [ ] Data is clean (no missing values, outliers handled)
- [ ] Features and target are properly separated
- [ ] Data is split (train/test or train/val/test)
- [ ] Features are scaled (for regularization)
- [ ] Multicollinearity checked (VIF)

### During Training

- [ ] Start with simple model (Linear Regression)
- [ ] Use cross-validation for hyperparameter tuning
- [ ] Monitor training and validation performance
- [ ] Check for overfitting/underfitting

### After Training

- [ ] Evaluate on test set (only once!)
- [ ] Calculate multiple metrics (RMSE, MAE, R²)
- [ ] Perform residual analysis
- [ ] Check assumptions (normality, homoscedasticity)
- [ ] Interpret coefficients
- [ ] Visualize predictions vs actual

### Model Selection

- [ ] Compare multiple algorithms
- [ ] Use appropriate metric for problem
- [ ] Consider interpretability needs
- [ ] Balance complexity and performance

---

## Feature Scaling

### When to Scale

| Algorithm | Needs Scaling? | Why |
|-----------|----------------|-----|
| Linear Regression | Optional | Not required, but helps interpretation |
| Ridge | **Required** | Regularization is scale-dependent |
| Lasso | **Required** | Regularization is scale-dependent |
| Elastic Net | **Required** | Regularization is scale-dependent |
| Polynomial | Optional | Helps with numerical stability |

### Scaling Methods

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (most common)
scaler = StandardScaler()  # Mean=0, Std=1
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler
scaler = MinMaxScaler()  # Range [0, 1]
X_scaled = scaler.fit_transform(X_train)

# RobustScaler (for outliers)
scaler = RobustScaler()  # Uses median and IQR
X_scaled = scaler.fit_transform(X_train)
```

---

## Regularization Parameter Selection

### Alpha Values Guide

| Alpha | Effect | Use Case |
|-------|--------|----------|
| 0.001 | Very weak regularization | Almost like linear regression |
| 0.01 | Weak regularization | Slight smoothing |
| 0.1 | Moderate regularization | Default starting point |
| 1.0 | Strong regularization | Default for Ridge |
| 10.0 | Very strong | High multicollinearity |
| 100.0 | Extreme | Many features, small dataset |

### Finding Optimal Alpha

```python
# Use cross-validation
from sklearn.model_selection import GridSearchCV

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
grid = GridSearchCV(
    Ridge(),
    {'alpha': alphas},
    cv=5,
    scoring='neg_mean_squared_error'
)
grid.fit(X_train_scaled, y_train)
best_alpha = grid.best_params_['alpha']
```

---

## Residual Analysis Checklist

- [ ] Residuals vs Predicted plot (should be random)
- [ ] Q-Q plot (check normality)
- [ ] Histogram of residuals (should be normal)
- [ ] Scale-Location plot (check homoscedasticity)
- [ ] Mean of residuals ≈ 0
- [ ] No patterns or trends

---

## Model Interpretation

### Understanding Coefficients

```python
# Feature importance (absolute coefficients)
importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

# Interpretation:
# - Positive coefficient: Feature increases target
# - Negative coefficient: Feature decreases target
# - Larger absolute value: Stronger effect
```

### Confidence Intervals

```python
from scipy import stats

# Calculate confidence intervals for coefficients
n = len(y_train)
p = X_train.shape[1]
t_value = stats.t.ppf(0.975, n - p - 1)  # 95% CI

# Standard errors (simplified)
# Use statsmodels for accurate CIs
```

---

## Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Model not learning | Check data quality, add features |
| Overfitting | Increase regularization (alpha) |
| Underfitting | Decrease regularization, add features |
| Poor performance | Try different algorithm, check assumptions |
| Slow training | Reduce features, use simpler model |
| Unstable results | Check multicollinearity, use Ridge |

---

## Resources

- [Main Regression Guide](regression.md)
- [Advanced Topics](regression-advanced-topics.md)
- [Project Tutorial](regression-project-tutorial.md)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

---

**Remember**: Start simple, validate assumptions, iterate and improve!

