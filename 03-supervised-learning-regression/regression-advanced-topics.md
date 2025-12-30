# Advanced Regression Topics

Comprehensive guide to advanced regression techniques, diagnostics, and best practices.

## Table of Contents

- [Gradient Descent for Linear Regression](#gradient-descent-for-linear-regression)
- [Residual Analysis and Diagnostics](#residual-analysis-and-diagnostics)
- [Handling Outliers](#handling-outliers)
- [Multicollinearity Detection and Handling](#multicollinearity-detection-and-handling)
- [Feature Transformations](#feature-transformations)
- [Model Interpretation](#model-interpretation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Common Regression Pitfalls](#common-regression-pitfalls)

---

## Gradient Descent for Linear Regression

### Understanding How Linear Regression Works

Linear regression finds the best line by minimizing the cost function (MSE) using gradient descent.

### Cost Function

**Mean Squared Error (MSE):**
```
J(θ) = (1/2m) * Σ(h(x) - y)²
where:
- h(x) = θ₀ + θ₁x (prediction)
- m = number of samples
- θ₀ = intercept, θ₁ = slope
```

### Gradient Descent Algorithm

**Update Rule:**
```
θ₀ = θ₀ - α * (1/m) * Σ(h(x) - y)
θ₁ = θ₁ - α * (1/m) * Σ(h(x) - y) * x
where α = learning rate
```

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent for linear regression
    """
    m = len(y)
    theta = np.zeros(2)  # [intercept, slope]
    cost_history = []
    
    for i in range(iterations):
        # Predictions
        h = theta[0] + theta[1] * X
        
        # Calculate gradients
        grad_0 = (1/m) * np.sum(h - y)  # Gradient for intercept
        grad_1 = (1/m) * np.sum((h - y) * X)  # Gradient for slope
        
        # Update parameters
        theta[0] = theta[0] - learning_rate * grad_0
        theta[1] = theta[1] - learning_rate * grad_1
        
        # Calculate cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, Theta = {theta}")
    
    return theta, cost_history

# Generate data
np.random.seed(42)
X = np.random.rand(100) * 10
y = 2.5 * X + 1.5 + np.random.randn(100) * 2

# Run gradient descent
theta, costs = gradient_descent(X, y, learning_rate=0.01, iterations=1000)

print(f"\nFinal parameters:")
print(f"Intercept (θ₀): {theta[0]:.2f}")
print(f"Slope (θ₁): {theta[1]:.2f}")

# Visualize cost convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
x_line = np.linspace(0, 10, 100)
y_line = theta[0] + theta[1] * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Learning Rate Selection

```python
# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, lr in enumerate(learning_rates):
    theta, costs = gradient_descent(X, y, learning_rate=lr, iterations=1000)
    axes[idx//2, idx%2].plot(costs)
    axes[idx//2, idx%2].set_title(f'Learning Rate = {lr}')
    axes[idx//2, idx%2].set_xlabel('Iteration')
    axes[idx//2, idx%2].set_ylabel('Cost')
    axes[idx//2, idx%2].grid(True)

plt.tight_layout()
plt.show()

# Too small: Slow convergence
# Too large: May diverge
# Just right: Fast, stable convergence
```

---

## Residual Analysis and Diagnostics

### What are Residuals?

**Residuals** = Actual - Predicted = y - ŷ

Residuals tell us how well our model fits the data.

### Diagnostic Plots

#### 1. Residual Plot

```python
def plot_residuals(y_true, y_pred):
    """
    Create comprehensive residual plots
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True)
    
    # 2. Q-Q Plot (Normality check)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    axes[0, 1].grid(True)
    
    # 3. Histogram of Residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True)
    
    # 4. Scale-Location Plot (Homoscedasticity)
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('√|Standardized Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    # Normality test
    stat, p_value = shapiro(residuals[:5000])  # Limit for large datasets
    print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")
    if p_value > 0.05:
        print("  Residuals appear normally distributed")
    else:
        print("  Residuals may not be normally distributed")
    
    # Mean of residuals (should be ~0)
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.6f} (should be ~0)")
    print(f"  Std:  {residuals.std():.2f}")
    print(f"  Min:  {residuals.min():.2f}")
    print(f"  Max:  {residuals.max():.2f}")

# Example usage
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

plot_residuals(y_test, y_pred)
```

### What to Look For

**Good Residuals:**
- Randomly scattered around zero
- No patterns or trends
- Constant variance (homoscedasticity)
- Normally distributed

**Bad Residuals (Problems):**
- **Funnel shape**: Heteroscedasticity (non-constant variance)
- **Curved pattern**: Non-linear relationship
- **Trend**: Missing important features
- **Outliers**: Extreme values affecting model

---

## Handling Outliers

### Detecting Outliers

#### 1. Using Z-Score

```python
def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using Z-score
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = z_scores > threshold
    return outliers

# Example
outliers = detect_outliers_zscore(y)
print(f"Number of outliers: {outliers.sum()}")
```

#### 2. Using IQR (Interquartile Range)

```python
def detect_outliers_iqr(data):
    """
    Detect outliers using IQR method
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

outliers = detect_outliers_iqr(y)
print(f"Number of outliers: {outliers.sum()}")
```

#### 3. Using Residuals

```python
# Outliers in regression context
residuals = y_test - y_pred
residual_std = residuals.std()
outliers = np.abs(residuals) > 3 * residual_std

print(f"Outliers based on residuals: {outliers.sum()}")
```

### Handling Outliers

#### Option 1: Remove Outliers

```python
# Remove outliers
X_clean = X[~outliers]
y_clean = y[~outliers]

# Retrain model
model_clean = LinearRegression()
model_clean.fit(X_clean, y_clean)
```

#### Option 2: Transform Target Variable

```python
# Log transformation (for right-skewed data)
y_log = np.log1p(y)  # log1p handles zeros

# Train on transformed target
model_log = LinearRegression()
model_log.fit(X_train_scaled, y_log)
y_pred_log = model_log.predict(X_test_scaled)

# Transform back
y_pred = np.expm1(y_pred_log)
```

#### Option 3: Use Robust Regression

```python
from sklearn.linear_model import HuberRegressor

# Huber regression is less sensitive to outliers
robust_model = HuberRegressor(epsilon=1.35)  # epsilon controls sensitivity
robust_model.fit(X_train_scaled, y_train)
y_pred_robust = robust_model.predict(X_test_scaled)

# Compare
print(f"Standard Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Robust Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_robust)):.2f}")
```

---

## Multicollinearity Detection and Handling

### What is Multicollinearity?

High correlation between features, making it hard to determine individual feature importance.

### Detecting Multicollinearity

#### 1. Correlation Matrix

```python
import pandas as pd
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = pd.DataFrame(X_train_scaled).corr()

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print("Highly correlated feature pairs (>0.8):")
for feat1, feat2, corr in high_corr_pairs:
    print(f"  {feat1} - {feat2}: {corr:.3f}")
```

#### 2. Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    Calculate VIF for each feature
    VIF > 10 indicates multicollinearity
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = range(X.shape[1])
    vif_data["VIF"] = [variance_inflation_factor(X, i) 
                       for i in range(X.shape[1])]
    return vif_data

vif_df = calculate_vif(X_train_scaled)
print("\nVariance Inflation Factors:")
print(vif_df)

# Features with VIF > 10 have multicollinearity
high_vif = vif_df[vif_df["VIF"] > 10]
if len(high_vif) > 0:
    print(f"\nFeatures with high VIF (>10): {len(high_vif)}")
    print("Consider removing or combining these features")
```

### Handling Multicollinearity

#### Option 1: Remove Highly Correlated Features

```python
# Remove one feature from each highly correlated pair
features_to_remove = set()
for feat1, feat2, corr in high_corr_pairs:
    # Keep feature with lower correlation to target
    corr1 = np.corrcoef(X_train[:, int(feat1)], y_train)[0, 1]
    corr2 = np.corrcoef(X_train[:, int(feat2)], y_train)[0, 1]
    if abs(corr1) < abs(corr2):
        features_to_remove.add(int(feat1))
    else:
        features_to_remove.add(int(feat2))

X_reduced = np.delete(X_train_scaled, list(features_to_remove), axis=1)
```

#### Option 2: Use Regularization (Ridge/Lasso)

```python
# Ridge regression handles multicollinearity well
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

#### Option 3: Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

# Reduce dimensions using PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_train_scaled)

# Train on reduced features
model_pca = LinearRegression()
model_pca.fit(X_pca, y_train)
```

---

## Feature Transformations

### Why Transform Features?

- Handle non-linear relationships
- Normalize distributions
- Reduce skewness
- Improve model performance

### Common Transformations

#### 1. Log Transformation

```python
# For right-skewed data
X_log = np.log1p(X)  # log1p = log(1+x), handles zeros

# Visualize effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(X.flatten(), bins=50, edgecolor='black')
axes[0].set_title('Original Distribution')
axes[0].set_xlabel('Value')

axes[1].hist(X_log.flatten(), bins=50, edgecolor='black')
axes[1].set_title('Log-Transformed Distribution')
axes[1].set_xlabel('Log(Value)')
plt.tight_layout()
plt.show()
```

#### 2. Square Root Transformation

```python
# For moderately skewed data
X_sqrt = np.sqrt(X)
```

#### 3. Box-Cox Transformation

```python
from scipy import stats

# Box-Cox requires positive values
X_positive = X - X.min() + 1
X_boxcox, lambda_param = stats.boxcox(X_positive.flatten())

print(f"Optimal lambda: {lambda_param:.3f}")
```

#### 4. Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (already covered in main guide)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### When to Use Each Transformation

- **Log**: Right-skewed data, multiplicative relationships
- **Square Root**: Moderately skewed data
- **Box-Cox**: Automatically finds best transformation
- **Polynomial**: Non-linear relationships

---

## Model Interpretation

### Understanding Coefficients

```python
def interpret_model(model, feature_names, X_scaled):
    """
    Interpret linear regression model
    """
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Create interpretation dataframe
    interpretation = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("Model Interpretation:")
    print(f"Intercept: {intercept:.3f}")
    print("\nFeature Coefficients (sorted by importance):")
    print(interpretation)
    
    # Interpretation
    print("\nInterpretation:")
    print("Positive coefficient: Feature increases target")
    print("Negative coefficient: Feature decreases target")
    print("Larger absolute value: Stronger effect")
    
    return interpretation

# Example
housing = fetch_california_housing()
feature_names = housing.feature_names
interpretation = interpret_model(model, feature_names, X_train_scaled)
```

### Confidence Intervals

```python
from scipy import stats

def confidence_intervals(model, X, y, alpha=0.05):
    """
    Calculate confidence intervals for coefficients
    """
    n = len(y)
    p = X.shape[1]  # number of features
    
    # Predictions
    y_pred = model.predict(X)
    
    # Residuals
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    
    # Standard errors
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # t-statistic
    t_value = stats.t.ppf(1 - alpha/2, n - p - 1)
    
    # Confidence intervals
    coef_with_intercept = np.concatenate([[model.intercept_], model.coef_])
    lower = coef_with_intercept - t_value * std_errors
    upper = coef_with_intercept + t_value * std_errors
    
    return lower, upper, std_errors

lower, upper, std_errors = confidence_intervals(model, X_train_scaled, y_train)

print("Coefficient Confidence Intervals (95%):")
print(f"Intercept: [{lower[0]:.3f}, {upper[0]:.3f}]")
for i, name in enumerate(feature_names):
    print(f"{name}: [{lower[i+1]:.3f}, {upper[i+1]:.3f}]")
```

---

## Hyperparameter Tuning

### Grid Search for Regularization

```python
from sklearn.model_selection import GridSearchCV

# Ridge regression
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_grid = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

ridge_grid.fit(X_train_scaled, y_train)

print("Best Ridge Parameters:")
print(f"  Alpha: {ridge_grid.best_params_['alpha']}")
print(f"  Best CV Score (RMSE): {np.sqrt(-ridge_grid.best_score_):.3f}")

# Lasso regression
lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

lasso_grid = GridSearchCV(
    Lasso(max_iter=10000),
    lasso_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

lasso_grid.fit(X_train_scaled, y_train)

print("\nBest Lasso Parameters:")
print(f"  Alpha: {lasso_grid.best_params_['alpha']}")
print(f"  Best CV Score (RMSE): {np.sqrt(-lasso_grid.best_score_):.3f}")

# Elastic Net
elastic_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_grid = GridSearchCV(
    ElasticNet(max_iter=10000),
    elastic_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

elastic_grid.fit(X_train_scaled, y_train)

print("\nBest Elastic Net Parameters:")
print(f"  Alpha: {elastic_grid.best_params_['alpha']}")
print(f"  L1 Ratio: {elastic_grid.best_params_['l1_ratio']}")
print(f"  Best CV Score (RMSE): {np.sqrt(-elastic_grid.best_score_):.3f}")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    """
    Plot learning curves to diagnose bias/variance
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('MSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curves(model, X_train_scaled, y_train)
```

---

## Common Regression Pitfalls

### Pitfall 1: Ignoring Assumptions

**Problem**: Linear regression assumes linearity, normality, homoscedasticity.

**Solution**: Always check assumptions with diagnostic plots.

### Pitfall 2: Not Handling Outliers

**Problem**: Outliers can heavily influence linear regression.

**Solution**: Detect and handle outliers (remove, transform, or use robust regression).

### Pitfall 3: Multicollinearity

**Problem**: Highly correlated features make interpretation difficult.

**Solution**: Check VIF, remove redundant features, or use regularization.

### Pitfall 4: Overfitting with Polynomial Features

**Problem**: High-degree polynomials can overfit.

**Solution**: Use cross-validation to select optimal degree, use regularization.

### Pitfall 5: Wrong Evaluation Metric

**Problem**: Using R² alone can be misleading.

**Solution**: Use multiple metrics (RMSE, MAE, R²) and residual analysis.

### Pitfall 6: Not Scaling Features

**Problem**: Regularization requires scaled features.

**Solution**: Always scale before Ridge/Lasso/Elastic Net.

### Pitfall 7: Data Leakage

**Problem**: Scaling or transforming before train/test split.

**Solution**: Always split first, then fit transformers on training data only.

---

## Key Takeaways

1. **Gradient Descent**: Understand how models learn
2. **Residual Analysis**: Essential for model diagnostics
3. **Outliers**: Detect and handle appropriately
4. **Multicollinearity**: Check VIF, use regularization if needed
5. **Transformations**: Handle non-linear relationships
6. **Interpretation**: Understand what coefficients mean
7. **Tuning**: Use cross-validation for hyperparameters
8. **Diagnostics**: Always check assumptions

---

**Remember**: A good regression model requires careful diagnostics and validation!

