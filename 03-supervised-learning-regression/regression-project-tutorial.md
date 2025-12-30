# Complete Regression Project Tutorial

Step-by-step walkthrough of building a real-world regression model from data exploration to deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Exploration](#step-1-data-loading-and-exploration)
- [Step 2: Data Cleaning and Preprocessing](#step-2-data-cleaning-and-preprocessing)
- [Step 3: Feature Engineering](#step-3-feature-engineering)
- [Step 4: Model Training](#step-4-model-training)
- [Step 5: Model Evaluation](#step-5-model-evaluation)
- [Step 6: Model Improvement](#step-6-model-improvement)
- [Step 7: Final Model and Predictions](#step-7-final-model-and-predictions)

---

## Project Overview

**Project**: Predict House Prices

**Dataset**: California Housing Dataset (or any house price dataset)

**Goal**: Build a regression model to predict median house values

**Type**: Multiple Linear Regression with Regularization

**Difficulty**: Intermediate

**Time**: 1-2 hours

---

## Step 1: Data Loading and Exploration

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
```

### Basic Statistics

```python
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable (MedHouseVal) Statistics:")
print(df['MedHouseVal'].describe())
```

### Visualizations

```python
# Distribution of target variable
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['MedHouseVal'], bins=50, edgecolor='black')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Distribution of House Values')

plt.subplot(1, 2, 2)
plt.boxplot(df['MedHouseVal'])
plt.ylabel('Median House Value')
plt.title('Box Plot of House Values')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter plots of features vs target
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, feature in enumerate(housing.feature_names):
    axes[idx].scatter(df[feature], df['MedHouseVal'], alpha=0.3)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('MedHouseVal')
    axes[idx].set_title(f'{feature} vs House Value')

plt.tight_layout()
plt.show()
```

**Insights:**
- Check for outliers in target variable
- Identify highly correlated features
- Understand feature distributions
- Check for non-linear relationships

---

## Step 2: Data Cleaning and Preprocessing

### Handle Missing Values

```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# If there are missing values, handle them
# Option 1: Drop rows with missing values
# df = df.dropna()

# Option 2: Fill with median (for numerical)
# df = df.fillna(df.median())

# Option 3: Fill with mean
# df = df.fillna(df.mean())
```

### Handle Outliers

```python
# Detect outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from target variable
print(f"Original shape: {df.shape}")
df_clean = remove_outliers_iqr(df, 'MedHouseVal')
print(f"After removing outliers: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} outliers ({100*(len(df) - len(df_clean))/len(df):.1f}%)")

# Visualize before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(df['MedHouseVal'])
axes[0].set_title('Before Outlier Removal')
axes[0].set_ylabel('MedHouseVal')

axes[1].boxplot(df_clean['MedHouseVal'])
axes[1].set_title('After Outlier Removal')
axes[1].set_ylabel('MedHouseVal')

plt.tight_layout()
plt.show()

df = df_clean  # Use cleaned data
```

### Prepare Features and Target

```python
# Separate features and target
X = df[housing.feature_names]
y = df['MedHouseVal']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

---

## Step 3: Feature Engineering

### Check for Multicollinearity

```python
# Calculate VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

vif_df = calculate_vif(X)
print("Variance Inflation Factors:")
print(vif_df)

# Features with VIF > 10 have multicollinearity
high_vif = vif_df[vif_df["VIF"] > 10]
if len(high_vif) > 0:
    print(f"\nWarning: {len(high_vif)} features have high VIF (>10)")
    print("Consider using regularization (Ridge/Lasso)")
```

### Create New Features (Optional)

```python
# Example: Create interaction features
# X['MedInc_x_AveRooms'] = X['MedInc'] * X['AveRooms']
# X['Population_x_HouseAge'] = X['Population'] * X['HouseAge']

# Example: Create polynomial features for specific features
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
# X_poly = poly.fit_transform(X[['MedInc', 'AveRooms']])
```

### Split Data

```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### Scale Features

```python
# Scale features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=housing.feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=housing.feature_names)

print("Features scaled successfully!")
```

---

## Step 4: Model Training

### Baseline Model (Linear Regression)

```python
# Train baseline model
baseline_model = LinearRegression()
baseline_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = baseline_model.predict(X_train_scaled)
y_test_pred = baseline_model.predict(X_test_scaled)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Baseline Model (Linear Regression):")
print(f"  Training RMSE: {train_rmse:.3f}")
print(f"  Test RMSE: {test_rmse:.3f}")
print(f"  Training R²: {train_r2:.3f}")
print(f"  Test R²: {test_r2:.3f}")

# Check for overfitting
if train_rmse < test_rmse * 0.9:
    print("  Warning: Possible overfitting (large gap between train and test)")
```

### Ridge Regression

```python
# Train Ridge regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

y_test_pred_ridge = ridge_model.predict(X_test_scaled)
test_rmse_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

print("\nRidge Regression:")
print(f"  Test RMSE: {test_rmse_ridge:.3f}")
print(f"  Test R²: {test_r2_ridge:.3f}")
```

### Lasso Regression

```python
# Train Lasso regression
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

y_test_pred_lasso = lasso_model.predict(X_test_scaled)
test_rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)

print("\nLasso Regression:")
print(f"  Test RMSE: {test_rmse_lasso:.3f}")
print(f"  Test R²: {test_r2_lasso:.3f}")

# Check feature selection
non_zero_coef = np.sum(np.abs(lasso_model.coef_) > 0.001)
print(f"  Features used: {non_zero_coef} out of {len(housing.feature_names)}")
```

---

## Step 5: Model Evaluation

### Compare All Models

```python
models = {
    'Linear Regression': (baseline_model, y_test_pred),
    'Ridge': (ridge_model, y_test_pred_ridge),
    'Lasso': (lasso_model, y_test_pred_lasso)
}

results = []
for name, (model, y_pred) in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
```

### Residual Analysis

```python
# Choose best model (lowest RMSE)
best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
best_model = models[best_model_name][0]
best_predictions = models[best_model_name][1]

print(f"\nBest Model: {best_model_name}")

# Residual analysis
residuals = y_test - best_predictions

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Predicted
axes[0, 0].scatter(best_predictions, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted')
axes[0, 0].grid(True)

# 2. Q-Q Plot
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

# 4. Actual vs Predicted
axes[1, 1].scatter(y_test, best_predictions, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title('Actual vs Predicted')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Residual statistics
print("\nResidual Statistics:")
print(f"  Mean: {residuals.mean():.6f} (should be ~0)")
print(f"  Std: {residuals.std():.3f}")
print(f"  Min: {residuals.min():.3f}")
print(f"  Max: {residuals.max():.3f}")
```

### Feature Importance

```python
# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': best_model.coef_,
    'Abs_Coefficient': np.abs(best_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (sorted by absolute coefficient):")
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.tight_layout()
plt.show()
```

---

## Step 6: Model Improvement

### Hyperparameter Tuning

```python
# Tune Ridge regression
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

print("Ridge Regression - Best Parameters:")
print(f"  Alpha: {ridge_grid.best_params_['alpha']}")
print(f"  Best CV RMSE: {np.sqrt(-ridge_grid.best_score_):.3f}")

# Tune Lasso regression
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

print("\nLasso Regression - Best Parameters:")
print(f"  Alpha: {lasso_grid.best_params_['alpha']}")
print(f"  Best CV RMSE: {np.sqrt(-lasso_grid.best_score_):.3f}")

# Use best model
best_tuned_model = ridge_grid.best_estimator_
y_test_pred_tuned = best_tuned_model.predict(X_test_scaled)

test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_test_pred_tuned))
test_r2_tuned = r2_score(y_test, y_test_pred_tuned)

print(f"\nTuned Model Performance:")
print(f"  Test RMSE: {test_rmse_tuned:.3f}")
print(f"  Test R²: {test_r2_tuned:.3f}")
```

### Cross-Validation

```python
# Cross-validation scores
cv_scores = cross_val_score(
    best_tuned_model, X_train_scaled, y_train,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"\nCross-Validation Results:")
print(f"  Mean RMSE: {np.sqrt(-cv_scores.mean()):.3f}")
print(f"  Std RMSE: {np.sqrt(cv_scores.std()):.3f}")
print(f"  95% Confidence Interval: "
      f"[{np.sqrt(-cv_scores.mean() - 1.96*cv_scores.std()):.3f}, "
      f"{np.sqrt(-cv_scores.mean() + 1.96*cv_scores.std()):.3f}]")
```

---

## Step 7: Final Model and Predictions

### Final Evaluation

```python
# Final model evaluation
def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive model evaluation"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - Final Evaluation:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    - Average prediction error: ${rmse*100000:.2f}")
    print(f"    - Model explains {r2*100:.1f}% of variance")
    
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

final_metrics = evaluate_model(y_test, y_test_pred_tuned, "Final Model")
```

### Make Predictions on New Data

```python
# Example: Predict for new house
new_house = {
    'MedInc': 8.0,           # Median income
    'HouseAge': 20.0,        # House age
    'AveRooms': 5.0,         # Average rooms
    'AveBedrms': 1.0,        # Average bedrooms
    'Population': 2000.0,    # Population
    'AveOccup': 3.0,         # Average occupancy
    'Latitude': 34.0,        # Latitude
    'Longitude': -118.0      # Longitude
}

# Convert to DataFrame
new_house_df = pd.DataFrame([new_house])

# Scale features
new_house_scaled = scaler.transform(new_house_df)

# Predict
predicted_value = best_tuned_model.predict(new_house_scaled)[0]

print(f"\nPrediction for New House:")
for key, value in new_house.items():
    print(f"  {key}: {value}")
print(f"\nPredicted Median House Value: ${predicted_value*100000:,.2f}")
```

### Save Model

```python
import joblib

# Save model and scaler
joblib.dump(best_tuned_model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and scaler saved successfully!")

# Load model (for future use)
# loaded_model = joblib.load('house_price_model.pkl')
# loaded_scaler = joblib.load('scaler.pkl')
```

---

## Complete Code Summary

```python
# Complete Regression Project Pipeline
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load and explore data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# 2. Prepare data
X = df[housing.feature_names]
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train and tune model
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train_scaled, y_train)

# 5. Evaluate
best_model = ridge_grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# 6. Save model
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## Key Takeaways

1. **Always explore data first** - Understand distributions and relationships
2. **Handle outliers appropriately** - Don't ignore them
3. **Check assumptions** - Use residual analysis
4. **Scale features** - Essential for regularization
5. **Tune hyperparameters** - Use cross-validation
6. **Evaluate comprehensively** - Multiple metrics and diagnostics
7. **Interpret results** - Understand what your model learned

---

**Congratulations!** You've built a complete regression model! 🎉

**Next Steps:**
- Try different feature engineering techniques
- Experiment with other regression algorithms
- Deploy your model as an API
- Move to [04-supervised-learning-classification](../04-supervised-learning-classification/README.md)

