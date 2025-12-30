"""
House Price Prediction Project
Predict house prices using California Housing dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X, y = housing.data
feature_names = housing.feature_names

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print(f"\nDataset shape: {df.shape}")
print(f"\nFeatures: {', '.join(feature_names)}")
print(f"\nFirst few rows:")
print(df.head())

# Basic statistics
print("\n" + "="*50)
print("Dataset Statistics")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("Missing Values")
print("="*50)
print(df.isnull().sum())

# Visualize target distribution
plt.figure(figsize=(10, 6))
plt.hist(df['MedHouseVal'], bins=50, edgecolor='black')
plt.xlabel('Median House Value (in $100,000s)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = {}

print("\n" + "="*50)
print("Model Training and Evaluation")
print("="*50)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }
    
    print(f"\n{name}:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Test MAE:   {test_mae:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")

# Compare models
print("\n" + "="*50)
print("Model Comparison")
print("="*50)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()]
})
print(comparison_df.to_string(index=False))

# Visualize predictions vs actual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    axes[idx].scatter(y_test, result['predictions'], alpha=0.5)
    axes[idx].plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[idx].set_xlabel('Actual Prices')
    axes[idx].set_ylabel('Predicted Prices')
    axes[idx].set_title(f'{name}\nR² = {result["test_r2"]:.3f}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance (for Linear Regression)
if 'Linear Regression' in results:
    lr_model = models['Linear Regression']
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\n" + "="*50)
    print("Feature Importance (Linear Regression)")
    print("="*50)
    print(feature_importance.to_string(index=False))

print("\n" + "="*50)
print("Project Complete!")
print("="*50)
print("\nNext steps:")
print("1. Try hyperparameter tuning")
print("2. Experiment with feature engineering")
print("3. Try the Kaggle House Prices competition")

