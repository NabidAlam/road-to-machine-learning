"""
Wine Quality Prediction Project
Predict wine quality based on chemical properties
Can be approached as both regression and classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("Wine Quality Prediction Project")
print("="*60)

# Note: Download from UCI ML Repository
# https://archive.ics.uci.edu/ml/datasets/wine+quality
print("\nNote: Download wine datasets from UCI ML Repository")
print("Place red and white wine CSV files in the data/ directory.\n")

# Try to load data
try:
    df_red = pd.read_csv('data/winequality-red.csv', sep=';')
    df_white = pd.read_csv('data/winequality-white.csv', sep=';')
    df_red['type'] = 'red'
    df_white['type'] = 'white'
    df = pd.concat([df_red, df_white], ignore_index=True)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Creating sample data for demonstration...")
    # Create sample wine data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'fixed acidity': np.random.normal(7, 1.5, n_samples),
        'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
        'citric acid': np.random.normal(0.3, 0.15, n_samples),
        'residual sugar': np.random.exponential(3, n_samples),
        'chlorides': np.random.normal(0.05, 0.02, n_samples),
        'free sulfur dioxide': np.random.exponential(15, n_samples),
        'total sulfur dioxide': np.random.exponential(40, n_samples),
        'density': np.random.normal(0.997, 0.002, n_samples),
        'pH': np.random.normal(3.3, 0.15, n_samples),
        'sulphates': np.random.normal(0.65, 0.15, n_samples),
        'alcohol': np.random.normal(10.5, 1.2, n_samples),
        'quality': np.random.choice(range(3, 9), n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.15])
    })
    df['type'] = np.random.choice(['red', 'white'], n_samples)
    print("Using sample data for demonstration purposes.")

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print("\n" + "="*60)
print("Missing Values")
print("="*60)
print(df.isnull().sum().sum(), "missing values")

# Quality distribution
print("\n" + "="*60)
print("Quality Distribution")
print("="*60)
print(df['quality'].value_counts().sort_index())

# Basic statistics
print("\n" + "="*60)
print("Dataset Statistics")
print("="*60)
print(df.describe())

# Correlation analysis
print("\n" + "="*60)
print("Feature Correlation with Quality")
print("="*60)
correlations = df.corr()['quality'].sort_values(ascending=False)
print(correlations)

# Visualize correlation
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare data for regression
feature_cols = [col for col in df.columns if col not in ['quality', 'type']]
X = df[feature_cols]
y_regression = df['quality']

# Prepare data for classification (good wine >= 7, bad < 7)
y_classification = (df['quality'] >= 7).astype(int)
print(f"\nClassification target distribution:")
print(f"Bad wine (< 7): {(y_classification == 0).sum()}")
print(f"Good wine (>= 7): {(y_classification == 1).sum()}")

# Split data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ========== REGRESSION APPROACH ==========
print("\n" + "="*60)
print("REGRESSION APPROACH")
print("="*60)

regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

regression_results = {}

for name, model in regression_models.items():
    model.fit(X_train_scaled, y_train_reg)
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    
    regression_results[name] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

# Compare regression models
print("\n" + "="*60)
print("Regression Models Comparison")
print("="*60)
reg_comparison = pd.DataFrame({
    'Model': list(regression_results.keys()),
    'RMSE': [regression_results[m]['rmse'] for m in regression_results.keys()],
    'MAE': [regression_results[m]['mae'] for m in regression_results.keys()],
    'R²': [regression_results[m]['r2'] for m in regression_results.keys()]
}).sort_values('RMSE')
print(reg_comparison.to_string(index=False))

# ========== CLASSIFICATION APPROACH ==========
print("\n" + "="*60)
print("CLASSIFICATION APPROACH")
print("="*60)

classification_models = {
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

classification_results = {}

for name, model in classification_models.items():
    model.fit(X_train_scaled, y_train_clf)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test_clf, y_pred)
    
    classification_results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_clf, y_pred, target_names=['Bad Wine', 'Good Wine']))

# Feature importance
if 'Random Forest Regressor' in regression_models:
    rf_reg = regression_models['Random Forest Regressor']
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_reg.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*60)
    print("Feature Importance (Random Forest)")
    print("="*60)
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest Regressor')
    plt.tight_layout()
    plt.show()

# Visualize predictions vs actual (regression)
best_reg_model = min(regression_results, key=lambda x: regression_results[x]['rmse'])
best_reg_predictions = regression_results[best_reg_model]['predictions']

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, best_reg_predictions, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title(f'Regression: Actual vs Predicted ({best_reg_model})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Project Complete!")
print("="*60)
print(f"\nBest Regression Model: {best_reg_model}")
print(f"RMSE: {regression_results[best_reg_model]['rmse']:.4f}")
print(f"\nBest Classification Model: {list(classification_results.keys())[0]}")
print(f"Accuracy: {list(classification_results.values())[0]['accuracy']:.4f}")
print("\nNext steps:")
print("1. Download actual wine quality dataset from UCI")
print("2. Try advanced feature engineering")
print("3. Experiment with ensemble methods")
print("4. Compare red vs white wine models")

