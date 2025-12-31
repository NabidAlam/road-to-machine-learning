# Complete Classification Project Tutorial

Step-by-step walkthrough of building a real-world classification model from data exploration to deployment.

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

**Project**: Spam Email Detection

**Dataset**: SMS Spam Collection Dataset (or any text classification dataset)

**Goal**: Build a classification model to detect spam emails/messages

**Type**: Binary Classification (Spam/Not Spam)

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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data (example with SMS spam dataset)
# Download from: https://www.kaggle.com/uciml/sms-spam-collection-dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Alternative: Use built-in dataset or create synthetic data
# For this tutorial, we'll create a synthetic dataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create synthetic text-like features for demonstration
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.8, 0.2],  # Imbalanced: 80% ham, 20% spam
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y
df['target'] = df['target'].map({0: 'ham', 1: 'spam'})

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
print(df.isnull().sum().sum())

print("\nTarget Variable Distribution:")
print(df['target'].value_counts())
print(f"\nClass Distribution (%):")
print(df['target'].value_counts(normalize=True) * 100)
```

### Visualizations

```python
# Class distribution
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Feature distributions by class
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, feature in enumerate(df.columns[:4]):
    row = idx // 2
    col = idx % 2
    df.boxplot(column=feature, by='target', ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Class')
    axes[row, col].set_xlabel('')
plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation = df.iloc[:, :-1].corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Step 2: Data Cleaning and Preprocessing

### Handle Missing Values

```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# If there are missing values, handle them
if df.isnull().sum().sum() > 0:
    # Option 1: Drop rows with missing values
    df = df.dropna()
    
    # Option 2: Fill with median/mean
    # df = df.fillna(df.median())
    
    # Option 3: Use imputation
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy='median')
    # df[df.columns[:-1]] = imputer.fit_transform(df[df.columns[:-1]])
```

### Encode Target Variable

```python
# Encode target variable
df['target_encoded'] = df['target'].map({'ham': 0, 'spam': 1})

# Verify encoding
print("Target encoding:")
print(df[['target', 'target_encoded']].head(10))
```

### Split Features and Target

```python
# Separate features and target
X = df.drop(['target', 'target_encoded'], axis=1)
y = df['target_encoded']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")
```

### Train-Test Split

```python
# Split data (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Important for imbalanced data!
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining class distribution:")
print(y_train.value_counts(normalize=True))
print(f"\nTest class distribution:")
print(y_test.value_counts(normalize=True))
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Scale features (important for SVM, Logistic Regression, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!

print("Features scaled successfully!")
print(f"Scaled training shape: {X_train_scaled.shape}")
```

---

## Step 3: Feature Engineering

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top k features
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features:")
print(selected_features.tolist())
```

### Create New Features (if applicable)

```python
# Example: Create interaction features
# For this dataset, we'll skip this step
# But in real projects, you might create:
# - Ratio features
# - Polynomial features
# - Domain-specific features
```

---

## Step 4: Model Training

### Train Multiple Models

```python
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print('='*50)
    
    # Use scaled data for Logistic Regression and SVM
    if name in ['Logistic Regression', 'SVM']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    # Train
    model.fit(X_train_model, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_model)
    y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
```

### Cross-Validation

```python
# Cross-validation for more robust evaluation
cv_results = {}

for name, model in models.items():
    if name in ['Logistic Regression', 'SVM']:
        X_model = X_train_scaled
    else:
        X_model = X_train
    
    scores = cross_val_score(
        model, X_model, y_train, 
        cv=5, 
        scoring='f1',  # Use F1 for imbalanced data
        n_jobs=-1
    )
    
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Step 5: Model Evaluation

### Confusion Matrix

```python
# Get best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print(f"\nConfusion Matrix for {best_model_name}:")
print(cm)
```

### ROC Curve

```python
# ROC curve for binary classification
if results[best_model_name]['probabilities'] is not None:
    y_pred_proba = results[best_model_name]['probabilities']
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"AUC: {roc_auc:.3f}")
```

### Model Comparison

```python
# Compare all models
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'CV F1 Mean': [cv_results[m]['mean'] for m in results.keys()],
    'CV F1 Std': [cv_results[m]['std'] for m in results.keys()]
}).sort_values('Accuracy', ascending=False)

print("\nModel Comparison:")
print(comparison)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
axes[0].barh(comparison['Model'], comparison['Accuracy'], color='skyblue')
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# CV F1 comparison
axes[1].barh(comparison['Model'], comparison['CV F1 Mean'], 
             xerr=comparison['CV F1 Std'], color='salmon', capsize=5)
axes[1].set_xlabel('Cross-Validated F1 Score', fontsize=12)
axes[1].set_title('Model F1 Score (CV)', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Step 6: Model Improvement

### Hyperparameter Tuning

```python
# Tune Random Forest (best performing model in this example)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Evaluate best model
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
print(f"\nTuned model accuracy: {accuracy_score(y_test, y_pred_tuned):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))
```

### Handle Class Imbalance

```python
# Try with class weights
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print("With class weights:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))
```

### Feature Importance

```python
# Feature importance from best model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

## Step 7: Final Model and Predictions

### Select Final Model

```python
# Choose best model (tuned Random Forest in this case)
final_model = best_rf

# Or use the best from initial comparison
# final_model = best_model

print("Final Model Selected: Random Forest (Tuned)")
```

### Make Predictions on New Data

```python
# Example: Predict on new samples
new_samples = X_test[:5]  # Use first 5 test samples as example

predictions = final_model.predict(new_samples)
probabilities = final_model.predict_proba(new_samples)

print("Predictions on New Samples:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    class_name = 'Spam' if pred == 1 else 'Ham'
    spam_prob = prob[1]
    print(f"Sample {i+1}: {class_name} (Spam probability: {spam_prob:.3f})")
```

### Save Model

```python
import joblib

# Save model and scaler
joblib.dump(final_model, 'spam_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

print("Model saved successfully!")
```

### Load and Use Model

```python
# Load model
loaded_model = joblib.load('spam_classifier_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_selector = joblib.load('feature_selector.pkl')

# Use for prediction
def predict_spam(features):
    """Predict if message is spam"""
    # Scale features
    features_scaled = loaded_scaler.transform([features])
    # Select features
    features_selected = loaded_selector.transform(features_scaled)
    # Predict
    prediction = loaded_model.predict(features_selected)[0]
    probability = loaded_model.predict_proba(features_selected)[0]
    
    return {
        'prediction': 'Spam' if prediction == 1 else 'Ham',
        'spam_probability': probability[1]
    }

# Example usage
example_features = X_test.iloc[0].values
result = predict_spam(example_features)
print(f"Prediction: {result['prediction']}")
print(f"Spam Probability: {result['spam_probability']:.3f}")
```

---

## Summary

### What We Learned

1. **Data Exploration**: Understanding class distribution and feature relationships
2. **Preprocessing**: Handling missing values, encoding, scaling
3. **Feature Engineering**: Feature selection and creation
4. **Model Training**: Training multiple algorithms and comparing
5. **Evaluation**: Using appropriate metrics for imbalanced data
6. **Improvement**: Hyperparameter tuning and handling class imbalance
7. **Deployment**: Saving and loading models for production use

### Key Takeaways

- Always check class distribution (imbalanced data is common!)
- Use stratified train-test split for imbalanced data
- Scale features for distance-based algorithms
- Use F1-score or AUC for imbalanced data evaluation
- Try multiple algorithms and compare
- Tune hyperparameters for better performance
- Save models for future use

### Next Steps

- Try with real text classification datasets
- Experiment with different feature engineering techniques
- Deploy model as a web API (Flask/FastAPI)
- Add model monitoring and retraining pipeline

---

**Congratulations!** You've built a complete classification model from scratch!

