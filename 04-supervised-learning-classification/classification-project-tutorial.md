# Complete Classification Project Tutorial

Step-by-step walkthrough of building a real-world classification model from data exploration to deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Exploration](#step-1-data-loading-and-exploration)
- [Step 2: Data Cleaning and Preprocessing](#step-2-data-cleaning-and-preprocessing)
- [Step 3: Feature Engineering](#step-3-feature-engineering)
- [Step 4: Model Training](#step-4-model-training)
- [Step 5: Model Evaluation](#step-5-model-evaluation)
- [Step 6: Handling Class Imbalance](#step-6-handling-class-imbalance)
- [Step 7: Model Improvement](#step-7-model-improvement)
- [Step 8: Final Model and Predictions](#step-8-final-model-and-predictions)

---

## Project Overview

**Project**: Predict Customer Churn

**Dataset**: Customer churn dataset (or any binary classification dataset)

**Goal**: Build a classification model to predict if customers will churn

**Type**: Binary Classification

**Challenge**: Class imbalance (most customers don't churn)

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Load data (example with synthetic data - replace with your dataset)
from sklearn.datasets import make_classification

# For tutorial, we'll create a realistic churn dataset
np.random.seed(42)
n_samples = 10000

# Create features
df = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'tenure': np.random.randint(0, 60, n_samples),
    'monthly_charges': np.random.uniform(20, 100, n_samples),
    'total_charges': np.random.uniform(0, 5000, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples)
})

# Create target (churn) with some logic
churn_prob = (df['tenure'] < 12).astype(int) * 0.3 + \
             (df['monthly_charges'] > 70).astype(int) * 0.2 + \
             (df['contract_type'] == 'Month-to-month').astype(int) * 0.3 + \
             np.random.random(n_samples) * 0.2

df['churn'] = (churn_prob > 0.5).astype(int)

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

print("\nTarget Variable (Churn) Distribution:")
print(df['churn'].value_counts())
print(f"\nChurn Rate: {df['churn'].mean()*100:.1f}%")
```

### Visualizations

```python
# Class distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['churn'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)

plt.subplot(1, 2, 2)
df['churn'].value_counts(normalize=True).plot(kind='bar', color=['green', 'red'])
plt.xlabel('Churn')
plt.ylabel('Proportion')
plt.title('Class Distribution (Proportions)')
plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)

plt.tight_layout()
plt.show()

# Feature distributions by churn
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features_to_plot = ['age', 'tenure', 'monthly_charges', 'total_charges']
for idx, feature in enumerate(features_to_plot):
    row = idx // 2
    col = idx % 2
    
    df[df['churn'] == 0][feature].hist(alpha=0.5, label='No Churn', ax=axes[row, col], bins=30)
    df[df['churn'] == 1][feature].hist(alpha=0.5, label='Churn', ax=axes[row, col], bins=30)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'{feature} by Churn Status')
    axes[row, col].legend()

plt.tight_layout()
plt.show()

# Correlation with target
numeric_features = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_features].corr()['churn'].sort_values(ascending=False)
print("\nCorrelation with Churn:")
print(correlation)
```

---

## Step 2: Data Cleaning and Preprocessing

### Handle Missing Values

```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# If there are missing values, handle them
# For numeric: fill with median
# For categorical: fill with mode
# Or drop if too many missing

# Example:
# df['column'] = df['column'].fillna(df['column'].median())
```

### Encode Categorical Variables

```python
# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

# Option 1: Label Encoding (for tree-based models)
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Option 2: One-Hot Encoding (for linear models)
# X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nAfter encoding:")
print(X_encoded.head())
print(f"\nShape: {X_encoded.shape}")
```

### Split Data

```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining class distribution:")
print(y_train.value_counts())
print(f"\nTest class distribution:")
print(y_test.value_counts())
```

### Scale Features

```python
# Scale features (important for logistic regression, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Features scaled successfully!")
```

---

## Step 3: Feature Engineering

### Create New Features

```python
# Example: Create interaction features
X_train_fe = X_train_scaled.copy()
X_test_fe = X_test_scaled.copy()

# Example features (adjust based on your domain)
# X_train_fe['age_tenure_ratio'] = X_train['age'] / (X_train['tenure'] + 1)
# X_train_fe['charges_ratio'] = X_train['total_charges'] / (X_train['monthly_charges'] + 1)

print(f"Features after engineering: {X_train_fe.shape[1]}")
```

### Feature Selection (Optional)

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top k features
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_fe, y_train)
X_test_selected = selector.transform(X_test_fe)

selected_features = X_train_fe.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Use selected features
X_train_final = X_train_fe[selected_features]
X_test_final = X_test_fe[selected_features]
```

---

## Step 4: Model Training

### Baseline Model (Logistic Regression)

```python
# Train baseline model
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = baseline_model.predict(X_train_scaled)
y_test_pred = baseline_model.predict(X_test_scaled)
y_test_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Baseline Model (Logistic Regression):")
print(f"  Training Accuracy: {train_accuracy:.3f}")
print(f"  Test Accuracy: {test_accuracy:.3f}")
print(f"  Test Precision: {precision_score(y_test, y_test_pred):.3f}")
print(f"  Test Recall: {recall_score(y_test, y_test_pred):.3f}")
print(f"  Test F1-Score: {f1_score(y_test, y_test_pred):.3f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
```

### Random Forest

```python
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_test_pred_rf = rf_model.predict(X_test_scaled)
y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nRandom Forest:")
print(f"  Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.3f}")
print(f"  Test Precision: {precision_score(y_test, y_test_pred_rf):.3f}")
print(f"  Test Recall: {recall_score(y_test, y_test_pred_rf):.3f}")
print(f"  Test F1-Score: {f1_score(y_test, y_test_pred_rf):.3f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_test_proba_rf):.3f}")
```

---

## Step 5: Model Evaluation

### Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrix for baseline
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Baseline Model')

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_test_pred_rf)

plt.subplot(1, 2, 2)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Churn', 'Churn'])
disp_rf.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest')

plt.tight_layout()
plt.show()

# Print confusion matrices
print("Baseline Model Confusion Matrix:")
print(cm)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)
```

### Classification Report

```python
print("Baseline Model - Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))

print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, y_test_pred_rf, target_names=['No Churn', 'Churn']))
```

### ROC Curves

```python
# Calculate ROC curves
fpr_base, tpr_base, _ = roc_curve(y_test, y_test_proba)
roc_auc_base = roc_auc_score(y_test, y_test_proba)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
roc_auc_rf = roc_auc_score(y_test, y_test_proba_rf)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {roc_auc_base:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()
```

### Precision-Recall Curves

```python
# Calculate PR curves
precision_base, recall_base, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc_base = average_precision_score(y_test, y_test_proba)

precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_test_proba_rf)
pr_auc_rf = average_precision_score(y_test, y_test_proba_rf)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall_base, precision_base, label=f'Baseline (AP = {pr_auc_base:.3f})')
plt.plot(recall_rf, precision_rf, label=f'Random Forest (AP = {pr_auc_rf:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Step 6: Handling Class Imbalance

### Check Imbalance

```python
print("Class Distribution:")
print(f"  Training: {y_train.value_counts().to_dict()}")
print(f"  Test: {y_test.value_counts().to_dict()}")

imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1")
```

### Strategy 1: Class Weights

```python
# Use balanced class weights
balanced_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
balanced_model.fit(X_train_scaled, y_train)

y_test_pred_balanced = balanced_model.predict(X_test_scaled)
y_test_proba_balanced = balanced_model.predict_proba(X_test_scaled)[:, 1]

print("Balanced Model:")
print(f"  Test F1-Score: {f1_score(y_test, y_test_pred_balanced):.3f}")
print(f"  Test Recall: {recall_score(y_test, y_test_pred_balanced):.3f}")
print(f"  Test Precision: {precision_score(y_test, y_test_pred_balanced):.3f}")
```

### Strategy 2: SMOTE

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE:")
print(f"  Original training size: {len(y_train)}")
print(f"  Resampled training size: {len(y_train_smote)}")
print(f"  Class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train on resampled data
smote_model = LogisticRegression(random_state=42, max_iter=1000)
smote_model.fit(X_train_smote, y_train_smote)

y_test_pred_smote = smote_model.predict(X_test_scaled)
y_test_proba_smote = smote_model.predict_proba(X_test_scaled)[:, 1]

print("\nSMOTE Model:")
print(f"  Test F1-Score: {f1_score(y_test, y_test_pred_smote):.3f}")
print(f"  Test Recall: {recall_score(y_test, y_test_pred_smote):.3f}")
print(f"  Test Precision: {precision_score(y_test, y_test_pred_smote):.3f}")
```

### Compare Strategies

```python
strategies = {
    'Baseline': (y_test_pred, y_test_proba),
    'Class Weights': (y_test_pred_balanced, y_test_proba_balanced),
    'SMOTE': (y_test_pred_smote, y_test_proba_smote)
}

results = []
for name, (y_pred, y_proba) in strategies.items():
    results.append({
        'Strategy': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results)
print("\nStrategy Comparison:")
print(results_df.to_string(index=False))
```

---

## Step 7: Model Improvement

### Threshold Tuning

```python
# Find optimal threshold for best model
best_model = smote_model  # Use SMOTE model
y_proba = smote_model.predict_proba(X_test_scaled)[:, 1]

# Calculate F1 at different thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

# Find best threshold
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best F1-Score: {best_f1:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold = {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Use optimal threshold
y_pred_optimal = (y_proba >= best_threshold).astype(int)
print(f"\nWith Optimal Threshold:")
print(f"  F1-Score: {f1_score(y_test, y_pred_optimal):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_optimal):.3f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Tune Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

rf_grid.fit(X_train_smote, y_train_smote)

print("Best Random Forest Parameters:")
print(rf_grid.best_params_)
print(f"Best CV F1-Score: {rf_grid.best_score_:.3f}")

# Evaluate best model
best_rf = rf_grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_scaled)
y_proba_best_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

print(f"\nBest Random Forest:")
print(f"  Test F1-Score: {f1_score(y_test, y_pred_best_rf):.3f}")
print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_proba_best_rf):.3f}")
```

---

## Step 8: Final Model and Predictions

### Final Evaluation

```python
# Choose best model
final_model = best_rf  # Or choose based on your criteria
final_predictions = y_pred_best_rf
final_probabilities = y_proba_best_rf

print("Final Model Evaluation:")
print(f"  Accuracy: {accuracy_score(y_test, final_predictions):.3f}")
print(f"  Precision: {precision_score(y_test, final_predictions):.3f}")
print(f"  Recall: {recall_score(y_test, final_predictions):.3f}")
print(f"  F1-Score: {f1_score(y_test, final_predictions):.3f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, final_probabilities):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, final_predictions, target_names=['No Churn', 'Churn']))
```

### Feature Importance

```python
# Feature importance (for tree-based models)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
```

### Make Predictions on New Data

```python
# Example: Predict for new customer
new_customer = {
    'age': 45,
    'tenure': 24,
    'monthly_charges': 65.0,
    'total_charges': 1560.0,
    'contract_type': 'One year',
    'payment_method': 'Electronic',
    'internet_service': 'Fiber optic'
}

# Convert to DataFrame
new_customer_df = pd.DataFrame([new_customer])

# Encode categorical variables (use same encoders)
for col in categorical_cols:
    if col in new_customer_df.columns:
        le = label_encoders[col]
        new_customer_df[col] = le.transform(new_customer_df[col])

# Scale features
new_customer_scaled = scaler.transform(new_customer_df)

# Predict
churn_probability = final_model.predict_proba(new_customer_scaled)[0, 1]
churn_prediction = final_model.predict(new_customer_scaled)[0]

print(f"\nPrediction for New Customer:")
for key, value in new_customer.items():
    print(f"  {key}: {value}")
print(f"\nChurn Probability: {churn_probability:.3f}")
print(f"Predicted Churn: {'Yes' if churn_prediction == 1 else 'No'}")
```

### Save Model

```python
import joblib

# Save model, scaler, and encoders
joblib.dump(final_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\nModel and preprocessing objects saved successfully!")

# Load model (for future use)
# loaded_model = joblib.load('churn_model.pkl')
# loaded_scaler = joblib.load('scaler.pkl')
# loaded_encoders = joblib.load('label_encoders.pkl')
```

---

## Complete Code Summary

```python
# Complete Classification Project Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load and explore data
df = pd.read_csv('churn_data.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# 2. Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Handle imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_smote, y_train_smote)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 8. Save
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
```

---

## Key Takeaways

1. **Always explore data first** - Understand class distribution and features
2. **Handle class imbalance** - Use appropriate strategies (SMOTE, class weights)
3. **Use multiple metrics** - Don't rely only on accuracy
4. **Tune threshold** - Find optimal threshold for your problem
5. **Evaluate comprehensively** - Use ROC, PR curves, confusion matrix
6. **Interpret results** - Understand feature importance and predictions

---

**Congratulations!** You've built a complete classification model!

**Next Steps:**
- Try different algorithms (SVM, XGBoost)
- Experiment with feature engineering
- Deploy your model as an API
- Move to [05-model-evaluation-optimization](../05-model-evaluation-optimization/README.md)


