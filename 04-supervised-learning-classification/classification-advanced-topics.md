# Advanced Classification Topics

Comprehensive guide to advanced classification techniques, handling imbalanced data, and best practices.

## Table of Contents

- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Feature Engineering for Classification](#feature-engineering-for-classification)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Interpretation](#model-interpretation)
- [Ensemble Methods for Classification](#ensemble-methods-for-classification)
- [Handling Missing Values](#handling-missing-values)
- [Common Classification Pitfalls](#common-classification-pitfalls)

---

## Handling Imbalanced Data

### The Problem

When classes are not equally represented, accuracy becomes misleading.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create imbalanced dataset (90% class 0, 10% class 1)
X, y = make_classification(
    n_samples=1000, 
    n_classes=2, 
    weights=[0.9, 0.1],
    random_state=42
)

print("Class distribution:")
print(pd.Series(y).value_counts())
print(f"\nBaseline (always predict majority): {max(pd.Series(y).value_counts()) / len(y):.3f}")

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClassification Report (imbalanced):")
print(classification_report(y_test, y_pred))
```

### Solution 1: Class Weights

Adjust weights to penalize misclassification of minority class.

```python
# Option 1: Balanced weights
model_balanced = LogisticRegression(
    class_weight='balanced',
    random_state=42
)
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)

print("\nWith balanced class weights:")
print(classification_report(y_test, y_pred_balanced))

# Option 2: Custom weights
model_custom = LogisticRegression(
    class_weight={0: 1, 1: 10},  # Penalize class 1 misclassification 10x more
    random_state=42
)
model_custom.fit(X_train, y_train)
y_pred_custom = model_custom.predict(X_test)
```

### Solution 2: Resampling

#### Oversampling (SMOTE)

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# SMOTE: Synthetic Minority Oversampling Technique
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_resampled).value_counts())

model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_resampled, y_resampled)
y_pred_smote = model_smote.predict(X_test)

print("\nWith SMOTE:")
print(classification_report(y_test, y_pred_smote))
```

#### Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)

print("After undersampling:")
print(pd.Series(y_under).value_counts())
```

#### Combined (SMOTE + Tomek Links)

```python
from imblearn.combine import SMOTETomek

# Combine oversampling and undersampling
smote_tomek = SMOTETomek(random_state=42)
X_combined, y_combined = smote_tomek.fit_resample(X_train, y_train)

print("After SMOTE + Tomek Links:")
print(pd.Series(y_combined).value_counts())
```

### Solution 3: Threshold Tuning

Adjust decision threshold instead of default 0.5.

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that maximizes F1-score
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")

# Use optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

print("\nWith optimal threshold:")
print(classification_report(y_test, y_pred_optimal))
```

### Comparison of Methods

```python
methods = {
    'Baseline': model,
    'Class Weights': model_balanced,
    'SMOTE': model_smote
}

results = {}
for name, model_method in methods.items():
    if name == 'SMOTE':
        y_pred_method = model_method.predict(X_test)
    else:
        y_pred_method = model_method.predict(X_test)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred_method)
    results[name] = f1
    print(f"{name:15s}: F1-Score = {f1:.3f}")
```

---

## Feature Engineering for Classification

### Creating Interaction Features

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_train)

print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
```

### Binning Continuous Features

```python
import pandas as pd

# Create bins for continuous features
df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
df['target'] = y_train

# Bin a feature
df['feature_0_binned'] = pd.cut(df['feature_0'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# One-hot encode bins
df_encoded = pd.get_dummies(df, columns=['feature_0_binned'])
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Method 1: Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train)

# Method 2: Recursive Feature Elimination
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train, y_train)

# Method 3: Feature importance from Random Forest
rf.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': range(X_train.shape[1]),
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 5 features:")
print(feature_importance.head())
```

---

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5, 
    scoring='f1',  # Use F1 for imbalanced data
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Random search
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,  # Number of iterations
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

### Bayesian Optimization

```python
# Using scikit-optimize (install: pip install scikit-optimize)
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# Define search space
space = [
    Integer(50, 300, name='n_estimators'),
    Integer(5, 30, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf')
]

# Objective function
@use_named_args(space=space)
def objective(**params):
    rf = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
    return -scores.mean()  # Minimize negative F1

# Optimize
result = gp_minimize(objective, space, n_calls=50, random_state=42)
print(f"Best parameters: {result.x}")
```

---

## Model Interpretation

### Feature Importance

```python
import matplotlib.pyplot as plt

# Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': [f'Feature {i}' for i in range(X_train.shape[1])],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### SHAP Values

```python
# Install: pip install shap
import shap

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed

# Summary plot
shap.summary_plot(shap_values, X_test[:100], plot_type="bar")
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42,
    scoring='f1'
)

# Visualize
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
plt.yticks(range(len(sorted_idx)), [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.show()
```

---

## Ensemble Methods for Classification

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    voting='soft'  # Use probabilities
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print("Voting Classifier:")
print(classification_report(y_test, y_pred_voting))
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier

# Stacking with meta-learner
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)

print("Stacking Classifier:")
print(classification_report(y_test, y_pred_stacking))
```

### Boosting (XGBoost, LightGBM)

```python
# XGBoost
try:
    import xgboost as xgb
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    print("XGBoost:")
    print(classification_report(y_test, y_pred_xgb))
except ImportError:
    print("Install XGBoost: pip install xgboost")

# LightGBM
try:
    import lightgbm as lgb
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    lgb_clf.fit(X_train, y_train)
    y_pred_lgb = lgb_clf.predict(X_test)
    print("LightGBM:")
    print(classification_report(y_test, y_pred_lgb))
except ImportError:
    print("Install LightGBM: pip install lightgbm")
```

---

## Handling Missing Values

### Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Strategy 1: Mean/Median/Mode imputation
imputer_mean = SimpleImputer(strategy='mean')
X_train_imputed = imputer_mean.fit_transform(X_train)
X_test_imputed = imputer_mean.transform(X_test)

# Strategy 2: KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train)
X_test_knn = knn_imputer.transform(X_test)

# Strategy 3: Indicator variable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create indicator for missing values
imputer = SimpleImputer(strategy='mean', add_indicator=True)
X_train_with_indicator = imputer.fit_transform(X_train)
```

---

## Common Classification Pitfalls

### Pitfall 1: Using Accuracy for Imbalanced Data

```python
# Bad: Using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")  # Misleading!

# Good: Use F1, Precision, Recall
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
```

### Pitfall 2: Data Leakage

```python
# Bad: Scaling before train-test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Good: Scale after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!
```

### Pitfall 3: Not Using Cross-Validation

```python
# Bad: Single train-test split
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Good: Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Pitfall 4: Ignoring Class Imbalance

```python
# Bad: Default model on imbalanced data
model = LogisticRegression()
model.fit(X_train, y_train)

# Good: Handle imbalance
model = LogisticRegression(class_weight='balanced')
# OR use resampling
# OR use appropriate metrics
```

---

## Key Takeaways

1. **Imbalanced Data**: Use class weights, resampling, or threshold tuning
2. **Feature Engineering**: Create meaningful features, select important ones
3. **Hyperparameter Tuning**: Use grid search, random search, or Bayesian optimization
4. **Model Interpretation**: Understand feature importance and predictions
5. **Ensemble Methods**: Combine models for better performance
6. **Avoid Pitfalls**: Don't use accuracy for imbalanced data, avoid data leakage

---

## Next Steps

- Practice with real imbalanced datasets
- Experiment with different resampling techniques
- Learn about model interpretability tools (SHAP, LIME)
- Move to ensemble methods module

**Remember**: Classification is more than just accuracy - understand your data and use appropriate metrics!

