# Classification Quick Reference Guide

Quick reference for classification algorithms, metrics, and best practices.

## Table of Contents

- [Algorithm Selection](#algorithm-selection)
- [Code Snippets](#code-snippets)
- [Evaluation Metrics](#evaluation-metrics)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Algorithm Selection

### Quick Decision Tree

```
Need to classify into categories?
│
├─ Need interpretability?
│  ├─ YES → Logistic Regression or Decision Tree
│  └─ NO → Continue
│
├─ Small dataset (< 10K samples)?
│  ├─ YES → SVM or KNN
│  └─ NO → Random Forest or XGBoost
│
├─ Need probabilities?
│  ├─ YES → Logistic Regression or Random Forest
│  └─ NO → SVM or Decision Tree
│
├─ Imbalanced data?
│  ├─ YES → Use class weights or resampling
│  └─ NO → Continue
│
└─ Non-linear relationships?
   ├─ YES → Random Forest, SVM, or KNN
   └─ NO → Logistic Regression
```

### Algorithm Comparison

| Algorithm | When to Use | Pros | Cons | Code |
|-----------|-------------|------|------|------|
| **Logistic Regression** | Linear relationships, interpretability | Fast, interpretable, probabilities | Assumes linearity | `LogisticRegression()` |
| **Decision Trees** | Non-linear, interpretability needed | Interpretable, no scaling needed | Prone to overfitting | `DecisionTreeClassifier()` |
| **Random Forests** | General purpose, robust | Handles overfitting, feature importance | Less interpretable, slower | `RandomForestClassifier()` |
| **SVM** | Complex boundaries, small datasets | Effective for non-linear, good generalization | Slow for large datasets, memory intensive | `SVC()` |
| **KNN** | Non-linear, small datasets | Simple, no assumptions | Slow for large datasets, sensitive to scale | `KNeighborsClassifier()` |
| **XGBoost** | Large datasets, high performance | Very accurate, handles missing values | Complex, many hyperparameters | `XGBClassifier()` |
| **Naive Bayes** | Text classification, small datasets | Fast, works well with few samples | Assumes feature independence | `GaussianNB()` |

---

## Code Snippets

### Basic Classification Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (for Logistic Regression, SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Basic
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# With class weights (for imbalanced data)
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Multiclass
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42
)
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier

# Basic
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# With hyperparameters
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',  # or 'entropy'
    random_state=42
)

# Visualize tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=feature_names)
plt.show()
```

### Random Forests

```python
from sklearn.ensemble import RandomForestClassifier

# Basic
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# With hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',  # For imbalanced data
    random_state=42
)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### SVM

```python
from sklearn.svm import SVC

# Linear SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)

# RBF kernel (most common)
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Enable predict_proba
    random_state=42
)

# Polynomial kernel
svm = SVC(kernel='poly', degree=3, random_state=42)
```

### KNN

```python
from sklearn.neighbors import KNeighborsClassifier

# Basic
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# With hyperparameters
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # or 'uniform'
    metric='euclidean',  # or 'manhattan', 'minkowski'
    algorithm='auto'
)
```

### XGBoost

```python
import xgboost as xgb

# Basic
xgb_clf = xgb.XGBClassifier(random_state=42)
xgb_clf.fit(X_train, y_train)

# With hyperparameters
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## Evaluation Metrics

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Warning: Misleading for imbalanced data!
```

### Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Binary classification
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Multiclass (weighted average)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
```

### ROC-AUC (Binary Only)

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))
```

### Complete Evaluation Function

```python
def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """Comprehensive classification evaluation"""
    from sklearn.metrics import (accuracy_score, precision_score, 
                                recall_score, f1_score, roc_auc_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics
```

---

## Common Issues & Solutions

### Issue 1: Imbalanced Data

**Problem**: Model always predicts majority class

**Solutions**:
```python
# Solution 1: Class weights
model = LogisticRegression(class_weight='balanced')

# Solution 2: SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Solution 3: Use appropriate metrics
# Don't use accuracy! Use F1-score or AUC
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
```

### Issue 2: Low Accuracy

**Problem**: Model performs poorly

**Solutions**:
```python
# Solution 1: Try different algorithms
models = [LogisticRegression(), RandomForestClassifier(), SVC()]

# Solution 2: Feature engineering
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Solution 3: Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Issue 3: Overfitting

**Problem**: High training accuracy, low test accuracy

**Solutions**:
```python
# Solution 1: Regularization
model = LogisticRegression(C=0.1)  # Lower C = more regularization

# Solution 2: Reduce model complexity
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10)

# Solution 3: Use ensemble methods
rf = RandomForestClassifier(n_estimators=100, max_depth=10)

# Solution 4: Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
```

### Issue 4: Slow Training

**Problem**: Model takes too long to train

**Solutions**:
```python
# Solution 1: Reduce dataset size
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

# Solution 2: Use faster algorithms
# Logistic Regression > Random Forest > SVM

# Solution 3: Reduce features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=50)
X_selected = selector.fit_transform(X_train, y_train)

# Solution 4: Parallel processing
model = RandomForestClassifier(n_jobs=-1)  # Use all CPUs
```

### Issue 5: Memory Error

**Problem**: Out of memory when training

**Solutions**:
```python
# Solution 1: Use batch processing
# Process data in chunks

# Solution 2: Use memory-efficient algorithms
# Logistic Regression uses less memory than Random Forest

# Solution 3: Reduce data size
# Sample data or use feature selection
```

---

## Best Practices Checklist

### Data Preparation
- [ ] Check for missing values
- [ ] Handle missing values appropriately
- [ ] Encode categorical variables
- [ ] Scale features (for Logistic Regression, SVM, KNN)
- [ ] Check class distribution (imbalanced data?)
- [ ] Use stratified train-test split

### Model Selection
- [ ] Start with simple baseline (Logistic Regression)
- [ ] Try multiple algorithms
- [ ] Compare models using cross-validation
- [ ] Choose model based on problem requirements

### Evaluation
- [ ] Use appropriate metrics (F1, AUC for imbalanced data)
- [ ] Don't rely only on accuracy
- [ ] Check confusion matrix
- [ ] Use cross-validation for robust evaluation
- [ ] Plot ROC curve for binary classification

### Improvement
- [ ] Handle class imbalance if present
- [ ] Perform feature engineering
- [ ] Tune hyperparameters
- [ ] Try ensemble methods
- [ ] Regularize to prevent overfitting

### Deployment
- [ ] Save model and preprocessing steps
- [ ] Document model performance
- [ ] Create prediction function
- [ ] Test on new data
- [ ] Monitor model performance

---

## Quick Tips

1. **Always use stratified split** for imbalanced data
2. **Scale features** for distance-based algorithms (SVM, KNN, Logistic Regression)
3. **Use F1-score or AUC** instead of accuracy for imbalanced data
4. **Start simple** - Logistic Regression is a great baseline
5. **Cross-validate** for robust evaluation
6. **Visualize** - Confusion matrix, ROC curve, feature importance
7. **Handle imbalance** - Class weights, SMOTE, or threshold tuning
8. **Tune hyperparameters** - Grid search or random search
9. **Try ensembles** - Random Forest, XGBoost often perform better
10. **Save everything** - Model, scaler, feature selector

---

## Common Mistakes to Avoid

1. ❌ Using accuracy for imbalanced data
2. ❌ Not scaling features for SVM/KNN
3. ❌ Data leakage (scaling before split)
4. ❌ Overfitting (too complex model)
5. ❌ Not using cross-validation
6. ❌ Ignoring class imbalance
7. ❌ Not trying multiple algorithms
8. ❌ Using test set for hyperparameter tuning
9. ❌ Not saving preprocessing steps
10. ❌ Not checking confusion matrix

---

## Resources

- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#classification)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)

---

**Remember**: Classification is about more than accuracy - understand your data, use appropriate metrics, and always validate your results!

