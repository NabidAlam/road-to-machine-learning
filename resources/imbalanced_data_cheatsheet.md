# Imbalanced Data Cheatsheet

Comprehensive quick reference for handling imbalanced datasets in machine learning.

## Table of Contents

- [Understanding Imbalanced Data](#understanding-imbalanced-data)
- [Resampling Techniques](#resampling-techniques)
- [Algorithm-Level Solutions](#algorithm-level-solutions)
- [Evaluation Metrics](#evaluation-metrics)
- [Best Practices](#best-practices)

---

## Understanding Imbalanced Data

### Problem Definition

**Imbalanced Data**: When classes are not equally represented in the dataset.

**Example**:
- Fraud detection: 99% legitimate, 1% fraud
- Medical diagnosis: 95% healthy, 5% disease
- Spam detection: 90% ham, 10% spam

### Why It's a Problem

```python
# Naive baseline: Always predict majority class
# Accuracy: 99% (but useless!)

# Example
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 9:1 ratio
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Always predict 0
accuracy_score(y_true, y_pred)  # 0.90 (misleading!)
```

---

## Resampling Techniques

### Oversampling

#### SMOTE (Synthetic Minority Oversampling Technique)

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# With custom sampling strategy
smote = SMOTE(
    sampling_strategy=0.5,  # Balance to 50:50
    random_state=42,
    k_neighbors=5
)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### ADASYN (Adaptive Synthetic Sampling)

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

#### Random Oversampling

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

### Undersampling

#### Random Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

#### Tomek Links

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)
```

#### Edited Nearest Neighbours

```python
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours()
X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
```

### Combined Methods

#### SMOTE + Tomek Links

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

#### SMOTE + ENN

```python
from imblearn.combine import SMOTEENN

sme = SMOTEENN(random_state=42)
X_resampled, y_resampled = sme.fit_resample(X_train, y_train)
```

---

## Algorithm-Level Solutions

### Class Weights

#### Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier

# Automatic balancing
model = RandomForestClassifier(class_weight='balanced')

# Custom weights
model = RandomForestClassifier(
    class_weight={0: 1, 1: 10}  # Give class 1 10x weight
)

# Calculate weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
```

#### XGBoost

```python
import xgboost as xgb

# Scale positive weight
model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
)
```

### Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Use optimal threshold
y_pred = (y_proba >= optimal_threshold).astype(int)
```

### Cost-Sensitive Learning

```python
from sklearn.ensemble import RandomForestClassifier

# Define cost matrix
cost_matrix = {
    (0, 0): 0,    # True negative: no cost
    (0, 1): 10,   # False positive: low cost
    (1, 0): 100,  # False negative: high cost
    (1, 1): 0     # True positive: no cost
}

# Use class weights based on cost
model = RandomForestClassifier(
    class_weight={0: 1, 1: 10}  # Penalize missing class 1 more
)
```

---

## Evaluation Metrics

### Appropriate Metrics

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# Don't use accuracy!
# accuracy = accuracy_score(y_test, y_pred)  # 

# Use these instead:
precision = precision_score(y_test, y_pred)  # 
recall = recall_score(y_test, y_pred)        # 
f1 = f1_score(y_test, y_pred)               # 
roc_auc = roc_auc_score(y_test, y_proba)     # 
pr_auc = average_precision_score(y_test, y_proba)  # 

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]

# Classification report
print(classification_report(y_test, y_pred))
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## Complete Workflow

### Step-by-Step Pipeline

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Create pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# Use F1-score for cross-validation
f1_scorer = make_scorer(f1_score)

# Cross-validate
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring=f1_scorer
)

print(f"F1-score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
```

### With Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

# Grid search with F1-score
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## Best Practices

### Do's 

- Use appropriate metrics (F1, PR-AUC, ROC-AUC)
- Try multiple resampling techniques
- Use class weights when possible
- Tune decision threshold
- Validate on original distribution
- Consider cost-sensitive learning
- Use stratified cross-validation
- Monitor both precision and recall

### Don'ts 

- Don't use accuracy as primary metric
- Don't resample test set
- Don't ignore class imbalance
- Don't use random train/test split
- Don't oversample before splitting
- Don't forget to validate on original data
- Don't use same resampling for all problems

### Decision Tree

```
Is data imbalanced?
│
├─ YES → Use appropriate metrics (F1, PR-AUC)
│   │
│   ├─ Try resampling (SMOTE, ADASYN)
│   │
│   ├─ Use class weights
│   │
│   ├─ Tune threshold
│   │
│   └─ Consider cost-sensitive learning
│
└─ NO → Use standard metrics (accuracy)
```

---

## Quick Reference

### Resampling Methods

| Method | Type | Best For |
|--------|------|----------|
| **SMOTE** | Oversampling | General use, creates synthetic samples |
| **ADASYN** | Oversampling | Adaptive, focuses on hard examples |
| **Random Oversampling** | Oversampling | Simple, duplicates samples |
| **Random Undersampling** | Undersampling | Large datasets, removes samples |
| **Tomek Links** | Undersampling | Clean boundary, removes noisy samples |
| **SMOTE + Tomek** | Combined | Best of both worlds |

### Metrics Comparison

| Metric | Use When | Formula |
|--------|----------|---------|
| **F1-Score** | Balance precision/recall | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ |
| **Precision** | Minimize false positives | $\frac{TP}{TP + FP}$ |
| **Recall** | Minimize false negatives | $\frac{TP}{TP + FN}$ |
| **PR-AUC** | Imbalanced data | Area under PR curve |
| **ROC-AUC** | General performance | Area under ROC curve |

### Class Weight Calculation

```python
# Automatic
class_weight='balanced'

# Manual
n_samples / (n_classes * np.bincount(y))

# Custom
class_weight={0: 1, 1: 10}
```

---

## Common Pitfalls

### Pitfall 1: Resampling Test Set

```python
# WRONG
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)

# CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Pitfall 2: Using Accuracy

```python
# WRONG
accuracy = accuracy_score(y_test, y_pred)

# CORRECT
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_proba)
```

### Pitfall 3: Not Validating on Original Data

```python
# Always evaluate on original test distribution
# Don't resample test set!
```

---

**Remember**: Imbalanced data requires special handling. Use appropriate metrics and techniques!

