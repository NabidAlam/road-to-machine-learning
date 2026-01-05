# Classification Quick Reference Guide

Quick reference for classification algorithms, metrics, and best practices.

## Table of Contents

- [Algorithm Selection](#algorithm-selection)
- [Code Snippets](#code-snippets)
- [Evaluation Metrics](#evaluation-metrics)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Algorithm Selection

### Quick Decision Tree

```
Need to classify into categories?
│
├─ Binary or Multi-class?
│  ├─ Binary → Continue
│  └─ Multi-class → Continue (most algorithms handle both)
│
├─ Need interpretability?
│  ├─ YES → Logistic Regression or Decision Tree
│  └─ NO → Continue
│
├─ Linear decision boundary?
│  ├─ YES → Logistic Regression or Linear SVM
│  └─ NO → Continue
│
├─ Need best performance?
│  ├─ YES → Random Forest or XGBoost
│  └─ NO → Continue
│
├─ Small dataset?
│  ├─ YES → SVM or KNN
│  └─ NO → Random Forest
│
└─ Text data?
   └─ YES → Naive Bayes or Neural Networks
```

### Algorithm Comparison

| Algorithm | When to Use | Pros | Cons | Code |
|-----------|-------------|------|------|------|
| **Logistic Regression** | Baseline, interpretable | Simple, fast, interpretable | Assumes linearity | `LogisticRegression()` |
| **Decision Tree** | Need interpretability | Very interpretable, handles non-linearity | Prone to overfitting | `DecisionTreeClassifier()` |
| **Random Forest** | General purpose | Good performance, robust | Less interpretable | `RandomForestClassifier()` |
| **SVM** | Small-medium datasets | Effective, handles non-linearity | Slow on large data | `SVC(kernel='rbf')` |
| **KNN** | Local patterns | Simple, no training | Slow prediction | `KNeighborsClassifier()` |
| **Naive Bayes** | Text classification | Fast, many features | Assumes independence | `MultinomialNB()` |

---

## Code Snippets

### Basic Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale (for logistic regression, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Binary classification
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Multiclass
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# With class weights (for imbalanced data)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create tree
tree = DecisionTreeClassifier(
    max_depth=5,              # Control depth
    min_samples_split=10,     # Minimum samples to split
    min_samples_leaf=5,      # Minimum samples in leaf
    random_state=42
)
tree.fit(X_train, y_train)

# Visualize tree
plot_tree(tree, filled=True, feature_names=feature_names)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1               # Use all CPUs
)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
```

### SVM

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# RBF kernel (most common)
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train_scaled, y_train)
```

### KNN

```python
from sklearn.neighbors import KNeighborsClassifier

# Create KNN
knn = KNeighborsClassifier(n_neighbors=5)  # k=5
knn.fit(X_train_scaled, y_train)

# Find optimal k
from sklearn.model_selection import cross_val_score

k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append(scores.mean())

best_k = k_range[np.argmax(k_scores)]
```

### Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

### Threshold Tuning

```python
from sklearn.metrics import f1_score, precision_recall_curve

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = f1_scores[:-1]

best_threshold = thresholds[np.argmax(f1_scores)]

# Use optimal threshold
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
```

---

## Evaluation Metrics

### Quick Reference

| Metric | Formula | When to Use | Interpretation |
|--------|---------|-------------|----------------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Balanced data | Overall correctness |
| **Precision** | `TP / (TP + FP)` | False positives costly | Of positive predictions, how many correct |
| **Recall** | `TP / (TP + FN)` | False negatives costly | Of actual positives, how many found |
| **F1-Score** | `2 * (P * R) / (P + R)` | Balance needed | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Balanced data | Probability of ranking positive higher |
| **PR-AUC** | Area under PR curve | Imbalanced data | Average precision |

### Code

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

# Plot
plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## Handling Imbalanced Data

### Detection

```python
from collections import Counter

# Check class distribution
class_counts = Counter(y_train)
for class_label, count in class_counts.items():
    percentage = count / len(y_train) * 100
    print(f"Class {class_label}: {count} ({percentage:.1f}%)")

# Calculate imbalance ratio
minority = min(class_counts.values())
majority = max(class_counts.values())
ratio = majority / minority
print(f"Imbalance Ratio: {ratio:.1f}:1")
```

### Strategies

| Strategy | When to Use | Code |
|----------|-------------|------|
| **Class Weights** | Quick fix, tree-based models | `class_weight='balanced'` |
| **SMOTE** | Need more minority samples | `SMOTE().fit_resample(X, y)` |
| **Undersampling** | Large dataset, can afford to lose data | `RandomUnderSampler().fit_resample(X, y)` |
| **Threshold Tuning** | Model probabilities are good | Find optimal threshold |
| **Ensemble** | Need robust solution | `BalancedRandomForestClassifier()` |

---

## Common Issues & Solutions

### Issue 1: Low Accuracy on Imbalanced Data

**Symptoms:**
- High accuracy but poor performance on minority class
- Model always predicts majority class

**Solutions:**
- Use appropriate metrics (F1, ROC-AUC, PR-AUC)
- Apply resampling (SMOTE) or class weights
- Tune threshold

### Issue 2: Overfitting

**Symptoms:**
- High training accuracy, low test accuracy
- Large gap between train and test performance

**Solutions:**
- Reduce model complexity (max_depth, min_samples_split)
- Use regularization
- Get more training data
- Use ensemble methods (Random Forest)

### Issue 3: Poor Performance on Minority Class

**Symptoms:**
- Low recall for minority class
- High precision but low recall

**Solutions:**
- Use class weights
- Apply SMOTE
- Lower classification threshold
- Use cost-sensitive learning

### Issue 4: Slow Training/Prediction

**Symptoms:**
- Model takes too long to train or predict

**Solutions:**
- Use faster algorithms (Logistic Regression, Random Forest)
- Reduce features
- Use smaller dataset for training
- For KNN: Use approximate nearest neighbors

### Issue 5: Uncalibrated Probabilities

**Symptoms:**
- Probabilities don't match actual frequencies
- 80% probability doesn't mean 80% correct

**Solutions:**
- Use CalibratedClassifierCV
- Apply Platt scaling or isotonic regression

---

## Best Practices Checklist

### Before Training

- [ ] Data is clean (no missing values, outliers handled)
- [ ] Categorical variables encoded
- [ ] Features scaled (for logistic regression, SVM)
- [ ] Data is split (train/test or train/val/test)
- [ ] Class distribution checked
- [ ] Stratified split used (for imbalanced data)

### During Training

- [ ] Start with simple model (Logistic Regression)
- [ ] Use cross-validation
- [ ] Handle class imbalance appropriately
- [ ] Monitor training and validation performance
- [ ] Check for overfitting

### After Training

- [ ] Evaluate on test set (only once!)
- [ ] Calculate multiple metrics (not just accuracy)
- [ ] Use appropriate metrics for problem type
- [ ] Check confusion matrix
- [ ] Plot ROC and PR curves
- [ ] Tune threshold if needed
- [ ] Interpret feature importance

### Model Selection

- [ ] Compare multiple algorithms
- [ ] Use appropriate metric for comparison
- [ ] Consider interpretability needs
- [ ] Balance complexity and performance
- [ ] Consider deployment constraints

---

## Metric Selection Guide

### When to Use Which Metric

| Problem Type | Primary Metric | Why |
|--------------|----------------|-----|
| **Medical Diagnosis** | Recall | Don't miss cases |
| **Spam Detection** | Precision | Don't mark real emails as spam |
| **Fraud Detection** | F1-Score or PR-AUC | Need balance, imbalanced data |
| **General Classification** | ROC-AUC | Balanced data, overall performance |
| **Imbalanced Data** | PR-AUC or F1-Score | Accuracy misleading |

---

## Algorithm-Specific Tips

### Logistic Regression

- Scale features (StandardScaler)
- Use `max_iter=1000` if convergence warning
- `class_weight='balanced'` for imbalanced data
- `multi_class='multinomial'` for multiclass

### Decision Tree

- Control `max_depth` to prevent overfitting
- Use `min_samples_split` and `min_samples_leaf`
- Prone to overfitting - use Random Forest instead

### Random Forest

- `n_estimators=100` is good starting point
- `max_depth=None` often works well
- Use `n_jobs=-1` for parallel processing
- Feature importance available

### SVM

- Always scale features
- RBF kernel works well for most problems
- `gamma='scale'` is good default
- Slow on large datasets (>10K samples)

### KNN

- Scale features (distance-based)
- Find optimal k using cross-validation
- Slow for large datasets
- Use approximate methods for speed

---

## Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Low accuracy | Check class imbalance, use appropriate metrics |
| Overfitting | Reduce complexity, use regularization |
| Poor minority class performance | Use class weights or SMOTE |
| Slow training | Use faster algorithm, reduce features |
| Convergence warning | Increase max_iter, check data scaling |
| Low precision | Increase threshold, use class weights |
| Low recall | Decrease threshold, use SMOTE |

---

## Resources

- [Main Classification Guide](classification.md)
- [Advanced Topics](classification-advanced-topics.md)
- [Project Tutorial](classification-project-tutorial.md)
- [Scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html#classification)

---

**Remember**: For classification, always use appropriate metrics and handle class imbalance!


