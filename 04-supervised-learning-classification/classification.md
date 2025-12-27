# Supervised Learning - Classification Complete Guide

Comprehensive guide to classification algorithms for predicting categories.

## Table of Contents

- [Introduction to Classification](#introduction-to-classification)
- [Logistic Regression](#logistic-regression)
- [Decision Trees](#decision-trees)
- [Random Forests](#random-forests)
- [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [Evaluation Metrics](#evaluation-metrics)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Classification

### What is Classification?

Classification predicts categorical labels/classes. Unlike regression (continuous values), classification predicts categories.

**Examples:**
- Spam/Not Spam
- Cat/Dog/Bird
- Healthy/Sick
- High/Medium/Low risk

### Types of Classification

1. **Binary Classification**: Two classes (spam/not spam)
2. **Multiclass Classification**: Multiple classes (cat/dog/bird)
3. **Multilabel Classification**: Multiple labels per sample

---

## Logistic Regression

### Why "Logistic"?

Uses logistic (sigmoid) function to map predictions to probabilities [0, 1].

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

### Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, 
                          n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probabilities

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Output:**
```
Accuracy: 0.890

Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.89      0.89       102
           1       0.89      0.89      0.89        98
    accuracy                           0.89       200
```

### Multiclass Classification

```python
from sklearn.datasets import load_iris

# Load Iris dataset (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train (automatically handles multiclass)
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### Decision Boundary

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 2D example for visualization
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'blue']))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

---

## Decision Trees

### How Decision Trees Work

Split data based on feature values to create tree-like structure.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names)
plt.show()

# Feature importance
for name, importance in zip(iris.feature_names, tree.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

### Hyperparameters

```python
# Control tree complexity
tree = DecisionTreeClassifier(
    max_depth=5,           # Maximum depth
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples in leaf
    max_features='sqrt',    # Features to consider
    random_state=42
)
tree.fit(X_train, y_train)
```

---

## Random Forests

### How Random Forests Work

Ensemble of decision trees. Each tree votes, majority wins.

```python
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

### Advantages

- Reduces overfitting (compared to single tree)
- Handles missing values
- Feature importance
- Works well out of the box

---

## Support Vector Machines (SVM)

### How SVM Works

Finds optimal decision boundary (maximum margin) between classes.

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred_linear):.3f}")

# RBF (Radial Basis Function) kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(f"RBF SVM Accuracy: {accuracy_score(y_test, y_pred_rbf):.3f}")

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
print(f"Polynomial SVM Accuracy: {accuracy_score(y_test, y_pred_poly):.3f}")
```

### Kernel Types

- **Linear**: For linearly separable data
- **Polynomial**: For polynomial relationships
- **RBF**: For complex non-linear boundaries (most common)

---

## K-Nearest Neighbors (KNN)

### How KNN Works

Classifies based on k nearest neighbors' labels.

```python
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # k=5
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.3f}")

# Find optimal k
k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_range[np.argmax(k_scores)]
print(f"Best k: {best_k}")

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal k')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.legend()
plt.show()
```

---

## Evaluation Metrics

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot()
plt.show()
```

### Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### ROC-AUC (Binary Classification)

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# For binary classification
X_binary, y_binary = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC: {roc_auc:.3f}")
```

---

## Practice Exercises

### Exercise 1: Compare Classification Algorithms

**Task:** Compare Logistic Regression, Decision Tree, Random Forest, and KNN on Iris dataset.

**Solution:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Exercise 2: Handle Imbalanced Data

**Task:** Create imbalanced dataset and compare different strategies.

**Solution:**
```python
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Create imbalanced data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                          random_state=42)

print("Original class distribution:")
print(pd.Series(y).value_counts())

# Strategy 1: SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Strategy 2: Class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)
```

---

## Key Takeaways

1. **Logistic Regression**: Good baseline, interpretable
2. **Decision Trees**: Interpretable, can overfit
3. **Random Forests**: Robust, handles overfitting
4. **SVM**: Good for complex boundaries
5. **KNN**: Simple, but slow for large datasets
6. **Evaluation**: Use multiple metrics, especially for imbalanced data

---

## Next Steps

- Practice with different datasets
- Experiment with hyperparameters
- Move to [05-model-evaluation-optimization](../05-model-evaluation-optimization/README.md)

**Remember**: Try multiple algorithms and compare performance!

