# Supervised Learning - Classification Complete Guide

Comprehensive guide to classification algorithms for predicting categories.

## Table of Contents

- [Introduction to Classification](#introduction-to-classification)
- [Logistic Regression](#logistic-regression)
- [Decision Trees](#decision-trees)
- [Random Forests](#random-forests)
- [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [Naive Bayes](#naive-bayes)
- [Multi-Class Classification Strategies](#multi-class-classification-strategies)
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
- Email categorization
- Disease diagnosis
- Image recognition

### Types of Classification

1. **Binary Classification**: Two classes (spam/not spam)
   - Most common type
   - Examples: Fraud detection, email spam, disease diagnosis

2. **Multiclass Classification**: Multiple classes (cat/dog/bird)
   - Each sample belongs to exactly one class
   - Examples: Image classification, text categorization

3. **Multilabel Classification**: Multiple labels per sample
   - Each sample can have multiple labels
   - Examples: Tagging system, document topics

### When to Use Classification

- **Target variable is categorical** (not continuous)
- **Want to predict a category/class**
- **Need to classify into groups**

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
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, 
                          n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)  # Probabilities

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show probabilities for first few predictions
print("\nFirst 5 predictions with probabilities:")
for i in range(5):
    print(f"Sample {i+1}: Predicted={y_pred[i]}, "
          f"Prob(Class 0)={y_pred_proba[i][0]:.3f}, "
          f"Prob(Class 1)={y_pred_proba[i][1]:.3f}")
```

**Output:**
```
Accuracy: 0.890

Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.89      0.89       102
           1       0.89      0.89      0.89        98
    accuracy                           0.89       200
   macro avg       0.89      0.89      0.89       200
weighted avg       0.89      0.89      0.89       200

First 5 predictions with probabilities:
Sample 1: Predicted=0, Prob(Class 0)=0.823, Prob(Class 1)=0.177
Sample 2: Predicted=1, Prob(Class 0)=0.234, Prob(Class 1)=0.766
...
```

### Assumptions of Logistic Regression

1. **Binary outcome**: Target has two classes
2. **Independence**: Observations are independent
3. **Linearity**: Log-odds is linear in features
4. **No multicollinearity**: Features are not highly correlated
5. **Large sample size**: Works better with more data

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

Split data based on feature values to create tree-like structure. Uses splitting criteria (Gini impurity or entropy) to find best splits.

**Splitting Criteria:**

1. **Gini Impurity**: Measures probability of misclassification
   ```
   Gini = 1 - Σ(p_i)²
   where p_i = proportion of class i
   ```

2. **Entropy**: Measures information gain
   ```
   Entropy = -Σ(p_i * log₂(p_i))
   ```

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train
tree = DecisionTreeClassifier(
    max_depth=3, 
    criterion='gini',  # or 'entropy'
    random_state=42
)
tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# Advanced visualization with dtreeviz (optional)
# Install: pip install dtreeviz
try:
    from dtreeviz.trees import dtreeviz
    
    viz = dtreeviz(
        tree,
        X_train,
        y_train,
        target_name='Species',
        feature_names=iris.feature_names,
        class_names=list(iris.target_names),
        title="Decision Tree Visualization"
    )
    viz.view()
except ImportError:
    print("Install dtreeviz for advanced visualizations: pip install dtreeviz")

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)
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
- No feature scaling needed
- Handles non-linear relationships

### Hyperparameters

```python
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Maximum depth of trees
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples in leaf
    max_features='sqrt',     # Features to consider ('sqrt', 'log2', or number)
    bootstrap=True,          # Bootstrap sampling
    random_state=42
)
rf.fit(X_train, y_train)
```

### Feature Importance Visualization

```python
import matplotlib.pyplot as plt

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

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
  - Fast, interpretable
  - Use when data is linearly separable

- **Polynomial**: For polynomial relationships
  - Degree parameter controls complexity
  - Can overfit with high degree

- **RBF (Radial Basis Function)**: For complex non-linear boundaries (most common)
  - Gamma parameter controls smoothness
  - Default choice for most problems

### Hyperparameters

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# RBF SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

svm = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
```

---

## K-Nearest Neighbors (KNN)

### How KNN Works

Classifies based on k nearest neighbors' labels. Instance-based learning - no explicit training, just stores data.

**Distance Metrics:**
- **Euclidean**: Default, straight-line distance
- **Manhattan**: Sum of absolute differences
- **Minkowski**: Generalization of Euclidean and Manhattan

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Create KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,           # k value
    weights='uniform',       # 'uniform' or 'distance'
    metric='euclidean',      # Distance metric
    algorithm='auto'         # Algorithm for computing neighbors
)
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
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal k for KNN')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

### Pros and Cons

**Pros:**
- Simple to understand and implement
- No assumptions about data distribution
- Works well for non-linear problems
- Can be used for both classification and regression

**Cons:**
- Slow for large datasets (computes distances for all samples)
- Sensitive to irrelevant features
- Sensitive to scale (needs feature scaling)
- Memory intensive (stores all training data)

---

## Naive Bayes

### Introduction to Naive Bayes

**Naive Bayes** is a probabilistic classification algorithm based on Bayes' theorem with a "naive" assumption of feature independence. Despite this simplification, it often performs surprisingly well and is particularly effective for text classification and small datasets.

**Key Idea**: Calculate the probability of each class given the features, then predict the class with the highest probability.

### Bayes' Theorem

Bayes' theorem states:
```
P(Class | Features) = P(Features | Class) × P(Class) / P(Features)
```

Where:
- **P(Class | Features)**: Posterior probability (what we want to predict)
- **P(Features | Class)**: Likelihood (probability of features given class)
- **P(Class)**: Prior probability (probability of class)
- **P(Features)**: Evidence (normalizing constant)

**Naive Assumption**: Features are conditionally independent given the class. This means:
```
P(Features | Class) = P(Feature₁ | Class) × P(Feature₂ | Class) × ... × P(Featureₙ | Class)
```

### Types of Naive Bayes

#### 1. Gaussian Naive Bayes

Assumes features follow a Gaussian (normal) distribution. Best for continuous numerical data.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Gaussian Naive Bayes Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# View class probabilities for first test sample
print(f"\nClass probabilities for first test sample:")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: {y_pred_proba[0][i]:.3f}")
```

**How it works:**
- For each class, estimate mean (μ) and variance (σ²) of each feature
- Calculate likelihood using Gaussian probability density function
- Combine with prior probabilities to get posterior probabilities

#### 2. Multinomial Naive Bayes

Best for discrete count data, especially text classification (word counts, document-term matrices).

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample text data
texts = [
    "I love this movie",
    "This movie is great",
    "I hate this film",
    "This film is terrible",
    "Amazing movie",
    "Worst film ever"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Convert text to feature vectors (word counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions
y_pred = mnb.predict(X_test)
y_pred_proba = mnb.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Multinomial Naive Bayes Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Test on new text
new_text = ["This movie is amazing"]
new_X = vectorizer.transform(new_text)
prediction = mnb.predict(new_X)[0]
probability = mnb.predict_proba(new_X)[0]
print(f"\nNew text: '{new_text[0]}'")
print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Probability: {probability[prediction]:.3f}")
```

**Use Cases:**
- Spam email detection
- Sentiment analysis
- Document classification
- Any text classification task

#### 3. Bernoulli Naive Bayes

Best for binary/boolean features (features that are either present or absent). Each feature is treated as a binary variable.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate binary feature data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Convert to binary features (0 or 1)
X_binary = (X > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y, test_size=0.2, random_state=42
)

# Create and train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# Make predictions
y_pred = bnb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Bernoulli Naive Bayes Accuracy: {accuracy:.3f}")
```

**Use Cases:**
- Text classification with binary word presence (word appears or doesn't)
- Binary feature datasets
- Recommendation systems

### When to Use Naive Bayes

**Advantages:**
- **Fast**: Very fast training and prediction
- **Simple**: Easy to understand and implement
- **Small Data**: Works well with small datasets
- **Text Classification**: Excellent for text/spam detection
- **Probabilistic**: Provides probability estimates
- **No Hyperparameters**: Few parameters to tune

**Disadvantages:**
- **Feature Independence Assumption**: Rarely true in practice, but often works anyway
- **Limited Expressiveness**: May not capture complex relationships
- **Sensitive to Irrelevant Features**: Can be affected by noisy features

**Best For:**
- Text classification (spam, sentiment, document categorization)
- Small to medium datasets
- When you need fast predictions
- When interpretability is important
- Multi-class problems with many classes

**Not Best For:**
- Complex non-linear relationships
- When feature interactions are important
- Very large datasets (other algorithms may perform better)

### Hyperparameters

```python
# Gaussian Naive Bayes
gnb = GaussianNB(
    var_smoothing=1e-9  # Additive smoothing for variance (prevents zero variance)
)

# Multinomial Naive Bayes
mnb = MultinomialNB(
    alpha=1.0,  # Additive smoothing parameter (Laplace smoothing)
    fit_prior=True  # Learn class prior probabilities
)

# Bernoulli Naive Bayes
bnb = BernoulliNB(
    alpha=1.0,  # Additive smoothing parameter
    binarize=0.0,  # Threshold for binarizing features
    fit_prior=True
)
```

### Comparison with Other Algorithms

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

models = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("Algorithm Comparison on Iris Dataset:")
print("=" * 50)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Key Takeaways

1. **Naive Bayes is fast and simple** - Great baseline for classification
2. **Excellent for text** - Multinomial NB is a go-to for text classification
3. **Works with small data** - Can perform well even with limited samples
4. **Probabilistic output** - Provides interpretable probability estimates
5. **Feature independence assumption** - Often violated but still works well
6. **Choose the right variant** - Gaussian for continuous, Multinomial for counts, Bernoulli for binary

---

## Multi-Class Classification Strategies

### Introduction

When you have more than two classes, some algorithms (like SVM, Logistic Regression) are inherently binary. We need strategies to extend them to multi-class problems.

### One-vs-Rest (OvR) / One-vs-All (OvA)

**How it works:**
- Train one binary classifier per class
- Each classifier distinguishes one class from all others
- For prediction, choose the class with the highest confidence/probability

**Example with 3 classes (A, B, C):**
- Classifier 1: A vs (B, C)
- Classifier 2: B vs (A, C)
- Classifier 3: C vs (A, B)
- Prediction: Class with highest probability wins

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OvR with SVM (SVM is binary by default)
ovr_svm = OneVsRestClassifier(SVC(probability=True, random_state=42))
ovr_svm.fit(X_train, y_train)
y_pred_ovr = ovr_svm.predict(X_test)

print("One-vs-Rest with SVM:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ovr):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))

# Logistic Regression uses OvR by default for multiclass
lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression (OvR by default):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
```

**Advantages:**
- Simple and intuitive
- Works with any binary classifier
- Computationally efficient (n classifiers for n classes)
- Good for many classes

**Disadvantages:**
- Class imbalance (one class vs many)
- May not capture class relationships
- Can be sensitive to outliers

**When to use:**
- Many classes
- When classes are well-separated
- Default choice for most binary classifiers

### One-vs-One (OvO)

**How it works:**
- Train one binary classifier for each pair of classes
- For n classes, train n×(n-1)/2 classifiers
- Use voting: each classifier votes for one class, class with most votes wins

**Example with 3 classes (A, B, C):**
- Classifier 1: A vs B
- Classifier 2: A vs C
- Classifier 3: B vs C
- Prediction: Majority vote

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# OvO with SVM
ovo_svm = OneVsOneClassifier(SVC(random_state=42))
ovo_svm.fit(X_train, y_train)
y_pred_ovo = ovo_svm.predict(X_test)

print("One-vs-One with SVM:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ovo):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ovo, target_names=iris.target_names))

# Number of classifiers for 3 classes: 3 × 2 / 2 = 3
print(f"\nNumber of binary classifiers: {len(ovo_svm.estimators_)}")
```

**Advantages:**
- Each classifier only sees two classes (simpler problem)
- Often more accurate than OvR
- Good for small datasets
- Better when classes overlap

**Disadvantages:**
- More classifiers needed (n×(n-1)/2)
- Slower training and prediction
- Can be computationally expensive for many classes

**When to use:**
- Few classes (typically < 10)
- Small datasets
- When classes overlap significantly
- When accuracy is more important than speed

### Comparison: OvR vs OvO

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import cross_val_score
import numpy as np

def compare_strategies(X, y, dataset_name):
    """Compare OvR and OvO strategies"""
    print(f"\n{dataset_name} Dataset ({len(np.unique(y))} classes):")
    print("=" * 60)
    
    # OvR
    ovr = OneVsRestClassifier(SVC(random_state=42))
    ovr_scores = cross_val_score(ovr, X, y, cv=5, scoring='accuracy')
    
    # OvO
    ovo = OneVsOneClassifier(SVC(random_state=42))
    ovo_scores = cross_val_score(ovo, X, y, cv=5, scoring='accuracy')
    
    print(f"One-vs-Rest:  {ovr_scores.mean():.3f} (+/- {ovr_scores.std():.3f})")
    print(f"One-vs-One:   {ovo_scores.mean():.3f} (+/- {ovo_scores.std():.3f})")
    
    n_classes = len(np.unique(y))
    print(f"\nNumber of classifiers:")
    print(f"  OvR: {n_classes}")
    print(f"  OvO: {n_classes * (n_classes - 1) // 2}")

# Test on different datasets
iris = load_iris()
wine = load_wine()

compare_strategies(iris.data, iris.target, "Iris")
compare_strategies(wine.data, wine.target, "Wine")
```

### Decision Guide

```
How many classes?
│
├─ 2 classes → Use binary classifier directly
│
├─ 3-10 classes → Consider OvO (often more accurate)
│  │
│  └─ Need speed? → Use OvR
│
└─ > 10 classes → Use OvR (OvO becomes too expensive)
   │
   └─ Small dataset? → Still consider OvO
```

### Native Multi-Class Support

Some algorithms natively support multi-class classification:
- **Decision Trees**: Native multi-class
- **Random Forests**: Native multi-class
- **K-Nearest Neighbors**: Native multi-class
- **Naive Bayes**: Native multi-class
- **Neural Networks**: Native multi-class

```python
# These work directly with multi-class problems
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# No need for OvR or OvO wrappers
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

iris = load_iris()
X, y = iris.data, iris.target

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Key Takeaways

1. **OvR**: Train n classifiers (one per class), good for many classes
2. **OvO**: Train n×(n-1)/2 classifiers (one per pair), often more accurate for few classes
3. **Choose based on**: Number of classes, dataset size, speed requirements
4. **Many algorithms**: Support multi-class natively (no wrapper needed)
5. **Default behavior**: Most sklearn classifiers use OvR automatically

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

Overall correctness of predictions. Good for balanced datasets, misleading for imbalanced data.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# When accuracy is misleading (imbalanced data)
# Example: 90% class A, 10% class B
# Model predicting always A would have 90% accuracy but is useless!
```

### Precision, Recall, F1-Score

**Precision**: Of positive predictions, how many are correct
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)**: Of actual positives, how many were found
```
Recall = TP / (TP + FN)
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# For binary classification
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# For multiclass classification
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Average options: 'micro', 'macro', 'weighted'
# - 'micro': Calculate globally
# - 'macro': Average of per-class metrics
# - 'weighted': Weighted average by class support

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### ROC-AUC (Binary Classification)

**ROC Curve**: Plots True Positive Rate vs False Positive Rate at different thresholds
**AUC**: Area Under ROC Curve. Range: 0 to 1 (1 = perfect, 0.5 = random)

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# For binary classification
X_binary, y_binary = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AUC: {roc_auc:.3f}")
# Interpretation:
# AUC = 0.5: Random classifier
# AUC = 1.0: Perfect classifier
# AUC > 0.8: Good classifier
```

### Precision-Recall Curve

Better than ROC for imbalanced datasets.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Complete Evaluation Function

```python
def evaluate_classification(y_true, y_pred, y_pred_proba=None, target_names=None):
    """Comprehensive classification evaluation"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix, classification_report)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("Classification Metrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"  ROC-AUC:   {roc_auc:.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Usage
if len(np.unique(y_test)) == 2:  # Binary classification
    metrics = evaluate_classification(y_test, y_pred, y_pred_proba[:, 1])
else:  # Multiclass
    metrics = evaluate_classification(y_test, y_pred, target_names=iris.target_names)
```

---

## Bias Auditing and Fairness in Classification

### Why Check for Bias?

Real-world classification models can inadvertently discriminate against certain groups, leading to:
- **Unfair hiring decisions** (biased against demographics)
- **Discriminatory lending** (race/gender bias)
- **Unequal healthcare** (treatment recommendations)
- **Legal and ethical issues**

**Bias auditing is not optional** - it's a critical skill for responsible ML.

### Types of Bias

1. **Demographic Parity**: Equal positive prediction rates across groups
2. **Equalized Odds**: Equal true positive and false positive rates
3. **Calibration**: Equal prediction accuracy across groups

### Practical Bias Auditing with Fairlearn

**Installation:**
```bash
pip install fairlearn
```

**Example: Auditing a Classification Model**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from fairlearn.postprocessing import ThresholdOptimizer

# Load data (example: hiring dataset)
# Assume we have: features, target (hired/not), sensitive_feature (gender)
data = pd.read_csv("hiring_data.csv")
X = data.drop(['hired', 'gender'], axis=1)
y = data['hired']
sensitive_features = data['gender']  # Protected attribute

# Split data
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_features, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 1. Check Demographic Parity
# Measures: Are positive predictions distributed equally?
dp_diff = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)
print(f"Demographic Parity Difference: {dp_diff:.3f}")
# 0.0 = perfect fairness, 1.0 = maximum bias
# Should be < 0.1 for fairness

# 2. Check Equalized Odds
# Measures: Are TPR and FPR equal across groups?
eo_diff = equalized_odds_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)
print(f"Equalized Odds Difference: {eo_diff:.3f}")
# Should be < 0.1 for fairness

# 3. Detailed Metrics by Group
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': lambda y_true, y_pred: y_pred.mean()
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

print("\nMetrics by Group:")
print(metric_frame.by_group)

# Check for disparities
print("\nDisparities:")
print(metric_frame.difference())
```

### Mitigating Bias

**1. Post-Processing: Threshold Optimization**

```python
# Adjust decision thresholds to achieve fairness
mitigator = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",  # or "equalized_odds"
    preprocessor=None
)

# Fit on training data
mitigator.fit(X_train, y_train, sensitive_features=sens_train)

# Predict with fairness constraints
fair_predictions = mitigator.predict(
    X_test, 
    sensitive_features=sens_test
)

# Check improved fairness
dp_diff_fair = demographic_parity_difference(
    y_true=y_test,
    y_pred=fair_predictions,
    sensitive_features=sens_test
)
print(f"Demographic Parity (after mitigation): {dp_diff_fair:.3f}")
```

**2. In-Processing: Fairness-Aware Training**

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Use exponentiated gradient for fairness-aware training
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator=RandomForestClassifier(n_estimators=100),
    constraints=constraint
)

mitigator.fit(X_train, y_train, sensitive_features=sens_train)
fair_predictions = mitigator.predict(X_test)

# Verify fairness
dp_diff = demographic_parity_difference(
    y_test, fair_predictions, sens_test
)
print(f"Demographic Parity: {dp_diff:.3f}")
```

### Project: Audit Titanic Survival Model

**Task**: Check if the Titanic survival model is biased by gender or class.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference

# Load Titanic data
df = pd.read_csv("titanic.csv")

# Prepare features
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(df.mean())
y = df['Survived']
sensitive_feature = df['Sex']  # Check gender bias

# Split
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Audit for gender bias
dp_diff = demographic_parity_difference(
    y_test, y_pred, sens_test
)

print(f"Gender Bias (Demographic Parity): {dp_diff:.3f}")
if dp_diff > 0.1:
    print("WARNING: Model shows significant gender bias!")
    print("Consider using ThresholdOptimizer to mitigate bias.")

# Check by passenger class
sensitive_feature_class = df.loc[y_test.index, 'Pclass']
dp_diff_class = demographic_parity_difference(
    y_test, y_pred, sensitive_feature_class
)
print(f"\nClass Bias (Demographic Parity): {dp_diff_class:.3f}")
```

### Best Practices for Bias Auditing

1. **Always Audit**: Check for bias in every classification model
2. **Multiple Metrics**: Use both demographic parity and equalized odds
3. **Multiple Attributes**: Check bias across all sensitive features
4. **Document Findings**: Record bias metrics in model documentation
5. **Mitigate When Needed**: Use post-processing or in-processing methods
6. **Monitor in Production**: Bias can change over time (concept drift)

### Additional Tools

**AIF360 (IBM):**
```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Convert to AIF360 format
dataset = BinaryLabelDataset(
    df=df,
    label_names=['Survived'],
    protected_attribute_names=['Sex'],
    favorable_label=1,
    unfavorable_label=0
)

# Check bias
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'Sex': 0}],
    privileged_groups=[{'Sex': 1}]
)

print(f"Disparate Impact: {metric.disparate_impact()}")
# Should be close to 1.0 for fairness
```

### Key Takeaways

- **Bias auditing is essential** for responsible ML
- **Check multiple fairness metrics** (demographic parity, equalized odds)
- **Mitigate bias** using post-processing or in-processing methods
- **Document your findings** and mitigation strategies
- **Fairness is a technical skill**, not just an ethical consideration

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
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Create imbalanced data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Original class distribution:")
print(pd.Series(y_train).value_counts())

# Strategy 1: SMOTE (oversampling)
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("\nAfter SMOTE:")
    print(pd.Series(y_resampled).value_counts())
except ImportError:
    print("Install imbalanced-learn: pip install imbalanced-learn")

# Strategy 2: Class weights
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compare with and without class weights
model_unbalanced = LogisticRegression(random_state=42)
model_unbalanced.fit(X_train, y_train)
y_pred_unbalanced = model_unbalanced.predict(X_test)

print("\nWith class weights:")
print(classification_report(y_test, y_pred))
print("\nWithout class weights:")
print(classification_report(y_test, y_pred_unbalanced))
```

### Exercise 3: Model Comparison with Cross-Validation

**Task:** Compare all classification algorithms using cross-validation on multiple datasets.

**Solution:**
```python
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer()
}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for dataset_name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n{dataset_name} Dataset:")
    print("=" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"{name:20s}: {mean_score:.3f} (+/- {std_score:.3f})")
        
        if dataset_name not in results:
            results[dataset_name] = {}
        results[dataset_name][name] = mean_score
```

---

## Algorithm Comparison

### When to Use Each Algorithm

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Logistic Regression** | Linear relationships, interpretability | Fast, interpretable, probabilities | Assumes linearity |
| **Decision Trees** | Non-linear, interpretability needed | Interpretable, no scaling needed | Prone to overfitting |
| **Random Forests** | General purpose, robust | Handles overfitting, feature importance | Less interpretable, slower |
| **SVM** | Complex boundaries, small datasets | Effective for non-linear, good generalization | Slow for large datasets, memory intensive |
| **KNN** | Non-linear, small datasets | Simple, no assumptions | Slow for large datasets, sensitive to scale |
| **Naive Bayes** | Text classification, small datasets | Very fast, works with small data, probabilistic | Assumes feature independence |

### Quick Selection Guide

```
Need interpretability?
│
├─ YES → Logistic Regression or Decision Tree
│
└─ NO → Continue
   │
   ├─ Small dataset (< 10K samples)?
   │  ├─ YES → SVM or KNN
   │  └─ NO → Random Forest
   │
   └─ Need probabilities?
      ├─ YES → Logistic Regression, Random Forest, or Naive Bayes
      └─ NO → SVM or Decision Tree
   │
   └─ Text classification or small dataset?
      ├─ YES → Naive Bayes
      └─ NO → Continue with other algorithms
```

## Key Takeaways

1. **Logistic Regression**: Good baseline, interpretable, provides probabilities
2. **Decision Trees**: Interpretable, can overfit, no scaling needed
3. **Random Forests**: Robust, handles overfitting, feature importance
4. **SVM**: Good for complex boundaries, memory intensive
5. **KNN**: Simple, but slow for large datasets, needs scaling
6. **Naive Bayes**: Very fast, excellent for text classification, works with small data
7. **Multi-Class Strategies**: Use OvR for many classes, OvO for few classes with overlapping
8. **Evaluation**: Use multiple metrics, especially for imbalanced data
9. **Feature Scaling**: Important for SVM, KNN, and Logistic Regression
10. **Hyperparameter Tuning**: Critical for optimal performance

---

## Next Steps

- Practice with different datasets
- Experiment with hyperparameters
- Learn about handling imbalanced data
- Move to [05-model-evaluation-optimization](../05-model-evaluation-optimization/README.md)

**Remember**: Try multiple algorithms, compare performance, and choose based on your specific needs!

