# Introduction to ML Cheatsheet

Quick reference for Machine Learning fundamentals, types, workflow, and key concepts.

## Table of Contents

- [ML Types](#ml-types)
- [ML Workflow](#ml-workflow)
- [Key Concepts](#key-concepts)
- [Scikit-learn Basics](#scikit-learn-basics)
- [Common Algorithms](#common-algorithms)

---

## ML Types

### Supervised Learning

**Definition**: Learn from labeled data (input-output pairs)

**Types**:
- **Regression**: Predict continuous values (price, temperature)
- **Classification**: Predict categories (spam/not spam, cat/dog)

**Example**:
```python
from sklearn.linear_model import LinearRegression

# Training data: X (features) and y (labels)
X_train = [[25], [30], [35]]
y_train = [50000, 60000, 70000]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[28]])  # Predict salary for age 28
```

### Unsupervised Learning

**Definition**: Learn from unlabeled data (find patterns)

**Types**:
- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Reduce feature space
- **Anomaly Detection**: Find outliers

**Example**:
```python
from sklearn.cluster import KMeans

# Unlabeled data
X = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]

# Cluster into 2 groups
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.predict(X)
```

### Reinforcement Learning

**Definition**: Learn through interaction with environment (trial and error)

**Components**:
- **Agent**: Learner
- **Environment**: World agent interacts with
- **Actions**: What agent can do
- **Rewards**: Feedback from environment
- **Policy**: Strategy for choosing actions

---

## ML Workflow

### 1. Problem Definition

```python
# Understand the business problem
# - What are we trying to predict?
# - What data do we have?
# - What are the success metrics?
```

### 2. Data Collection

```python
import pandas as pd

# Load data
data = pd.read_csv("data.csv")
print(data.head())
print(data.info())
print(data.describe())
```

### 3. Data Preparation

```python
# Handle missing values
data = data.dropna()  # or fillna()

# Handle outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature selection
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
```

### 4. Feature Engineering

```python
# Create new features
data['new_feature'] = data['feature1'] * data['feature2']

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. Model Selection

```python
# Choose algorithm based on problem type
# - Regression: Linear Regression, Random Forest, XGBoost
# - Classification: Logistic Regression, SVM, Random Forest
# - Clustering: K-Means, DBSCAN
```

### 6. Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 7. Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

### 8. Deployment

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')

# Make predictions
predictions = model.predict(new_data)
```

### 9. Monitoring

```python
# Track model performance over time
# - Monitor accuracy/error metrics
# - Detect data drift
# - Retrain when needed
```

---

## Key Concepts

### Overfitting vs Underfitting

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Overfitting** | High train accuracy, low test accuracy | Regularization, more data, simpler model |
| **Underfitting** | Low train and test accuracy | More features, complex model, longer training |

### Bias-Variance Tradeoff

- **High Bias**: Model too simple (underfitting)
- **High Variance**: Model too complex (overfitting)
- **Goal**: Balance both

### Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

# Split into train and temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Use:
# - Train: Fit model
# - Validation: Tune hyperparameters
# - Test: Final evaluation (only once!)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# K-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean score: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
```

---

## Scikit-learn Basics

### Common Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Predict
y_pred = model.predict(X_test_scaled)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

### Model Types

```python
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Dimensionality Reduction
from sklearn.decomposition import PCA
```

---

## Common Algorithms

### Classification Algorithms

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Logistic Regression** | Binary classification | Fast, interpretable | Linear boundaries |
| **Decision Tree** | Interpretable models | Easy to understand | Prone to overfitting |
| **Random Forest** | General purpose | Robust, handles non-linearity | Less interpretable |
| **SVM** | High-dimensional data | Effective, versatile | Slow on large datasets |
| **KNN** | Simple, lazy learning | No training needed | Slow prediction |

### Regression Algorithms

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Linear Regression** | Linear relationships | Fast, interpretable | Assumes linearity |
| **Polynomial Regression** | Non-linear relationships | Flexible | Can overfit |
| **Random Forest** | Non-linear patterns | Robust | Less interpretable |

### Clustering Algorithms

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **K-Means** | Spherical clusters | Fast, simple | Need to specify k |
| **DBSCAN** | Arbitrary shapes | No need for k | Sensitive to parameters |
| **Hierarchical** | Hierarchical structure | Visual dendrogram | Computationally expensive |

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred)
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

## Best Practices

### Do's 

- Start with simple models
- Use appropriate evaluation metrics
- Split data properly (train/val/test)
- Handle missing values
- Scale features when needed
- Use cross-validation
- Document everything
- Version control code and data

### Don'ts 

- Don't touch test set until final evaluation
- Don't use test set for hyperparameter tuning
- Don't ignore data leakage
- Don't use accuracy for imbalanced data
- Don't skip data exploration
- Don't forget to handle missing values
- Don't use random split for time series

---

## Quick Reference

### Problem Type → Algorithm

| Problem Type | Recommended Algorithms |
|--------------|------------------------|
| **Binary Classification** | Logistic Regression, Random Forest, SVM |
| **Multi-class Classification** | Random Forest, XGBoost, Neural Networks |
| **Regression** | Linear Regression, Random Forest, XGBoost |
| **Clustering** | K-Means, DBSCAN, Hierarchical |
| **Anomaly Detection** | Isolation Forest, One-Class SVM |

### Data Size → Algorithm

| Data Size | Recommended Approach |
|-----------|---------------------|
| **Small (< 1K samples)** | Simple models (Logistic Regression, SVM) |
| **Medium (1K-100K)** | Random Forest, XGBoost |
| **Large (> 100K)** | XGBoost, LightGBM, Neural Networks |

---

**Remember**: Start simple, iterate, and always validate your approach!

