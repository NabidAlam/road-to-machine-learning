# Feature Engineering Complete Guide

Comprehensive guide to creating and selecting the best features for your models.

## Table of Contents

- [Introduction](#introduction)
- [Feature Selection](#feature-selection)
- [Feature Transformation](#feature-transformation)
- [Handling Categorical Variables](#handling-categorical-variables)
- [Feature Scaling](#feature-scaling)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Creating New Features](#creating-new-features)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Feature Engineering?

**"Garbage In, Garbage Out"** - Good features are more important than algorithms!

**Impact:**
- Better features → Better models
- Can improve performance more than algorithm choice
- Domain knowledge is key

---

## Feature Selection

### Filter Methods

Statistical tests to select features.

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Select top k features using F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_features = selector.get_support()
print("Selected features:", feature_names[selected_features])
```

### Wrapper Methods

Use model performance to select features.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=5)
X_selected = selector.fit_transform(X, y)

print("Selected features:", feature_names[selector.support_])
```

### Embedded Methods

Feature selection built into model training.

```python
from sklearn.linear_model import LassoCV

# Lasso automatically selects features
lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Features with non-zero coefficients
selected = lasso.coef_ != 0
print("Selected features:", feature_names[selected])
```

---

## Feature Transformation

### Log Transformation

Handle skewed distributions.

```python
# Log transform
df['log_feature'] = np.log1p(df['feature'])  # log1p handles zeros

# Before and after
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['feature'], bins=50)
plt.title('Original (Skewed)')

plt.subplot(1, 2, 2)
plt.hist(df['log_feature'], bins=50)
plt.title('Log Transformed')
plt.show()
```

### Power Transformation

Box-Cox transformation.

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox transformation
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

### Binning

Convert continuous to categorical.

```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'], bins=5, labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])

# Equal-frequency bins
df['income_group'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
```

---

## Handling Categorical Variables

### One-Hot Encoding

Binary columns for each category.

```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encoding
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(df[['category']])

# Or with pandas
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
```

### Label Encoding

Numeric labels (for tree models).

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

### Target Encoding

Mean target per category.

```python
# Calculate mean target per category
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)
```

---

## Feature Scaling

### Standardization

Mean 0, Std 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify
print(f"Mean: {X_scaled.mean(axis=0)}")  # Should be ~0
print(f"Std: {X_scaled.std(axis=0)}")    # Should be ~1
```

### Normalization

Scale to [0, 1] range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Verify
print(f"Min: {X_normalized.min(axis=0)}")  # Should be 0
print(f"Max: {X_normalized.max(axis=0)}")  # Should be 1
```

### Robust Scaling

Using median and IQR (handles outliers).

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.show()
```

### t-SNE

Non-linear visualization.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('t-SNE Visualization')
plt.show()
```

---

## Creating New Features

### Domain Features

```python
# Example: E-commerce
df['total_spent'] = df['quantity'] * df['price']
df['avg_order_value'] = df['total_spent'] / df['order_count']
df['days_since_last_purchase'] = (today - df['last_purchase']).days
```

### Interaction Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
```

---

## Practice Exercises

### Exercise 1: Feature Selection

**Task:** Select top 5 features using different methods and compare.

**Solution:**
```python
# Method 1: Filter (F-test)
selector1 = SelectKBest(score_func=f_classif, k=5)
X1 = selector1.fit_transform(X, y)

# Method 2: Embedded (Lasso)
lasso = LassoCV()
lasso.fit(X, y)
selected2 = lasso.coef_ != 0
X2 = X[:, selected2]

# Compare
print("Filter method features:", feature_names[selector1.get_support()])
print("Lasso selected features:", feature_names[selected2])
```

---

## Key Takeaways

1. **Feature selection**: Remove irrelevant features
2. **Transformation**: Handle skewed data
3. **Encoding**: Convert categorical to numeric
4. **Scaling**: Required for distance-based algorithms
5. **Domain knowledge**: Most valuable!

---

## Next Steps

- Practice with real datasets
- Experiment with different techniques
- Move to [08-unsupervised-learning](../08-unsupervised-learning/README.md)

**Remember**: Good features beat complex algorithms!

