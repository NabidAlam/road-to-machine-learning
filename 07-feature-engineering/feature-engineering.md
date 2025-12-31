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
- Often the difference between winning and losing in competitions

### What is Feature Engineering?

Feature engineering is the process of creating, selecting, and transforming features to improve model performance. It includes:

1. **Feature Creation**: Creating new features from existing ones
2. **Feature Selection**: Choosing the most relevant features
3. **Feature Transformation**: Transforming features to better distributions
4. **Feature Encoding**: Converting categorical to numerical
5. **Feature Scaling**: Normalizing feature scales

### When to Do Feature Engineering

**Always!** But especially when:
- Model performance is poor
- Features have different scales
- Categorical variables present
- High-dimensional data
- Skewed distributions
- Missing relationships between features

---

## Feature Selection

### Why Feature Selection?

- Reduces overfitting
- Improves model interpretability
- Reduces training time
- Removes noise and irrelevant features
- Prevents curse of dimensionality

### Filter Methods

Statistical tests to select features. Fast and independent of model.

**Common Methods:**
- **F-test (ANOVA)**: For numerical features and categorical target
- **Chi-square**: For categorical features and categorical target
- **Mutual Information**: Measures dependency between features and target
- **Correlation**: Removes highly correlated features

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
import pandas as pd

# Method 1: F-test (for numerical features)
selector_f = SelectKBest(score_func=f_classif, k=5)
X_selected_f = selector_f.fit_transform(X, y)

# Get scores and selected features
feature_scores = pd.DataFrame({
    'feature': feature_names,
    'score': selector_f.scores_,
    'selected': selector_f.get_support()
}).sort_values('score', ascending=False)

print("F-test Feature Selection:")
print(feature_scores)

# Method 2: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=5)
X_selected_mi = selector_mi.fit_transform(X, y)

# Method 3: Chi-square (for categorical features)
# selector_chi2 = SelectKBest(score_func=chi2, k=5)
# X_selected_chi2 = selector_chi2.fit_transform(X_categorical, y)

# Compare methods
print(f"\nF-test selected: {feature_names[selector_f.get_support()]}")
print(f"MI selected: {feature_names[selector_mi.get_support()]}")
```

### Correlation-Based Selection

```python
# Remove highly correlated features
correlation_matrix = pd.DataFrame(X).corr().abs()

# Find pairs with high correlation
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.8:  # Threshold
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print("Highly correlated feature pairs:")
for feat1, feat2, corr in high_corr_pairs:
    print(f"  {feat1} - {feat2}: {corr:.3f}")

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Wrapper Methods

Use model performance to select features. More accurate but slower than filter methods.

**Types:**
- **Forward Selection**: Start with no features, add one at a time
- **Backward Elimination**: Start with all features, remove one at a time
- **Recursive Feature Elimination (RFE)**: Recursively removes least important features

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Recursive Feature Elimination
estimator = LogisticRegression(random_state=42, max_iter=1000)
selector = RFE(estimator, n_features_to_select=5, step=1)
X_selected = selector.fit_transform(X, y)

print("RFE Selected features:")
print(feature_names[selector.support_])
print(f"\nFeature rankings (1 = selected):")
for i, (name, rank) in enumerate(zip(feature_names, selector.ranking_)):
    print(f"  {name}: {rank}")

# RFE with Cross-Validation (finds optimal number of features)
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(X, y)

print(f"\nOptimal number of features: {rfecv.n_features_}")
print(f"Selected features: {feature_names[rfecv.support_]}")

# Plot number of features vs CV score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of Features Selected', fontsize=12)
plt.ylabel('Cross-Validated Score', fontsize=12)
plt.title('RFE Cross-Validation', fontsize=14, fontweight='bold')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Sequential Feature Selection

```python
from sklearn.feature_selection import SequentialFeatureSelector

# Forward selection
sfs_forward = SequentialFeatureSelector(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=5,
    direction='forward',
    cv=5,
    n_jobs=-1
)
sfs_forward.fit(X, y)

print("Forward Selection features:")
print(feature_names[sfs_forward.get_support()])

# Backward elimination
sfs_backward = SequentialFeatureSelector(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=5,
    direction='backward',
    cv=5,
    n_jobs=-1
)
sfs_backward.fit(X, y)

print("\nBackward Elimination features:")
print(feature_names[sfs_backward.get_support()])
```

### Embedded Methods

Feature selection built into model training. More efficient than wrapper methods.

**Common Methods:**
- **Lasso (L1)**: Sets coefficients to zero
- **Elastic Net**: Combines L1 and L2
- **Tree-based**: Feature importance from Random Forest, XGBoost

```python
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier

# Method 1: Lasso (L1 regularization)
lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
lasso.fit(X, y)

# Features with non-zero coefficients
selected_lasso = lasso.coef_ != 0
print("Lasso selected features:")
print(feature_names[selected_lasso])
print(f"Number of features: {selected_lasso.sum()}")
print(f"Optimal alpha: {lasso.alpha_:.4f}")

# Method 2: Elastic Net
elastic = ElasticNetCV(cv=5, random_state=42, n_jobs=-1)
elastic.fit(X, y)

selected_elastic = elastic.coef_ != 0
print(f"\nElastic Net selected features:")
print(feature_names[selected_elastic])

# Method 3: Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Select top k features by importance
importance_threshold = np.percentile(rf.feature_importances_, 75)
selected_rf = rf.feature_importances_ > importance_threshold

print(f"\nRandom Forest selected features (top 25%):")
print(feature_names[selected_rf])

# Compare methods
print("\nFeature Selection Comparison:")
comparison = pd.DataFrame({
    'Feature': feature_names,
    'Lasso': selected_lasso,
    'Elastic Net': selected_elastic,
    'Random Forest': selected_rf
})
print(comparison[comparison.any(axis=1)])  # Show only selected features
```

---

## Feature Transformation

### Why Transform Features?

- Handle skewed distributions
- Meet algorithm assumptions (normality)
- Improve model performance
- Reduce impact of outliers

### Log Transformation

Handle right-skewed distributions. Common for count data, prices, sizes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Log transform
df['log_feature'] = np.log1p(df['feature'])  # log1p handles zeros

# Before and after comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['feature'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Original (Skewed)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Value', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['log_feature'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Log Transformed', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(Value)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check skewness
from scipy.stats import skew
print(f"Original skewness: {skew(df['feature']):.3f}")
print(f"Log transformed skewness: {skew(df['log_feature']):.3f}")

# Different log transformations
df['log'] = np.log1p(df['feature'])        # Natural log
df['log10'] = np.log10(df['feature'] + 1)  # Base 10 log
df['sqrt'] = np.sqrt(df['feature'])        # Square root (weaker transformation)
```

### Power Transformation

Box-Cox and Yeo-Johnson transformations. Automatically finds best transformation.

```python
from sklearn.preprocessing import PowerTransformer
from scipy import stats

# Box-Cox transformation (requires positive values)
# For positive values only
X_positive = X - X.min() + 1  # Make all positive

pt_boxcox = PowerTransformer(method='box-cox')
X_boxcox = pt_boxcox.fit_transform(X_positive)

# Yeo-Johnson transformation (works with any values)
pt_yeojohnson = PowerTransformer(method='yeo-johnson')
X_yeojohnson = pt_yeojohnson.fit_transform(X)

# Compare transformations
print("Skewness Comparison:")
print(f"Original: {skew(X.flatten()):.3f}")
print(f"Box-Cox: {skew(X_boxcox.flatten()):.3f}")
print(f"Yeo-Johnson: {skew(X_yeojohnson.flatten()):.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(X.flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Original', fontsize=12, fontweight='bold')
axes[1].hist(X_boxcox.flatten(), bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Box-Cox', fontsize=12, fontweight='bold')
axes[2].hist(X_yeojohnson.flatten(), bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[2].set_title('Yeo-Johnson', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Square Root Transformation

Weaker than log, good for count data.

```python
# Square root transformation
df['sqrt_feature'] = np.sqrt(df['feature'])

# Cube root (even weaker)
df['cbrt_feature'] = np.cbrt(df['feature'])
```

### Binning

Convert continuous to categorical. Useful for non-linear relationships and handling outliers.

```python
# Equal-width bins (fixed width)
df['age_group'] = pd.cut(
    df['age'], 
    bins=5, 
    labels=['Very Young', 'Young', 'Adult', 'Senior', 'Elderly']
)

# Equal-frequency bins (quantiles)
df['income_group'] = pd.qcut(
    df['income'], 
    q=4, 
    labels=['Low', 'Medium', 'High', 'Very High'],
    duplicates='drop'  # Handle duplicate edges
)

# Custom bins
df['custom_age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

# One-hot encode bins
df_encoded = pd.get_dummies(df, columns=['age_group'], prefix='age')

# Compare binning strategies
print("Equal-width bins:")
print(df['age_group'].value_counts().sort_index())

print("\nEqual-frequency bins:")
print(df['income_group'].value_counts().sort_index())
```

### Discretization

```python
from sklearn.preprocessing import KBinsDiscretizer

# KBinsDiscretizer with different strategies
strategies = ['uniform', 'quantile', 'kmeans']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, strategy in enumerate(strategies):
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=strategy)
    X_discrete = discretizer.fit_transform(X[:, [0]])  # Transform one feature
    
    axes[idx].hist(X_discrete.flatten(), bins=5, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{strategy.capitalize()} Strategy', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Bin', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Handling Categorical Variables

### Why Encode Categorical Variables?

Most ML algorithms require numerical input. Categorical variables must be converted to numbers.

### One-Hot Encoding

Binary columns for each category. Best for nominal categories (no order).

**Pros:**
- No ordinality assumption
- Works well with linear models
- Preserves all information

**Cons:**
- Creates many features (curse of dimensionality)
- Can cause multicollinearity

```python
from sklearn.preprocessing import OneHotEncoder

# Method 1: Scikit-learn OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
X_encoded = encoder.fit_transform(df[['category']])

# Get feature names
feature_names_encoded = encoder.get_feature_names_out(['category'])
print("Encoded features:", feature_names_encoded)

# Method 2: Pandas get_dummies (easier for DataFrames)
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True, prefix='cat')

# Compare
print(f"Original categories: {df['category'].nunique()}")
print(f"One-hot encoded columns: {len(feature_names_encoded)}")
```

### Label Encoding

Numeric labels (for tree models). Best for ordinal categories.

**Pros:**
- Only one column
- Preserves ordinality
- Works well with tree models

**Cons:**
- Assumes ordinality (can mislead linear models)
- May create false relationships

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Method 1: LabelEncoder (for single column)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Method 2: OrdinalEncoder (for multiple columns)
ordinal_encoder = OrdinalEncoder()
X_ordinal = ordinal_encoder.fit_transform(df[['category1', 'category2']])

# Custom mapping for ordinal data
ordinal_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}
df['priority_encoded'] = df['priority'].map(ordinal_mapping)
```

### Target Encoding

Mean target per category. Powerful but can overfit.

**Pros:**
- Captures target relationship
- Only one column
- Often improves performance

**Cons:**
- Risk of overfitting
- Needs careful cross-validation

```python
from sklearn.model_selection import KFold

# Method 1: Simple target encoding (can overfit!)
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)

# Method 2: Cross-validated target encoding (better!)
def target_encode_cv(df, cat_col, target_col, cv=5):
    """Target encoding with cross-validation to prevent overfitting"""
    df_encoded = df.copy()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train_mean = df.iloc[train_idx].groupby(cat_col)[target_col].mean()
        df_encoded.loc[val_idx, f'{cat_col}_target_encoded'] = \
            df.loc[val_idx, cat_col].map(train_mean)
    
    # Fill any remaining NaN with global mean
    global_mean = df[target_col].mean()
    df_encoded[f'{cat_col}_target_encoded'].fillna(global_mean, inplace=True)
    
    return df_encoded

df_encoded = target_encode_cv(df, 'category', 'target', cv=5)
```

### Frequency Encoding

Count of category occurrences.

```python
# Frequency encoding
category_counts = df['category'].value_counts()
df['category_frequency'] = df['category'].map(category_counts)

# Normalized frequency
category_freq_norm = df['category'].value_counts(normalize=True)
df['category_freq_norm'] = df['category'].map(category_freq_norm)
```

### Binary Encoding

Combines hash encoding and one-hot encoding.

```python
# Binary encoding (for high cardinality)
import category_encoders as ce

# Install: pip install category-encoders
try:
    encoder = ce.BinaryEncoder(cols=['category'])
    df_encoded = encoder.fit_transform(df)
    print("Binary encoding reduces features for high cardinality")
except ImportError:
    print("Install category-encoders: pip install category-encoders")
```

### Encoding Comparison

```python
# Compare encoding methods
encoding_results = {}

# One-hot encoding
encoder_ohe = OneHotEncoder(drop='first', sparse=False)
X_ohe = encoder_ohe.fit_transform(df[['category']])
model_ohe = LogisticRegression(random_state=42, max_iter=1000)
scores_ohe = cross_val_score(model_ohe, X_ohe, y, cv=5)
encoding_results['One-Hot'] = scores_ohe.mean()

# Label encoding
le = LabelEncoder()
X_le = le.fit_transform(df['category']).reshape(-1, 1)
model_le = LogisticRegression(random_state=42, max_iter=1000)
scores_le = cross_val_score(model_le, X_le, y, cv=5)
encoding_results['Label'] = scores_le.mean()

# Target encoding
df_target = target_encode_cv(df, 'category', 'target')
X_target = df_target[['category_target_encoded']].values
model_target = LogisticRegression(random_state=42, max_iter=1000)
scores_target = cross_val_score(model_target, X_target, y, cv=5)
encoding_results['Target'] = scores_target.mean()

print("Encoding Method Comparison:")
for method, score in sorted(encoding_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method}: {score:.3f}")
```

---

## Feature Scaling

### Why Scale Features?

- Distance-based algorithms (KNN, SVM) are sensitive to scale
- Gradient descent converges faster
- Some algorithms assume features are on similar scales
- Prevents features with large values from dominating

### When to Scale

**Always scale for:**
- K-Nearest Neighbors
- Support Vector Machines
- Neural Networks
- Logistic Regression (with regularization)
- K-Means clustering
- PCA

**Don't need to scale for:**
- Decision Trees
- Random Forests
- Gradient Boosting
- XGBoost, LightGBM, CatBoost

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

Using median and IQR. Best for data with outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

# Uses median and IQR instead of mean and std
# Less affected by outliers

# Compare scaling methods
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].boxplot(X[:, :5].T)
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')

# Standardized
axes[0, 1].boxplot(X_scaled[:, :5].T)
axes[0, 1].set_title('Standardized', fontsize=12, fontweight='bold')

# Min-Max
axes[1, 0].boxplot(X_normalized[:, :5].T)
axes[1, 0].set_title('Min-Max Normalized', fontsize=12, fontweight='bold')

# Robust
axes[1, 1].boxplot(X_robust[:, :5].T)
axes[1, 1].set_title('Robust Scaled', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

### MaxAbs Scaling

Scale by maximum absolute value. Preserves sparsity.

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_maxabs = scaler.fit_transform(X)

# Scales to [-1, 1] range
print(f"Min: {X_maxabs.min(axis=0)[:5]}")
print(f"Max: {X_maxabs.max(axis=0)[:5]}")
```

### Scaling Comparison

```python
# Compare impact on model performance
from sklearn.neighbors import KNeighborsClassifier

scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

results = {}

for name, scaler in scalers.items():
    if scaler is None:
        X_model = X
    else:
        X_model = scaler.fit_transform(X)
    
    model = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(model, X_model, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Dimensionality Reduction

### Why Reduce Dimensions?

- Reduce overfitting
- Improve visualization
- Reduce computational cost
- Remove noise
- Handle multicollinearity

### Principal Component Analysis (PCA)

Linear dimensionality reduction. Finds directions of maximum variance.

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance per component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Find optimal number of components
pca_full = PCA()
pca_full.fit(X)

# Cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontsize=12)
plt.title('PCA Explained Variance', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Visualize in 2D
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Use PCA for dimensionality reduction
pca_reduced = PCA(n_components=0.95)  # Keep 95% variance
X_pca_reduced = pca_reduced.fit_transform(X)
print(f"\nOriginal dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca_reduced.shape[1]}")
```

### t-SNE

Non-linear visualization. Good for visualization, not for feature reduction.

```python
from sklearn.manifold import TSNE

# t-SNE (slow for large datasets)
# Use subset for demonstration
X_sample = X[:1000] if len(X) > 1000 else X
y_sample = y[:1000] if len(y) > 1000 else y

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.title('t-SNE Visualization', fontsize=14, fontweight='bold')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Note: t-SNE is for visualization, not feature engineering
# Use PCA or other methods for actual dimensionality reduction
```

### UMAP

Modern alternative to t-SNE. Faster and preserves global structure.

```python
try:
    import umap
    
    # UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title('UMAP Visualization', fontsize=14, fontweight='bold')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install UMAP: pip install umap-learn")
```

---

## Creating New Features

### Why Create New Features?

- Capture relationships between features
- Encode domain knowledge
- Improve model performance
- Create non-linear relationships

### Domain-Specific Features

Use domain knowledge to create meaningful features.

```python
# Example: E-commerce
df['total_spent'] = df['quantity'] * df['price']
df['avg_order_value'] = df['total_spent'] / df['order_count']
df['days_since_last_purchase'] = (pd.Timestamp.today() - df['last_purchase']).dt.days
df['purchase_frequency'] = df['total_purchases'] / df['days_as_customer']
df['discount_ratio'] = df['discount_amount'] / df['original_price']

# Example: Time series
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month
df['quarter'] = df['timestamp'].dt.quarter

# Example: Text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']
```

### Interaction Features

Combine features to capture interactions.

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features (all combinations)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")

# Interaction-only (no squared terms)
poly_interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = poly_interaction.fit_transform(X)

print(f"Interaction features: {X_interaction.shape[1]}")

# Manual interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-6)  # Avoid division by zero
df['feature1_plus_feature2'] = df['feature1'] + df['feature2']
df['feature1_minus_feature2'] = df['feature1'] - df['feature2']
```

### Aggregation Features

Group statistics and aggregations.

```python
# Group aggregations
group_stats = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()

# Flatten column names
group_stats.columns = ['category', 'value_mean', 'value_std', 'value_min', 'value_max', 'value_count']

# Merge back
df = df.merge(group_stats, on='category', how='left')

# Rolling statistics (for time series)
df['value_rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['value_rolling_std_7'] = df['value'].rolling(window=7).std()
df['value_lag_1'] = df['value'].shift(1)  # Previous value
df['value_diff'] = df['value'].diff()  # Difference from previous
```

### Ratio and Difference Features

```python
# Ratio features
df['price_per_unit'] = df['price'] / df['quantity']
df['income_to_expense'] = df['income'] / (df['expenses'] + 1)

# Difference features
df['age_difference'] = df['age'] - df['avg_age']
df['price_difference'] = df['price'] - df['avg_price']

# Percentage change
df['price_pct_change'] = df['price'].pct_change()
```

### Binning and Grouping Features

```python
# Create bins and use as features
df['age_group'] = pd.cut(df['age'], bins=5, labels=False)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=False, duplicates='drop')

# Group by multiple features
df['group_id'] = df.groupby(['category', 'region']).ngroup()
```

### Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Create feature engineering pipeline
numeric_features = ['feature1', 'feature2', 'feature3']
categorical_features = ['category', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Use in pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)
```

---

## Feature Engineering Workflow

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                  PowerTransformer, FunctionTransformer)
from sklearn.feature_selection import SelectKBest, f_classif

# Define feature types
numeric_features = ['feature1', 'feature2', 'feature3']
categorical_features = ['category', 'region']
skewed_features = ['price', 'income']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('skewed', PowerTransformer(method='yeo-johnson'), skewed_features),
        ('categorical', OneHotEncoder(drop='first', sparse=False), categorical_features)
    ],
    remainder='passthrough'
)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', selector),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit and evaluate
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {score:.3f}")
```

## Practice Exercises

### Exercise 1: Feature Selection Comparison

**Task:** Select top 5 features using different methods and compare performance.

**Solution:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Method 1: Filter (F-test)
selector1 = SelectKBest(score_func=f_classif, k=5)
X1 = selector1.fit_transform(X, y)
model1 = LogisticRegression(random_state=42, max_iter=1000)
scores1 = cross_val_score(model1, X1, y, cv=5)
print(f"F-test selection: {scores1.mean():.3f} (+/- {scores1.std():.3f})")

# Method 2: Embedded (Lasso)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
selected2 = lasso.coef_ != 0
X2 = X[:, selected2]
model2 = LogisticRegression(random_state=42, max_iter=1000)
scores2 = cross_val_score(model2, X2, y, cv=5)
print(f"Lasso selection: {scores2.mean():.3f} (+/- {scores2.std():.3f})")

# Method 3: RFE
from sklearn.feature_selection import RFE
selector3 = RFE(LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=5)
X3 = selector3.fit_transform(X, y)
scores3 = cross_val_score(selector3.estimator, X3, y, cv=5)
print(f"RFE selection: {scores3.mean():.3f} (+/- {scores3.std():.3f})")

# Compare selected features
print("\nSelected features comparison:")
print(f"F-test: {feature_names[selector1.get_support()]}")
print(f"Lasso: {feature_names[selected2]}")
print(f"RFE: {feature_names[selector3.support_]}")
```

### Exercise 2: Feature Transformation Impact

**Task:** Compare different transformations and their impact on model performance.

**Solution:**
```python
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

transformations = {
    'Original': None,
    'Log': FunctionTransformer(np.log1p),
    'Square Root': FunctionTransformer(np.sqrt),
    'Box-Cox': PowerTransformer(method='box-cox'),
    'Yeo-Johnson': PowerTransformer(method='yeo-johnson'),
    'Quantile': QuantileTransformer(output_distribution='normal')
}

results = {}

for name, transformer in transformations.items():
    if transformer is None:
        X_transformed = X
    else:
        X_transformed = transformer.fit_transform(X)
    
    # Scale after transformation
    X_scaled = StandardScaler().fit_transform(X_transformed)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Find best transformation
best_transform = max(results, key=results.get)
print(f"\nBest transformation: {best_transform} ({results[best_transform]:.3f})")
```

### Exercise 3: Categorical Encoding Comparison

**Task:** Compare different encoding methods for categorical variables.

**Solution:**
```python
# Create sample data with categorical variable
df_cat = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
    'feature1': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Different encoding methods
encodings = {
    'One-Hot': OneHotEncoder(drop='first', sparse=False),
    'Label': LabelEncoder(),
    'Target': None  # Will implement manually
}

encoding_results = {}

for name, encoder in encodings.items():
    if name == 'One-Hot':
        X_encoded = encoder.fit_transform(df_cat[['category']])
        X_final = np.hstack([X_encoded, df_cat[['feature1']].values])
    elif name == 'Label':
        X_encoded = encoder.fit_transform(df_cat['category']).reshape(-1, 1)
        X_final = np.hstack([X_encoded, df_cat[['feature1']].values])
    else:  # Target encoding
        target_mean = df_cat.groupby('category')['target'].mean()
        X_encoded = df_cat['category'].map(target_mean).values.reshape(-1, 1)
        X_final = np.hstack([X_encoded, df_cat[['feature1']].values])
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(model, X_final, df_cat['target'], cv=5)
    encoding_results[name] = scores.mean()
    print(f"{name:15s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Feature Engineering Best Practices

### Do's

- Start with domain knowledge
- Create features that make sense
- Handle missing values before feature engineering
- Scale features appropriately
- Use cross-validation for target encoding
- Document all transformations
- Test feature importance

### Don'ts

- Don't create too many features (curse of dimensionality)
- Don't leak target information (data leakage)
- Don't overfit to validation set
- Don't ignore feature interactions
- Don't forget to scale after transformation
- Don't use test set for feature engineering decisions

---

## Key Takeaways

1. **Feature selection**: Remove irrelevant features using filter, wrapper, or embedded methods
2. **Transformation**: Handle skewed data with log, power, or other transformations
3. **Encoding**: Convert categorical to numeric (one-hot, label, target encoding)
4. **Scaling**: Required for distance-based algorithms (standardization, normalization, robust)
5. **Dimensionality reduction**: Use PCA for high-dimensional data
6. **Feature creation**: Domain knowledge is most valuable for creating features
7. **Pipeline**: Use pipelines to automate feature engineering

---

## Next Steps

- Practice with real datasets
- Experiment with different techniques
- Learn domain-specific feature engineering
- Move to [08-unsupervised-learning](../08-unsupervised-learning/README.md)

**Remember**: Good features beat complex algorithms! Invest time in feature engineering.

