# Feature Engineering Quick Reference Guide

Quick reference for feature engineering techniques, code snippets, and best practices.

## Table of Contents

- [Feature Engineering Decision Tree](#feature-engineering-decision-tree)
- [Code Snippets](#code-snippets)
- [Feature Selection Methods](#feature-selection-methods)
- [Transformation Methods](#transformation-methods)
- [Encoding Methods](#encoding-methods)
- [Scaling Methods](#scaling-methods)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Feature Engineering Decision Tree

### Quick Decision Tree

```
Need to engineer features?
│
├─ Missing values?
│  ├─ YES → Impute (mean/median/mode) or drop
│  └─ NO → Continue
│
├─ Skewed distributions?
│  ├─ YES → Log/Power transformation
│  └─ NO → Continue
│
├─ Categorical variables?
│  ├─ Low cardinality (< 10) → One-Hot Encoding
│  ├─ High cardinality (> 10) → Target/Frequency Encoding
│  └─ Ordinal → Label Encoding
│
├─ Different scales?
│  ├─ YES → Standardize/Normalize
│  └─ NO → Continue
│
├─ Too many features?
│  ├─ YES → Feature Selection or PCA
│  └─ NO → Continue
│
└─ Need interactions?
   ├─ YES → Polynomial/Interaction Features
   └─ NO → Done
```

### Feature Engineering Workflow

```
1. Data Exploration
   ↓
2. Handle Missing Values
   ↓
3. Transform Skewed Features
   ↓
4. Encode Categorical Variables
   ↓
5. Scale Features
   ↓
6. Create New Features
   ↓
7. Feature Selection
   ↓
8. Dimensionality Reduction (if needed)
```

---

## Code Snippets

### Complete Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                  PowerTransformer, SimpleImputer)
from sklearn.feature_selection import SelectKBest, f_classif

# Define feature types
numeric_features = ['feature1', 'feature2']
categorical_features = ['category', 'region']
skewed_features = ['price', 'income']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('skewed', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ]), skewed_features),
        
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False))
        ]), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('model', LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)
```

### Missing Value Handling

```python
from sklearn.impute import SimpleImputer

# Strategy 1: Mean/Median for numeric
imputer_numeric = SimpleImputer(strategy='median')
X_numeric_imputed = imputer_numeric.fit_transform(X_numeric)

# Strategy 2: Mode for categorical
imputer_categorical = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = imputer_categorical.fit_transform(X_categorical)

# Strategy 3: Drop missing
df_clean = df.dropna()

# Strategy 4: Forward fill (time series)
df_filled = df.fillna(method='ffill')
```

---

## Feature Selection Methods

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# F-test (for numeric features)
selector_f = SelectKBest(score_func=f_classif, k=10)
X_selected = selector_f.fit_transform(X, y)

# Chi-square (for categorical features)
selector_chi2 = SelectKBest(score_func=chi2, k=10)
X_selected = selector_chi2.fit_transform(X_categorical, y)

# Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector_mi.fit_transform(X, y)
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV

# Recursive Feature Elimination
selector = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_selected = selector.fit_transform(X, y)

# RFE with Cross-Validation (finds optimal number)
rfecv = RFECV(estimator=LogisticRegression(), cv=5, scoring='accuracy')
rfecv.fit(X, y)
X_selected = rfecv.transform(X)
print(f"Optimal features: {rfecv.n_features_}")
```

### Embedded Methods

```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Lasso (L1 regularization)
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = lasso.coef_ != 0

# Tree-based importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X, y)
```

### Comparison Table

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Filter (F-test)** | Fast | Medium | Quick feature selection |
| **Filter (MI)** | Fast | Medium | Non-linear relationships |
| **RFE** | Slow | High | Model-specific selection |
| **RFECV** | Very Slow | Very High | Optimal feature count |
| **Lasso** | Medium | High | Sparse feature sets |
| **Tree-based** | Medium | High | Non-linear relationships |

---

## Transformation Methods

### Log Transformation

```python
import numpy as np

# Natural log (handles zeros)
df['log_feature'] = np.log1p(df['feature'])

# Base 10 log
df['log10_feature'] = np.log10(df['feature'] + 1)

# Square root (weaker transformation)
df['sqrt_feature'] = np.sqrt(df['feature'])
```

### Power Transformation

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (positive values only)
pt_boxcox = PowerTransformer(method='box-cox')
X_boxcox = pt_boxcox.fit_transform(X_positive)

# Yeo-Johnson (any values)
pt_yeo = PowerTransformer(method='yeo-johnson')
X_yeo = pt_yeo.fit_transform(X)
```

### Binning

```python
import pandas as pd

# Equal-width bins
df['binned'] = pd.cut(df['feature'], bins=5, labels=['Low', 'Med', 'High'])

# Equal-frequency bins
df['binned'] = pd.qcut(df['feature'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Custom bins
df['binned'] = pd.cut(df['feature'], bins=[0, 25, 50, 75, 100])
```

### When to Transform

| Transformation | When to Use | Example |
|----------------|-------------|---------|
| **Log** | Right-skewed, count data | Prices, counts |
| **Square Root** | Moderate skew | Count data |
| **Box-Cox** | Positive values, normal distribution needed | Income |
| **Yeo-Johnson** | Any values, normal distribution needed | General numeric |
| **Binning** | Non-linear relationships | Age groups |

---

## Encoding Methods

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Scikit-learn
encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df[['category']])

# Pandas (easier)
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# Single column
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Multiple columns
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df[['cat1', 'cat2']])
```

### Target Encoding

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, cat_col, target_col, cv=5):
    """Target encoding with cross-validation"""
    df_encoded = df.copy()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train_mean = df.iloc[train_idx].groupby(cat_col)[target_col].mean()
        df_encoded.loc[val_idx, f'{cat_col}_encoded'] = \
            df.loc[val_idx, cat_col].map(train_mean)
    
    global_mean = df[target_col].mean()
    df_encoded[f'{cat_col}_encoded'].fillna(global_mean, inplace=True)
    return df_encoded

df_encoded = target_encode_cv(df, 'category', 'target')
```

### Encoding Comparison

| Method | Cardinality | Pros | Cons | Code |
|--------|-------------|------|------|------|
| **One-Hot** | Low (< 10) | No assumptions, preserves info | Many features | `OneHotEncoder()` |
| **Label** | Any | One column, preserves order | Assumes ordinality | `LabelEncoder()` |
| **Target** | High (> 10) | Captures target relationship | Risk of overfitting | Custom function |
| **Frequency** | High | Simple, no overfitting | Loses category info | `value_counts()` |

---

## Scaling Methods

### Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transform new data
X_new_scaled = scaler.transform(X_new)
```

### Min-Max Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X)
```

### Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

### When to Scale

| Algorithm | Needs Scaling? | Recommended Method |
|-----------|----------------|-------------------|
| **KNN** | Yes | StandardScaler |
| **SVM** | Yes | StandardScaler |
| **Neural Networks** | Yes | StandardScaler or MinMaxScaler |
| **Logistic Regression** | Yes (with regularization) | StandardScaler |
| **PCA** | Yes | StandardScaler |
| **Decision Trees** | No | - |
| **Random Forest** | No | - |
| **XGBoost** | No | - |

---

## Common Issues & Solutions

### Issue 1: Data Leakage

**Problem**: Using future or target information in features

**Solution**:
```python
# WRONG: Using target to create features
df['target_mean'] = df.groupby('category')['target'].transform('mean')

# CORRECT: Use cross-validation
df_encoded = target_encode_cv(df, 'category', 'target', cv=5)
```

### Issue 2: Scaling Before Split

**Problem**: Fitting scaler on entire dataset

**Solution**:
```python
# WRONG
scaler.fit(X)  # Uses test data!
X_train, X_test = train_test_split(X)

# CORRECT
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # Only fit on training
X_test_scaled = scaler.transform(X_test)
```

### Issue 3: Too Many Features

**Problem**: Curse of dimensionality, overfitting

**Solution**:
```python
# Use feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Or use regularization
model = LogisticRegression(C=0.1)  # Lower C = more regularization
```

### Issue 4: High Cardinality Categoricals

**Problem**: One-hot encoding creates too many features

**Solution**:
```python
# Use target encoding or frequency encoding
df_encoded = target_encode_cv(df, 'high_card_category', 'target')

# Or group rare categories
df['category_grouped'] = df['category'].replace(
    df['category'].value_counts()[df['category'].value_counts() < 10].index,
    'Other'
)
```

### Issue 5: Skewed Features

**Problem**: Model assumptions violated, poor performance

**Solution**:
```python
# Check skewness
from scipy.stats import skew
skewness = skew(df['feature'])

# Transform if |skew| > 1
if abs(skewness) > 1:
    df['feature_log'] = np.log1p(df['feature'])
    # Or use PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    df[['feature_transformed']] = pt.fit_transform(df[['feature']])
```

---

## Best Practices Checklist

### Data Preparation
- [ ] Handle missing values appropriately
- [ ] Check for and handle outliers
- [ ] Verify data types are correct
- [ ] Check for duplicate rows

### Feature Transformation
- [ ] Identify and transform skewed features
- [ ] Apply appropriate transformations (log, power, etc.)
- [ ] Verify transformation improved distribution
- [ ] Consider binning for non-linear relationships

### Categorical Encoding
- [ ] Choose encoding method based on cardinality
- [ ] Use cross-validation for target encoding
- [ ] Handle unknown categories in test set
- [ ] Consider frequency encoding for high cardinality

### Feature Scaling
- [ ] Scale features for distance-based algorithms
- [ ] Fit scaler only on training data
- [ ] Choose appropriate scaling method
- [ ] Verify scaling worked correctly

### Feature Creation
- [ ] Create domain-specific features
- [ ] Create interaction features
- [ ] Create aggregation features
- [ ] Document all created features

### Feature Selection
- [ ] Remove highly correlated features
- [ ] Apply feature selection methods
- [ ] Compare multiple selection methods
- [ ] Verify selected features improve performance

### Pipeline
- [ ] Create preprocessing pipeline
- [ ] Use ColumnTransformer for mixed types
- [ ] Prevent data leakage
- [ ] Test pipeline on new data

### Validation
- [ ] Use cross-validation for feature engineering decisions
- [ ] Separate train/validation/test sets
- [ ] Compare before/after feature engineering
- [ ] Document improvements

---

## Quick Code Templates

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False))
        ]), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
```

### Feature Selection Template

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = feature_names[selector.get_support()]
```

### Transformation Template

```python
from sklearn.preprocessing import PowerTransformer

# Check skewness
from scipy.stats import skew
if abs(skew(df['feature'])) > 1:
    pt = PowerTransformer(method='yeo-johnson')
    df[['feature_transformed']] = pt.fit_transform(df[['feature']])
```

---

## Key Takeaways

1. **Always split data before feature engineering** to prevent leakage
2. **Scale features** for distance-based algorithms
3. **Transform skewed features** to meet model assumptions
4. **Choose encoding method** based on cardinality and model type
5. **Use pipelines** to automate and prevent errors
6. **Validate feature engineering** with cross-validation
7. **Document all transformations** for reproducibility

---

## Next Steps

- Practice with real datasets
- Experiment with different techniques
- Learn domain-specific feature engineering
- Move to next module

**Remember**: Good features beat complex algorithms!

