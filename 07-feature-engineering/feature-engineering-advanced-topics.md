# Advanced Feature Engineering Topics

Comprehensive guide to advanced feature engineering techniques, automation, and best practices.

## Table of Contents

- [Automated Feature Engineering](#automated-feature-engineering)
- [Feature Interactions and Polynomial Features](#feature-interactions-and-polynomial-features)
- [Time Series Feature Engineering](#time-series-feature-engineering)
- [Text Feature Engineering](#text-feature-engineering)
- [Feature Engineering Pipelines](#feature-engineering-pipelines)
- [Feature Importance and Selection](#feature-importance-and-selection)
- [Common Feature Engineering Pitfalls](#common-feature-engineering-pitfalls)

---

## Automated Feature Engineering

### Featuretools

Automated feature engineering library.

```python
try:
    import featuretools as ft
    
    # Create entity set
    es = ft.EntitySet(id='data')
    
    # Add entities
    es = es.entity_from_dataframe(
        entity_id='customers',
        dataframe=df_customers,
        index='customer_id'
    )
    
    es = es.entity_from_dataframe(
        entity_id='orders',
        dataframe=df_orders,
        index='order_id',
        time_index='order_date'
    )
    
    # Define relationships
    es = es.add_relationship(
        ft.Relationship(es['customers']['customer_id'],
                       es['orders']['customer_id'])
    )
    
    # Generate features
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_entity='customers',
        max_depth=2,
        verbose=1
    )
    
    print(f"Generated {len(feature_defs)} features")
    
except ImportError:
    print("Install featuretools: pip install featuretools")
```

### AutoFeat

Automatic feature engineering and selection.

```python
try:
    from autofeat import AutoFeatClassifier
    
    # Automatic feature engineering
    autofeat = AutoFeatClassifier(verbose=1)
    X_new = autofeat.fit_transform(X, y)
    
    print(f"Original features: {X.shape[1]}")
    print(f"New features: {X_new.shape[1]}")
    
except ImportError:
    print("Install autofeat: pip install autofeat")
```

---

## Feature Interactions and Polynomial Features

### Manual Interaction Features

```python
# Create interaction features manually
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-6)
df['feature1_plus_feature2'] = df['feature1'] + df['feature2']
df['feature1_minus_feature2'] = df['feature1'] - df['feature2']
df['feature1_power_feature2'] = df['feature1'] ** df['feature2']

# Ratio features
df['ratio_1_2'] = df['feature1'] / (df['feature2'] + 1e-6)
df['ratio_2_1'] = df['feature2'] / (df['feature1'] + 1e-6)

# Difference features
df['diff_1_2'] = df['feature1'] - df['feature2']
df['abs_diff_1_2'] = np.abs(df['feature1'] - df['feature2'])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Full polynomial features (degree 2)
poly_full = PolynomialFeatures(degree=2, include_bias=False)
X_poly_full = poly_full.fit_transform(X)

print(f"Original: {X.shape[1]} features")
print(f"Polynomial: {X_poly_full.shape[1]} features")

# Interaction-only (no squared terms)
poly_interaction = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
X_poly_interaction = poly_interaction.fit_transform(X)

print(f"Interaction-only: {X_poly_interaction.shape[1]} features")

# Compare performance
from sklearn.linear_model import LogisticRegression

models = {
    'Original': LogisticRegression(random_state=42, max_iter=1000),
    'Polynomial': LogisticRegression(random_state=42, max_iter=1000),
    'Interaction': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}
results['Original'] = cross_val_score(
    models['Original'], X, y, cv=5
).mean()

results['Polynomial'] = cross_val_score(
    models['Polynomial'], X_poly_full, y, cv=5
).mean()

results['Interaction'] = cross_val_score(
    models['Interaction'], X_poly_interaction, y, cv=5
).mean()

for name, score in results.items():
    print(f"{name:15s}: {score:.3f}")
```

### Feature Crosses

```python
# Create feature crosses (combinations)
# Example: Combine categorical features
df['category_region_cross'] = df['category'].astype(str) + '_' + df['region'].astype(str)

# One-hot encode the cross
df_cross_encoded = pd.get_dummies(df, columns=['category_region_cross'], prefix='cross')
```

---

## Time Series Feature Engineering

### Temporal Features

```python
# Extract time components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['week'] = df['timestamp'].dt.isocalendar().week
df['quarter'] = df['timestamp'].dt.quarter
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# Cyclical encoding (for periodic features)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Boolean features
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_start'] = df['day'] <= 3
df['is_month_end'] = df['day'] >= 28
df['is_holiday'] = df['date'].isin(holidays).astype(int)
```

### Lag Features

```python
# Lag features (previous values)
df['value_lag_1'] = df['value'].shift(1)
df['value_lag_7'] = df['value'].shift(7)  # Weekly lag
df['value_lag_30'] = df['value'].shift(30)  # Monthly lag

# Difference features
df['value_diff_1'] = df['value'].diff(1)
df['value_diff_7'] = df['value'].diff(7)

# Percentage change
df['value_pct_change'] = df['value'].pct_change()
df['value_pct_change_7'] = df['value'].pct_change(7)
```

### Rolling Statistics

```python
# Rolling window statistics
window_sizes = [3, 7, 14, 30]

for window in window_sizes:
    df[f'value_rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
    df[f'value_rolling_std_{window}'] = df['value'].rolling(window=window).std()
    df[f'value_rolling_min_{window}'] = df['value'].rolling(window=window).min()
    df[f'value_rolling_max_{window}'] = df['value'].rolling(window=window).max()
    df[f'value_rolling_median_{window}'] = df['value'].rolling(window=window).median()

# Expanding window statistics
df['value_expanding_mean'] = df['value'].expanding().mean()
df['value_expanding_std'] = df['value'].expanding().std()
```

### Time-Based Aggregations

```python
# Group by time periods
df['date'] = pd.to_datetime(df['timestamp'])

# Daily aggregations
daily_stats = df.groupby(df['date'].dt.date).agg({
    'value': ['mean', 'std', 'min', 'max', 'count']
})

# Weekly aggregations
df['week'] = df['date'].dt.to_period('W')
weekly_stats = df.groupby('week')['value'].agg(['mean', 'std', 'min', 'max'])

# Monthly aggregations
df['month'] = df['date'].dt.to_period('M')
monthly_stats = df.groupby('month')['value'].agg(['mean', 'std', 'min', 'max'])

# Merge back
df = df.merge(weekly_stats, left_on='week', right_index=True, suffixes=('', '_weekly'))
```

---

## Text Feature Engineering

### Basic Text Features

```python
# Length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['sentence_count'] = df['text'].str.count(r'[.!?]+')
df['paragraph_count'] = df['text'].str.count('\n\n') + 1

# Average features
df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1)

# Character features
df['uppercase_count'] = df['text'].str.findall(r'[A-Z]').str.len()
df['lowercase_count'] = df['text'].str.findall(r'[a-z]').str.len()
df['digit_count'] = df['text'].str.findall(r'[0-9]').str.len()
df['special_char_count'] = df['text'].str.findall(r'[^a-zA-Z0-9\s]').str.len()
```

### N-gram Features

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Unigrams and bigrams
vectorizer = CountVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=1000
)
X_text = vectorizer.fit_transform(df['text'])

# TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=1000
)
X_tfidf = tfidf.fit_transform(df['text'])
```

### Text Statistics

```python
# Readability features
df['avg_syllables_per_word'] = df['text'].apply(count_syllables) / df['word_count']

# Sentiment features (if using textblob)
try:
    from textblob import TextBlob
    
    df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
except ImportError:
    print("Install textblob: pip install textblob")
```

---

## Comprehensive sklearn Pipeline and ColumnTransformer Guide

### Introduction to Pipelines

Pipelines automate the ML workflow and prevent data leakage by ensuring transformations are applied consistently.

**Benefits:**
- Prevents data leakage
- Ensures consistent transformations
- Simplifies code
- Makes models reproducible
- Easy to deploy

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)
```

### ColumnTransformer Basics

ColumnTransformer applies different transformations to different columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define column types
numeric_cols = ['age', 'income', 'score']
categorical_cols = ['city', 'category']

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
    ],
    remainder='drop'  # What to do with remaining columns
)

# Use in pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
```

### Advanced ColumnTransformer

**Multiple Transformations per Column Type:**
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Complex preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        # Numeric: impute then scale
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['age', 'income']),
        
        # Skewed numeric: impute, transform, scale
        ('skewed', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('scaler', RobustScaler())
        ]), ['price', 'revenue']),
        
        # Categorical: impute then encode
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ]), ['city', 'category']),
        
        # Binary: just encode
        ('binary', OneHotEncoder(drop='if_binary', sparse=False), ['is_active'])
    ],
    remainder='passthrough'  # Keep other columns
)
```

### Custom Transformers

**Creating Custom Transformers:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transformation with error handling"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(np.maximum(X, 0))  # Handle negative values
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class FeatureInteraction(BaseEstimator, TransformerMixin):
    """Create interaction features"""
    
    def __init__(self, interactions):
        self.interactions = interactions  # List of (i, j) tuples
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        for i, j in self.interactions:
            X_new = np.hstack([X_new, (X[:, i] * X[:, j]).reshape(-1, 1)])
        return X_new

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding with cross-validation"""
    
    def __init__(self, cv=5):
        self.cv = cv
        self.encodings = {}
    
    def fit(self, X, y):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Calculate encodings for each fold
        for train_idx, val_idx in kf.split(X):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            for category in np.unique(X_train_fold):
                mask = X_train_fold == category
                if category not in self.encodings:
                    self.encodings[category] = []
                self.encodings[category].append(y_train_fold[mask].mean())
        
        # Average across folds
        for cat in self.encodings:
            self.encodings[cat] = np.mean(self.encodings[cat])
        
        return self
    
    def transform(self, X):
        return np.array([self.encodings.get(cat, 0) for cat in X])
```

### Complex Pipeline Example

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# Step 1: Preprocessing
preprocessing = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False))
        ]), categorical_features)
    ]
)

# Step 2: Feature Engineering
feature_engineering = FeatureUnion([
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('interactions', FeatureInteraction([(0, 1), (0, 2)]))
])

# Step 3: Feature Selection
feature_selection = SelectKBest(score_func=f_classif, k=20)

# Step 4: Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Complete pipeline
full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('feature_engineering', feature_engineering),
    ('feature_selection', feature_selection),
    ('model', model)
])

# Fit and evaluate
full_pipeline.fit(X_train, y_train)
score = full_pipeline.score(X_test, y_test)
```

### Pipeline Best Practices

**1. Always Use Pipelines:**
```python
# Bad: Manual transformations
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Easy to forget!
model.fit(X_train_scaled, y_train)

# Good: Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
pipeline.fit(X_train, y_train)  # Automatically handles train/test
```

**2. Get Feature Names:**
```python
# After fitting, get feature names
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
print(f"Feature names: {feature_names}")
```

**3. Access Intermediate Steps:**
```python
# Access fitted transformers
scaler = pipeline.named_steps['scaler']
print(f"Mean: {scaler.mean_}")
print(f"Scale: {scaler.scale_}")
```

**4. Grid Search with Pipelines:**
```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'preprocessor__num__scaler': [StandardScaler(), RobustScaler()],
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 15]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best score: {grid_search.best_score_:.3f}")
print(f"Best params: {grid_search.best_params_}")
```

### Common Pipeline Patterns

**Pattern 1: Simple Preprocessing + Model**
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

**Pattern 2: Mixed Data Types**
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

**Pattern 3: Feature Engineering + Selection + Model**
```python
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('selector', SelectKBest(k=10)),
    ('model', LogisticRegression())
])
```

### Debugging Pipelines

```python
# Check pipeline steps
print("Pipeline steps:", pipeline.named_steps.keys())

# Check intermediate outputs
X_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
print(f"Transformed shape: {X_transformed.shape}")

# Visualize pipeline
from sklearn.utils import estimator_html_repr
print(estimator_html_repr(pipeline))
```

---

## Feature Engineering Pipelines

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                  PowerTransformer, FunctionTransformer)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# Define feature types
numeric_features = ['feature1', 'feature2', 'feature3']
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
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='drop'
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

# Get feature names
feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
selected_features = feature_names_out[pipeline.named_steps['feature_selection'].get_support()]
print(f"Selected features: {selected_features}")
```

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformer"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

class InteractionTransformer(BaseEstimator, TransformerMixin):
    """Create interaction features"""
    
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        interactions = []
        for i, j in self.feature_indices:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack([X] + interactions)

# Use in pipeline
interaction_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('interactions', InteractionTransformer([(0, 1), (0, 2)])),
    ('model', LogisticRegression(random_state=42))
])
```

---

## Feature Importance and Selection

### Multiple Selection Methods

```python
from sklearn.feature_selection import (SelectKBest, RFE, RFECV,
                                      SelectFromModel, mutual_info_classif)
from sklearn.ensemble import RandomForestClassifier

# Method 1: Univariate selection
selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_kbest = selector_kbest.fit_transform(X, y)

# Method 2: Mutual information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi = selector_mi.fit_transform(X, y)

# Method 3: RFE
selector_rfe = RFE(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=10
)
X_rfe = selector_rfe.fit_transform(X, y)

# Method 4: Model-based selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
selector_model = SelectFromModel(rf, threshold='median')
X_model = selector_model.fit_transform(X, y)

# Compare methods
methods = {
    'KBest (F-test)': (X_kbest, selector_kbest),
    'Mutual Information': (X_mi, selector_mi),
    'RFE': (X_rfe, selector_rfe),
    'Model-based': (X_model, selector_model)
}

for name, (X_selected, selector) in methods.items():
    model = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(model, X_selected, y, cv=5)
    print(f"{name:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Feature Importance Analysis

```python
# Multiple importance methods
importance_methods = {}

# Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance_methods['Random Forest'] = rf.feature_importances_

# XGBoost importance
try:
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X, y)
    importance_methods['XGBoost'] = xgb_model.feature_importances_
except ImportError:
    pass

# Permutation importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
importance_methods['Permutation'] = perm_importance.importances_mean

# Compare
importance_df = pd.DataFrame(importance_methods, index=feature_names)
importance_df = importance_df.sort_values('Random Forest', ascending=False)

print("Feature Importance Comparison:")
print(importance_df.head(10))
```

---

## Common Feature Engineering Pitfalls

### Pitfall 1: Data Leakage

**Problem:** Using future information or target information in features

**Solution:**
```python
# WRONG: Using target to create features
df['target_mean_by_category'] = df.groupby('category')['target'].transform('mean')

# CORRECT: Use cross-validation for target encoding
def target_encode_cv(df, cat_col, target_col, cv=5):
    """Target encoding with cross-validation"""
    df_encoded = df.copy()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train_mean = df.iloc[train_idx].groupby(cat_col)[target_col].mean()
        df_encoded.loc[val_idx, f'{cat_col}_target_encoded'] = \
            df.loc[val_idx, cat_col].map(train_mean)
    
    global_mean = df[target_col].mean()
    df_encoded[f'{cat_col}_target_encoded'].fillna(global_mean, inplace=True)
    return df_encoded
```

### Pitfall 2: Scaling Before Split

**Problem:** Fitting scaler on entire dataset including test set

**Solution:**
```python
# WRONG: Scale before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data!
X_train, X_test = train_test_split(X_scaled, y)

# CORRECT: Scale after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform!
```

### Pitfall 3: Too Many Features

**Problem:** Creating too many features causes overfitting

**Solution:**
```python
# Use feature selection
selector = SelectKBest(score_func=f_classif, k=20)  # Limit features
X_selected = selector.fit_transform(X, y)

# Or use regularization
model = LogisticRegression(C=0.1, random_state=42)  # Lower C = more regularization
```

### Pitfall 4: Ignoring Feature Interactions

**Problem:** Missing important feature relationships

**Solution:**
```python
# Create interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_interaction = poly.fit_transform(X)

# Or manually create important interactions
df['important_interaction'] = df['feature1'] * df['feature2']
```

---

## Key Takeaways

1. **Automation**: Use tools like Featuretools for automated feature engineering
2. **Interactions**: Create interaction and polynomial features
3. **Time Series**: Extract temporal features, lags, and rolling statistics
4. **Text Features**: Extract length, n-grams, and sentiment features
5. **Pipelines**: Use pipelines to automate and prevent data leakage
6. **Feature Selection**: Use multiple methods to find best features
7. **Avoid Pitfalls**: Watch for data leakage, scaling issues, and overfitting

---

## Next Steps

- Practice with domain-specific feature engineering
- Learn automated feature engineering tools
- Experiment with feature interactions
- Move to unsupervised learning module

**Remember**: Feature engineering is both art and science - domain knowledge is invaluable!

