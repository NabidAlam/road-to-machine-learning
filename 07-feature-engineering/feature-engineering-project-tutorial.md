# Complete Feature Engineering Project Tutorial

Step-by-step walkthrough of comprehensive feature engineering for a machine learning project.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Exploration](#step-1-data-loading-and-exploration)
- [Step 2: Handling Missing Values](#step-2-handling-missing-values)
- [Step 3: Feature Transformation](#step-3-feature-transformation)
- [Step 4: Categorical Encoding](#step-4-categorical-encoding)
- [Step 5: Feature Scaling](#step-5-feature-scaling)
- [Step 6: Creating New Features](#step-6-creating-new-features)
- [Step 7: Feature Selection](#step-7-feature-selection)
- [Step 8: Dimensionality Reduction](#step-8-dimensionality-reduction)
- [Step 9: Final Pipeline and Evaluation](#step-9-final-pipeline-and-evaluation)

---

## Project Overview

**Project**: Comprehensive Feature Engineering for Classification

**Dataset**: Adult Income Dataset (or any dataset with mixed data types)

**Goal**: Apply comprehensive feature engineering to improve model performance

**Type**: Classification with Feature Engineering

**Difficulty**: Intermediate

**Time**: 2-3 hours

---

## Step 1: Data Loading and Exploration

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                  OneHotEncoder, LabelEncoder, PowerTransformer)
from sklearn.feature_selection import (SelectKBest, f_classif, RFE, RFECV,
                                      mutual_info_classif)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data (example with synthetic data)
# In practice, load from: https://archive.ics.uci.edu/ml/datasets/adult
np.random.seed(42)
n_samples = 1000

# Create synthetic dataset
df = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'workclass': np.random.choice(['Private', 'Self-emp', 'Government'], n_samples),
    'education': np.random.choice(['HS', 'Bachelors', 'Masters', 'PhD'], n_samples),
    'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n_samples),
    'occupation': np.random.choice(['Tech', 'Sales', 'Management', 'Service'], n_samples),
    'hours_per_week': np.random.randint(20, 60, n_samples),
    'income': np.random.choice(['<=50K', '>50K'], n_samples),
    'capital_gain': np.random.exponential(1000, n_samples),
    'capital_loss': np.random.exponential(100, n_samples)
})

# Add some missing values
df.loc[df.sample(frac=0.1).index, 'workclass'] = np.nan
df.loc[df.sample(frac=0.05).index, 'occupation'] = np.nan

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
```

### Data Exploration

```python
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nCategorical Variables:")
print(df.select_dtypes(include=['object']).nunique())

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0, 0].hist(df['age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Capital gain (skewed)
axes[0, 1].hist(df['capital_gain'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Capital Gain Distribution (Skewed)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Capital Gain')

# Categorical distribution
df['income'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Income Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Income')
axes[1, 0].set_ylabel('Count')

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1],
            square=True, linewidths=0.5)
axes[1, 1].set_title('Feature Correlation', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Step 2: Handling Missing Values

### Analyze Missing Values

```python
print("Missing Values Analysis:")
print(df.isnull().sum())
print(f"\nMissing Percentage:")
print((df.isnull().sum() / len(df)) * 100)

# Visualize missing values
import missingno as msno
try:
    msno.matrix(df)
    plt.title('Missing Values Pattern', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install missingno: pip install missingno")
```

### Impute Missing Values

```python
# Strategy 1: Mode for categorical
df['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)

# Strategy 2: Median for numerical (if any)
# df['numeric_col'].fillna(df['numeric_col'].median(), inplace=True)

# Verify
print("Missing values after imputation:")
print(df.isnull().sum().sum())
```

---

## Step 3: Feature Transformation

### Identify Skewed Features

```python
from scipy.stats import skew

# Check skewness
numeric_features = df.select_dtypes(include=[np.number]).columns
skewness = {}

for col in numeric_features:
    skewness[col] = skew(df[col].dropna())

skew_df = pd.DataFrame({
    'Feature': list(skewness.keys()),
    'Skewness': list(skewness.values())
}).sort_values('Skewness', key=abs, ascending=False)

print("Feature Skewness:")
print(skew_df)

# Identify highly skewed features (|skew| > 1)
highly_skewed = skew_df[abs(skew_df['Skewness']) > 1]['Feature'].tolist()
print(f"\nHighly skewed features: {highly_skewed}")
```

### Apply Transformations

```python
# Transform skewed features
for col in highly_skewed:
    # Log transformation
    df[f'{col}_log'] = np.log1p(df[col])
    
    # Check improvement
    original_skew = skew(df[col].dropna())
    transformed_skew = skew(df[f'{col}_log'].dropna())
    print(f"{col}: {original_skew:.2f} -> {transformed_skew:.2f}")

# Power transformation for remaining skewed features
pt = PowerTransformer(method='yeo-johnson')
for col in highly_skewed:
    if col not in ['capital_gain', 'capital_loss']:  # Already transformed
        df[[f'{col}_power']] = pt.fit_transform(df[[col]])
```

### Binning Continuous Features

```python
# Bin age into groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 30, 45, 60, 100],
    labels=['Young', 'Adult', 'Middle', 'Senior']
)

# Bin hours per week
df['hours_group'] = pd.qcut(
    df['hours_per_week'],
    q=4,
    labels=['Part-time', 'Regular', 'Full-time', 'Overtime'],
    duplicates='drop'
)

print("Age groups:")
print(df['age_group'].value_counts())
print("\nHours groups:")
print(df['hours_group'].value_counts())
```

---

## Step 4: Categorical Encoding

### Prepare Categorical Variables

```python
# Identify categorical variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols.remove('income')  # Target variable

print("Categorical variables to encode:")
print(categorical_cols)

# Check cardinality
print("\nCardinality (number of unique values):")
for col in categorical_cols:
    print(f"  {col}: {df[col].nunique()}")
```

### Apply Different Encoding Methods

```python
# Method 1: One-Hot Encoding
df_ohe = df.copy()
df_ohe = pd.get_dummies(df_ohe, columns=categorical_cols, drop_first=True, prefix=categorical_cols)

print(f"Original columns: {len(df.columns)}")
print(f"After one-hot encoding: {len(df_ohe.columns)}")

# Method 2: Label Encoding (for tree models)
df_label = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_label[f'{col}_encoded'] = le.fit_transform(df_label[col].astype(str))
    label_encoders[col] = le

# Method 3: Target Encoding (with cross-validation)
from sklearn.model_selection import KFold

def target_encode_cv(df, cat_col, target_col, cv=5):
    """Target encoding with cross-validation"""
    df_encoded = df.copy()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Encode target first
    target_encoded = LabelEncoder().fit_transform(df[target_col])
    
    for train_idx, val_idx in kf.split(df):
        train_mean = pd.Series(target_encoded[train_idx]).groupby(
            df.iloc[train_idx][cat_col]
        ).mean()
        df_encoded.loc[val_idx, f'{cat_col}_target_encoded'] = \
            df.loc[val_idx, cat_col].map(train_mean)
    
    global_mean = target_encoded.mean()
    df_encoded[f'{cat_col}_target_encoded'].fillna(global_mean, inplace=True)
    return df_encoded

df_target = df.copy()
for col in categorical_cols:
    df_target = target_encode_cv(df_target, col, 'income', cv=5)
```

### Compare Encoding Methods

```python
# Prepare target
y = (df['income'] == '>50K').astype(int)

# Compare encoding methods
encoding_results = {}

# One-hot encoding
X_ohe = df_ohe.drop('income', axis=1).select_dtypes(include=[np.number])
model_ohe = LogisticRegression(random_state=42, max_iter=1000)
scores_ohe = cross_val_score(model_ohe, X_ohe, y, cv=5, scoring='accuracy')
encoding_results['One-Hot'] = scores_ohe.mean()

# Label encoding
X_label = df_label[categorical_cols + ['age', 'hours_per_week', 'capital_gain', 'capital_loss']]
X_label = X_label.select_dtypes(include=[np.number])
model_label = RandomForestClassifier(n_estimators=100, random_state=42)
scores_label = cross_val_score(model_label, X_label, y, cv=5, scoring='accuracy')
encoding_results['Label'] = scores_label.mean()

# Target encoding
X_target = df_target[[f'{col}_target_encoded' for col in categorical_cols] + 
                      ['age', 'hours_per_week', 'capital_gain', 'capital_loss']]
model_target = LogisticRegression(random_state=42, max_iter=1000)
scores_target = cross_val_score(model_target, X_target, y, cv=5, scoring='accuracy')
encoding_results['Target'] = scores_target.mean()

print("Encoding Method Comparison:")
for method, score in sorted(encoding_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method:15s}: {score:.3f}")

# Choose best encoding
best_encoding = max(encoding_results, key=encoding_results.get)
print(f"\nBest encoding method: {best_encoding}")
```

---

## Step 5: Feature Scaling

### Apply Scaling

```python
# Split data first
if best_encoding == 'One-Hot':
    X = df_ohe.drop('income', axis=1).select_dtypes(include=[np.number])
elif best_encoding == 'Label':
    X = df_label[categorical_cols + ['age', 'hours_per_week', 'capital_gain', 'capital_loss']]
    X = X.select_dtypes(include=[np.number])
else:
    X = df_target[[f'{col}_target_encoded' for col in categorical_cols] + 
                  ['age', 'hours_per_week', 'capital_gain', 'capital_loss']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare scaling methods
scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

scaling_results = {}

for name, scaler in scalers.items():
    if scaler is None:
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    scaling_results[name] = scores.mean()
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Choose best scaler
best_scaler_name = max(scaling_results, key=scaling_results.get)
print(f"\nBest scaler: {best_scaler_name}")

# Apply best scaler
if best_scaler_name != 'No Scaling':
    best_scaler = scalers[best_scaler_name]
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test
```

---

## Step 6: Creating New Features

### Domain Features

```python
# Create domain-specific features
df_features = df.copy()

# Age-related features
df_features['age_squared'] = df_features['age'] ** 2
df_features['is_senior'] = (df_features['age'] >= 65).astype(int)

# Work-related features
df_features['work_intensity'] = df_features['hours_per_week'] / 40  # Normalized to 40-hour week
df_features['is_full_time'] = (df_features['hours_per_week'] >= 40).astype(int)

# Financial features
df_features['net_capital'] = df_features['capital_gain'] - df_features['capital_loss']
df_features['capital_ratio'] = df_features['capital_gain'] / (df_features['capital_loss'] + 1)

# Interaction features
df_features['age_hours'] = df_features['age'] * df_features['hours_per_week']
df_features['age_education_interaction'] = df_features['age'].astype(str) + '_' + df_features['education']

print("New features created:")
new_features = [col for col in df_features.columns if col not in df.columns]
print(new_features)
```

### Aggregation Features

```python
# Group-based aggregations
# Example: Average hours by workclass
workclass_hours = df.groupby('workclass')['hours_per_week'].agg(['mean', 'std']).reset_index()
workclass_hours.columns = ['workclass', 'avg_hours_by_workclass', 'std_hours_by_workclass']
df_features = df_features.merge(workclass_hours, on='workclass', how='left')

# Education level encoding (ordinal)
education_order = {'HS': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
df_features['education_level'] = df_features['education'].map(education_order)

print(f"Total features after engineering: {len(df_features.columns)}")
```

---

## Step 7: Feature Selection

### Apply Feature Selection

```python
# Prepare final dataset with new features
# (Combine all engineered features)
X_final = df_features.drop('income', axis=1).select_dtypes(include=[np.number])
X_final = X_final.fillna(X_final.median())  # Handle any remaining NaN

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)

print(f"Features before selection: {X_train_final_scaled.shape[1]}")

# Method 1: Univariate selection
selector_kbest = SelectKBest(score_func=f_classif, k=15)
X_kbest = selector_kbest.fit_transform(X_train_final_scaled, y_train_final)
selected_features_kbest = X_final.columns[selector_kbest.get_support()]

# Method 2: RFE
selector_rfe = RFE(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    n_features_to_select=15
)
X_rfe = selector_rfe.fit_transform(X_train_final_scaled, y_train_final)
selected_features_rfe = X_final.columns[selector_rfe.support_]

# Method 3: RFECV (finds optimal number)
rfecv = RFECV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    step=1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(X_train_final_scaled, y_train_final)

print(f"\nOptimal number of features (RFECV): {rfecv.n_features_}")
selected_features_rfecv = X_final.columns[rfecv.support_]

# Compare methods
selection_results = {}

# KBest
model_kbest = LogisticRegression(random_state=42, max_iter=1000)
scores_kbest = cross_val_score(model_kbest, X_kbest, y_train_final, cv=5)
selection_results['KBest'] = scores_kbest.mean()

# RFE
model_rfe = LogisticRegression(random_state=42, max_iter=1000)
scores_rfe = cross_val_score(model_rfe, X_rfe, y_train_final, cv=5)
selection_results['RFE'] = scores_rfe.mean()

# RFECV
X_rfecv = rfecv.transform(X_train_final_scaled)
scores_rfecv = cross_val_score(rfecv.estimator_, X_rfecv, y_train_final, cv=5)
selection_results['RFECV'] = scores_rfecv.mean()

print("\nFeature Selection Comparison:")
for method, score in sorted(selection_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method:10s}: {score:.3f}")

# Plot RFECV results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, 'o-')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', 
            label=f'Optimal: {rfecv.n_features_}')
plt.xlabel('Number of Features Selected', fontsize=12)
plt.ylabel('Cross-Validated Accuracy', fontsize=12)
plt.title('RFECV Feature Selection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Step 8: Dimensionality Reduction

### Apply PCA

```python
# Apply PCA
pca = PCA()
pca.fit(X_train_final_scaled)

# Cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Apply PCA with optimal components
pca_optimal = PCA(n_components=0.95)
X_train_pca = pca_optimal.fit_transform(X_train_final_scaled)
X_test_pca = pca_optimal.transform(X_test_final_scaled)

print(f"Original dimensions: {X_train_final_scaled.shape[1]}")
print(f"PCA dimensions: {X_train_pca.shape[1]}")

# Compare with and without PCA
model_no_pca = LogisticRegression(random_state=42, max_iter=1000)
scores_no_pca = cross_val_score(model_no_pca, X_train_final_scaled, y_train_final, cv=5)

model_pca = LogisticRegression(random_state=42, max_iter=1000)
scores_pca = cross_val_score(model_pca, X_train_pca, y_train_final, cv=5)

print(f"\nWithout PCA: {scores_no_pca.mean():.3f} (+/- {scores_no_pca.std():.3f})")
print(f"With PCA: {scores_pca.mean():.3f} (+/- {scores_pca.std():.3f})")
```

---

## Step 9: Final Pipeline and Evaluation

### Build Complete Pipeline

```python
# Define feature types
numeric_features = ['age', 'hours_per_week', 'capital_gain', 'capital_loss']
categorical_features = categorical_cols

# Create complete pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough'
)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=15)

# Complete pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', selector),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit pipeline
final_pipeline.fit(X_train, y_train)

# Evaluate
train_score = final_pipeline.score(X_train, y_train)
test_score = final_pipeline.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Predictions
y_pred = final_pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Compare Before and After Feature Engineering

```python
# Baseline: Simple model without feature engineering
X_baseline = df[['age', 'hours_per_week']].fillna(df[['age', 'hours_per_week']].median())
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42, stratify=y
)

scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

model_baseline = LogisticRegression(random_state=42, max_iter=1000)
model_baseline.fit(X_train_base_scaled, y_train_base)
baseline_score = model_baseline.score(X_test_base_scaled, y_test_base)

print("Model Comparison:")
print(f"Baseline (2 features): {baseline_score:.3f}")
print(f"With Feature Engineering: {test_score:.3f}")
print(f"Improvement: {test_score - baseline_score:.3f}")
```

### Summary

```python
print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

print(f"\n1. Original features: {len(df.columns) - 1}")  # Exclude target
print(f"2. After encoding: {len(X_final.columns)}")
print(f"3. After feature selection: {selector.n_features_to_select}")
print(f"4. Final test accuracy: {test_score:.3f}")
print(f"5. Improvement over baseline: {test_score - baseline_score:.3f}")

print(f"\nKey Steps Completed:")
print(f"  - Missing value imputation")
print(f"  - Feature transformation (log, power)")
print(f"  - Categorical encoding ({best_encoding})")
print(f"  - Feature scaling ({best_scaler_name})")
print(f"  - Feature creation (domain features, interactions)")
print(f"  - Feature selection (RFECV)")
print(f"  - Dimensionality reduction (PCA)")
```

---

## Key Takeaways

1. **Data exploration**: Understand your data before engineering
2. **Handle missing values**: Impute appropriately
3. **Transform skewed features**: Log, power transformations
4. **Encode categoricals**: Choose method based on cardinality and model
5. **Scale features**: Required for distance-based algorithms
6. **Create features**: Domain knowledge is valuable
7. **Select features**: Remove irrelevant features
8. **Use pipelines**: Automate and prevent data leakage

---

**Congratulations!** You've completed comprehensive feature engineering!

