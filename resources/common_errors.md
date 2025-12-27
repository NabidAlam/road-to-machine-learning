# Common ML Errors & Debugging Guide

A practical guide to identifying, understanding, and fixing common errors in machine learning projects.

## Table of Contents

- [Data-Related Errors](#data-related-errors)
- [Model Training Errors](#model-training-errors)
- [Evaluation Errors](#evaluation-errors)
- [Deployment Errors](#deployment-errors)
- [Performance Issues](#performance-issues)
- [Debugging Strategies](#debugging-strategies)
- [Error Prevention](#error-prevention)

---

## Data-Related Errors

### Error: "ValueError: Input contains NaN"

**Cause**: Missing values in your dataset

**Solution**:
```python
# Check for missing values
print(df.isnull().sum())

# Option 1: Drop rows with missing values
df = df.dropna()

# Option 2: Fill missing values
df = df.fillna(df.mean())  # For numerical
df = df.fillna(df.mode().iloc[0])  # For categorical

# Option 3: Use imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)
```

### Error: "ValueError: Found array with 0 sample(s)"

**Cause**: Empty dataset or incorrect data splitting

**Solution**:
```python
# Check data shape
print(X.shape, y.shape)

# Verify data is not empty
assert len(X) > 0, "X is empty"
assert len(y) > 0, "y is empty"

# Check train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### Error: "ValueError: could not convert string to float"

**Cause**: Categorical data not encoded

**Solution**:
```python
# Check data types
print(df.dtypes)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Or use one-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'])
```

### Error: "ValueError: Input X has wrong number of features"

**Cause**: Feature mismatch between training and prediction

**Solution**:
```python
# Ensure same features during training and prediction
# Save feature names during training
feature_names = X_train.columns.tolist()

# During prediction, ensure same features
X_pred = X_pred[feature_names]  # Reorder if needed
X_pred = X_pred.reindex(columns=feature_names, fill_value=0)
```

---

## Model Training Errors

### Error: "ConvergenceWarning: Maximum number of iterations reached"

**Cause**: Model didn't converge within iterations

**Solution**:
```python
# Increase max_iter
model = LogisticRegression(max_iter=1000)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# Check for data issues
print("Data shape:", X.shape)
print("Unique labels:", np.unique(y))
```

### Error: "ValueError: Unknown label type: 'unknown'"

**Cause**: Wrong data type for target variable

**Solution**:
```python
# Convert to numeric
y = y.astype(int)
# or
y = pd.to_numeric(y, errors='coerce')

# For classification, ensure labels are 0-indexed
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### Error: "RuntimeError: CUDA out of memory"

**Cause**: GPU memory insufficient for model/data size

**Solution**:
```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=32)  # Reduce from 64

# Use CPU instead
device = torch.device('cpu')

# Clear cache
torch.cuda.empty_cache()

# Use gradient accumulation
# Train with smaller batches, accumulate gradients
```

### Error: "ValueError: Classification metrics can't handle a mix of binary and continuous targets"

**Cause**: Using classification metrics with regression model or vice versa

**Solution**:
```python
# For classification
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)  # Returns classes
accuracy_score(y_test, y_pred)

# For regression
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)  # Returns continuous values
mean_squared_error(y_test, y_pred)
```

---

## Evaluation Errors

### Error: "ValueError: Found input variables with inconsistent numbers of samples"

**Cause**: Mismatch between X and y sizes

**Solution**:
```python
# Check shapes
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Ensure same length
assert len(X) == len(y), "X and y must have same length"

# Reset indices if needed
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
```

### Error: "ValueError: Number of classes must be greater than one"

**Cause**: Only one class in target variable

**Solution**:
```python
# Check unique values
print("Unique classes:", np.unique(y))
print("Class distribution:", pd.Series(y).value_counts())

# Ensure multiple classes
if len(np.unique(y)) < 2:
    print("Error: Need at least 2 classes")
    # Check data collection or filtering
```

### Error: "UndefinedMetricWarning: Precision is ill-defined"

**Cause**: No positive predictions or no positive samples

**Solution**:
```python
# Check predictions
print("Unique predictions:", np.unique(y_pred))
print("Unique actual:", np.unique(y_test))

# Handle edge cases
from sklearn.metrics import precision_score, recall_score
try:
    precision = precision_score(y_test, y_pred, zero_division=0)
except:
    precision = 0.0
```

---

## Deployment Errors

### Error: "ModuleNotFoundError: No module named 'sklearn'"

**Cause**: Missing dependencies in deployment environment

**Solution**:
```python
# Create requirements.txt
pip freeze > requirements.txt

# Include all dependencies
# requirements.txt:
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0

# Install in deployment
pip install -r requirements.txt
```

### Error: "ValueError: Feature names mismatch"

**Cause**: Different feature order or names during prediction

**Solution**:
```python
# Save feature names during training
import joblib
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Load and use during prediction
feature_names = joblib.load('feature_names.pkl')
X_pred = X_pred[feature_names]  # Reorder columns
```

### Error: "TypeError: Object of type float32 is not JSON serializable"

**Cause**: NumPy types not JSON serializable

**Solution**:
```python
import json
import numpy as np

# Convert to native Python types
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Use in API
prediction = float(model.predict([features])[0])
return json.dumps({"prediction": prediction})
```

---

## Performance Issues

### Issue: Model Training is Very Slow

**Diagnosis**:
```python
import time

start = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - start} seconds")
```

**Solutions**:
```python
# 1. Reduce data size (for testing)
X_sample = X_train.sample(10000)
y_sample = y_train.loc[X_sample.index]

# 2. Use faster algorithms
from sklearn.linear_model import SGDClassifier  # Faster than LogisticRegression

# 3. Reduce features
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=100)
X_selected = selector.fit_transform(X_train, y_train)

# 4. Use parallel processing
model = RandomForestClassifier(n_jobs=-1)  # Use all CPUs
```

### Issue: Memory Error

**Solutions**:
```python
# 1. Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)

# 2. Use data types efficiently
df['column'] = df['column'].astype('category')  # Saves memory
df['int_col'] = df['int_col'].astype('int32')  # Instead of int64

# 3. Delete unused variables
del large_dataframe
import gc
gc.collect()
```

### Issue: Overfitting

**Symptoms**: High training accuracy, low validation accuracy

**Solutions**:
```python
# 1. Add regularization
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # L2 regularization

# 2. Reduce model complexity
model = RandomForestClassifier(max_depth=5, n_estimators=50)

# 3. Use more data
# 4. Feature selection
# 5. Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

---

## Debugging Strategies

### 1. Data Inspection

```python
# Always inspect your data first
print("Data shape:", df.shape)
print("Data types:", df.dtypes)
print("Missing values:", df.isnull().sum())
print("First few rows:", df.head())
print("Summary statistics:", df.describe())
```

### 2. Step-by-Step Execution

```python
# Break down complex operations
# Instead of:
result = complex_function(data)

# Do:
step1 = prepare_data(data)
print("Step 1 done:", step1.shape)
step2 = process_features(step1)
print("Step 2 done:", step2.shape)
result = train_model(step2)
```

### 3. Use Assertions

```python
# Validate assumptions
assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
assert not X.isnull().any().any(), "X should not have NaN values"
assert len(np.unique(y)) > 1, "y must have at least 2 classes"
```

### 4. Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Training model with {X.shape[0]} samples")
logger.info(f"Model parameters: {model.get_params()}")
logger.info(f"Training accuracy: {train_score}")
```

### 5. Visualization

```python
# Visualize data distribution
import matplotlib.pyplot as plt

plt.hist(y)
plt.title("Target Distribution")
plt.show()

# Visualize predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

---

## Error Prevention

### Best Practices

1. **Data Validation**
```python
def validate_data(X, y):
    assert X.shape[0] == y.shape[0]
    assert not X.isnull().any().any()
    assert len(np.unique(y)) > 1
    return True
```

2. **Type Checking**
```python
def process_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
    # Process data
```

3. **Error Handling**
```python
try:
    model.fit(X_train, y_train)
except ValueError as e:
    print(f"Error in training: {e}")
    # Handle error appropriately
```

4. **Unit Tests**
```python
def test_model_training():
    X, y = make_classification(n_samples=100, n_features=10)
    model = LogisticRegression()
    model.fit(X, y)
    assert model.score(X, y) > 0.5
```

5. **Version Control**
- Track data versions
- Track model versions
- Document changes

---

## Common Patterns

### Pattern 1: Data Pipeline

```python
def data_pipeline(raw_data):
    # 1. Load
    df = pd.read_csv(raw_data)
    
    # 2. Validate
    assert not df.empty, "Data is empty"
    
    # 3. Clean
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 4. Transform
    df = encode_categorical(df)
    df = scale_features(df)
    
    # 5. Split
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    return X_train, X_test, y_train, y_test
```

### Pattern 2: Model Training with Error Handling

```python
def train_model_safe(X_train, y_train):
    try:
        # Validate inputs
        assert X_train.shape[0] > 0
        assert len(np.unique(y_train)) > 1
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        print(f"Training failed: {e}")
        return None
```

---

## Quick Debugging Checklist

- [ ] Check data shapes match
- [ ] Verify no missing values
- [ ] Check data types are correct
- [ ] Ensure features are scaled/normalized
- [ ] Verify target variable format
- [ ] Check for class imbalance
- [ ] Validate train/test split
- [ ] Check model parameters
- [ ] Verify evaluation metrics match problem type
- [ ] Check for memory issues
- [ ] Validate feature names match

---

**Remember**: Most errors come from data issues. Always inspect and validate your data first!

