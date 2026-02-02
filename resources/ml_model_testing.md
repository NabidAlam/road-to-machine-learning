# ML Model Testing Guide

Comprehensive guide to testing machine learning models, pipelines, and APIs for production readiness.

## Table of Contents

- [Introduction](#introduction)
- [Unit Testing for ML](#unit-testing-for-ml)
- [Integration Testing](#integration-testing)
- [Model Testing Strategies](#model-testing-strategies)
- [Testing Data Preprocessing](#testing-data-preprocessing-functions)
- [Testing Model Training](#testing-model-training)
- [Testing Predictions](#testing-predictions)
- [Testing APIs](#testing-apis)
- [Property-Based Testing](#property-based-testing)
- [Best Practices](#best-practices)

---

## Introduction

### Why Test ML Models?

Testing ML code is crucial because:
- **Reproducibility**: Ensure models behave consistently
- **Regression Prevention**: Catch bugs before deployment
- **Confidence**: Deploy with confidence
- **Maintainability**: Easier to refactor and improve
- **Documentation**: Tests serve as usage examples

### Challenges in ML Testing

- **Non-determinism**: Random seeds, data shuffling
- **Data dependencies**: Need test data that represents production
- **Model complexity**: Hard to test black-box models
- **Performance metrics**: Need to test model quality, not just correctness

---

## Unit Testing for ML

### Testing Data Preprocessing Functions

```python
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_standard_scaler():
    """Test that StandardScaler normalizes data correctly"""
    # Arrange
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    
    # Act
    X_scaled = scaler.fit_transform(X)
    
    # Assert
    assert X_scaled.mean(axis=0).all() < 1e-10  # Mean should be ~0
    assert np.allclose(X_scaled.std(axis=0), 1.0)  # Std should be 1.0
    assert X_scaled.shape == X.shape

def test_handle_missing_values():
    """Test missing value imputation"""
    # Arrange
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, 20, 30, np.nan, 50]
    })
    
    # Act
    df_filled = df.fillna(df.mean())
    
    # Assert
    assert df_filled.isna().sum().sum() == 0
    assert df_filled['feature1'].iloc[2] == df['feature1'].mean()
    assert df_filled['feature2'].iloc[3] == df['feature2'].mean()

def test_feature_engineering():
    """Test feature creation functions"""
    # Arrange
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
    })
    
    # Act
    df['age_squared'] = df['age'] ** 2
    df['income_per_age'] = df['income'] / df['age']
    
    # Assert
    assert 'age_squared' in df.columns
    assert 'income_per_age' in df.columns
    assert df['age_squared'].iloc[0] == 625
    assert df['income_per_age'].iloc[0] == 2000.0
```

### Testing Model Training

```python
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_model_training():
    """Test that model trains successfully"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Act
    model.fit(X_train, y_train)
    
    # Assert
    assert hasattr(model, 'feature_importances_')
    assert model.n_estimators == 10
    assert len(model.estimators_) == 10

def test_model_predictions_shape():
    """Test that predictions have correct shape"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Act
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Assert
    assert predictions.shape == (len(X_test),)
    assert probabilities.shape == (len(X_test), len(np.unique(y)))
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

def test_model_reproducibility():
    """Test that model produces same results with same random seed"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Act - Train two models with same seed
    model1 = RandomForestClassifier(random_state=42)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)
    
    model2 = RandomForestClassifier(random_state=42)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)
    
    # Assert
    np.testing.assert_array_equal(pred1, pred2)
```

### Testing Predictions

```python
def test_prediction_output_types():
    """Test that predictions are correct type"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Act
    predictions = model.predict(X[:5])
    probabilities = model.predict_proba(X[:5])
    
    # Assert
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype in [np.int32, np.int64]
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.dtype == np.float64

def test_prediction_bounds():
    """Test that predictions are within valid range"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Act
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Assert
    assert np.all(predictions >= 0)
    assert np.all(predictions < len(np.unique(y)))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

def test_model_performance_threshold():
    """Test that model meets minimum performance requirement"""
    # Arrange
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    min_accuracy = 0.7
    
    # Act
    accuracy = model.score(X_test, y_test)
    
    # Assert
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.3f} below threshold {min_accuracy}"
```

---

## Integration Testing

### Testing End-to-End Pipeline

```python
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def test_complete_pipeline():
    """Test entire ML pipeline from data to prediction"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Act
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    
    # Assert
    assert len(predictions) == len(y_test)
    assert score > 0.5  # Reasonable performance
    assert all(pred in [0, 1] for pred in predictions)

def test_pipeline_with_missing_data():
    """Test pipeline handles missing data correctly"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    # Introduce missing values
    X[0, 0] = np.nan
    X[1, 1] = np.nan
    
    from sklearn.impute import SimpleImputer
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Act & Assert - Should not raise error
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
```

### Testing Model Persistence

```python
import joblib
import tempfile
import os

def test_model_save_and_load():
    """Test that model can be saved and loaded correctly"""
    # Arrange
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Act - Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
        model_path = f.name
        joblib.dump(model, model_path)
    
    # Act - Load
    loaded_model = joblib.load(model_path)
    
    # Assert
    original_pred = model.predict(X[:5])
    loaded_pred = loaded_model.predict(X[:5])
    np.testing.assert_array_equal(original_pred, loaded_pred)
    
    # Cleanup
    os.unlink(model_path)
```

---

## Testing APIs

### Testing FastAPI ML Endpoints

```python
import pytest
from fastapi.testclient import TestClient
from your_api import app  # Your FastAPI app
import numpy as np

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(client):
    """Test prediction endpoint"""
    # Arrange
    test_data = {
        "features": [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    # Act
    response = client.post("/predict", json=test_data)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))

def test_predict_endpoint_validation(client):
    """Test that endpoint validates input"""
    # Arrange - Invalid input (wrong number of features)
    test_data = {
        "features": [1.0, 2.0]  # Wrong number
    }
    
    # Act
    response = client.post("/predict", json=test_data)
    
    # Assert
    assert response.status_code == 400  # Bad request

def test_batch_predict_endpoint(client):
    """Test batch prediction endpoint"""
    # Arrange
    test_data = {
        "requests": [
            {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
            {"features": [2.0, 3.0, 4.0, 5.0, 6.0]}
        ]
    }
    
    # Act
    response = client.post("/predict/batch", json=test_data)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all("prediction" in item for item in data)
```

---

## Model Testing Strategies

### Testing Model Assumptions

```python
def test_model_assumptions():
    """Test that model assumptions are met"""
    # Example: Test linear regression assumptions
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from scipy import stats
    
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    
    # Test residuals are normally distributed
    predictions = model.predict(X)
    residuals = y - predictions
    _, p_value = stats.normaltest(residuals)
    
    # Assert (with some tolerance)
    assert p_value > 0.05, "Residuals are not normally distributed"
```

### Testing Edge Cases

```python
def test_empty_input():
    """Test model handles empty input"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Should handle empty array gracefully
    with pytest.raises(ValueError):
        model.predict(np.array([]).reshape(0, 10))

def test_single_sample():
    """Test model handles single sample prediction"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Should work with single sample
    prediction = model.predict(X[0:1])
    assert len(prediction) == 1

def test_extreme_values():
    """Test model handles extreme input values"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Test with very large values
    X_extreme = np.array([[1e10] * 10])
    prediction = model.predict(X_extreme)
    assert len(prediction) == 1
```

### Testing with Synthetic Data

```python
def test_model_on_synthetic_data():
    """Test model on known synthetic patterns"""
    # Create data with known pattern
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Create target with clear pattern
    y = (X[:, 0] > 0).astype(int)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Model should learn the pattern
    accuracy = model.score(X, y)
    assert accuracy > 0.8, "Model should learn clear pattern"
```

---

## Property-Based Testing

### Using Hypothesis for ML Testing

```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    n_samples=st.integers(min_value=10, max_value=1000),
    n_features=st.integers(min_value=2, max_value=50),
    n_classes=st.integers(min_value=2, max_value=10)
)
def test_model_handles_various_input_sizes(n_samples, n_features, n_classes):
    """Test model works with various input dimensions"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=42
    )
    
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == n_samples
    assert all(0 <= p < n_classes for p in predictions)
```

---

## Best Practices

### 1. Use Fixtures for Common Setup

```python
@pytest.fixture
def sample_data():
    """Fixture for sample classification data"""
    return make_classification(n_samples=100, n_features=10, random_state=42)

@pytest.fixture
def trained_model(sample_data):
    """Fixture for trained model"""
    X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def test_with_fixtures(sample_data, trained_model):
    """Test using fixtures"""
    X, y = sample_data
    predictions = trained_model.predict(X)
    assert len(predictions) == len(y)
```

### 2. Test Data Quality

```python
def test_data_quality():
    """Test that training data meets quality requirements"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Check for missing values
    assert not np.isnan(X).any(), "Data contains NaN values"
    
    # Check for infinite values
    assert np.isfinite(X).all(), "Data contains infinite values"
    
    # Check class balance (for classification)
    unique, counts = np.unique(y, return_counts=True)
    min_class_ratio = counts.min() / counts.max()
    assert min_class_ratio > 0.1, "Classes are too imbalanced"
```

### 3. Test Model Performance Regression

```python
def test_performance_regression():
    """Test that model performance doesn't degrade"""
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Baseline performance (from previous version)
    baseline_accuracy = 0.85
    
    current_accuracy = model.score(X_test, y_test)
    
    # Allow small degradation (5%)
    assert current_accuracy >= baseline_accuracy * 0.95, \
        f"Performance degraded: {current_accuracy:.3f} < {baseline_accuracy * 0.95:.3f}"
```

### 4. Test Model Consistency

```python
def test_model_consistency():
    """Test that model produces consistent results"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train multiple models with same seed
    models = []
    for _ in range(3):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        models.append(model)
    
    # All should produce same predictions
    predictions = [model.predict(X_test) for model in models]
    for i in range(1, len(predictions)):
        np.testing.assert_array_equal(predictions[0], predictions[i])
```

### 5. Mock External Dependencies

```python
from unittest.mock import Mock, patch

def test_model_with_mocked_data_source():
    """Test model training with mocked data"""
    # Mock data loading
    with patch('your_module.load_data') as mock_load:
        mock_load.return_value = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        
        # Your code that uses load_data
        X, y = mock_load()
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        assert model.score(X, y) > 0.5
```

---

## Key Takeaways

1. **Test preprocessing functions** - Ensure data transformations work correctly
2. **Test model training** - Verify models train and produce valid outputs
3. **Test predictions** - Check output types, shapes, and bounds
4. **Test pipelines** - Verify end-to-end workflows
5. **Test APIs** - Ensure deployment endpoints work correctly
6. **Test edge cases** - Handle empty inputs, extreme values, single samples
7. **Test performance** - Set minimum performance thresholds
8. **Use fixtures** - Reduce code duplication
9. **Test reproducibility** - Ensure consistent results with same seeds
10. **Mock dependencies** - Isolate unit tests from external systems

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Testing Library](https://hypothesis.readthedocs.io/)
- [Testing Machine Learning Systems](https://www.oreilly.com/library/view/testing-machine-learning/9781492047957/)

---

**Remember**: Good tests give you confidence to deploy and refactor. Start with critical paths and expand coverage over time!

