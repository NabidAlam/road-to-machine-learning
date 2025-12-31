# Advanced MLOps Topics

Comprehensive guide to advanced MLOps techniques and tools.

## Table of Contents

- [Advanced Experiment Tracking](#advanced-experiment-tracking)
- [Feature Stores](#feature-stores)
- [Model Monitoring](#model-monitoring)
- [Automated Retraining](#automated-retraining)
- [Kubeflow](#kubeflow)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Experiment Tracking

### Hyperparameter Tuning with MLflow

```python
import mlflow
from sklearn.model_selection import ParameterGrid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        mlflow.log_metric("accuracy", score)
        
        if score > best_score:
            best_score = score
            best_params = params
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("best_model", "true")
```

---

## Feature Stores

### What is a Feature Store?

Centralized repository for features used in training and serving.

**Benefits:**
- Reuse features across models
- Consistency between training and serving
- Feature versioning
- Real-time feature serving

### Simple Feature Store

```python
class FeatureStore:
    def __init__(self):
        self.features = {}
        self.versions = {}
    
    def register_feature(self, name, feature_fn, version=1):
        """Register a feature computation function"""
        self.features[name] = feature_fn
        self.versions[name] = version
    
    def get_feature(self, name, data):
        """Compute feature"""
        return self.features[name](data)
```

---

## Model Monitoring

### Performance Monitoring

```python
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
    
    def log_prediction(self, prediction, actual=None):
        """Log prediction for monitoring"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(datetime.now())
    
    def calculate_drift(self):
        """Calculate prediction drift"""
        if len(self.predictions) < 100:
            return None
        
        recent_preds = self.predictions[-100:]
        older_preds = self.predictions[-200:-100]
        
        # Statistical test for drift
        from scipy import stats
        statistic, p_value = stats.ks_2samp(older_preds, recent_preds)
        return p_value < 0.05  # Significant drift
```

---

## Automated Retraining

### Retraining Pipeline

```python
def should_retrain(monitor):
    """Determine if model should be retrained"""
    # Check data drift
    if monitor.calculate_drift():
        return True
    
    # Check performance degradation
    recent_accuracy = calculate_recent_accuracy(monitor)
    if recent_accuracy < threshold:
        return True
    
    return False

def retrain_pipeline():
    """Automated retraining pipeline"""
    # 1. Check if retraining needed
    if not should_retrain(monitor):
        return
    
    # 2. Load new data
    new_data = load_latest_data()
    
    # 3. Train new model
    with mlflow.start_run():
        model = train_model(new_data)
        mlflow.sklearn.log_model(model, "model")
    
    # 4. Evaluate
    if evaluate_model(model) > current_model_performance:
        # 5. Deploy new model
        deploy_model(model)
```

---

## Kubeflow

### Kubeflow Pipelines

```python
try:
    import kfp
    from kfp import dsl
    
    @dsl.pipeline(
        name='ML Pipeline',
        description='End-to-end ML pipeline'
    )
    def ml_pipeline():
        # Data preparation
        prepare_op = dsl.ContainerOp(
            name='prepare',
            image='prepare-image',
            command=['python', 'prepare.py']
        )
        
        # Training
        train_op = dsl.ContainerOp(
            name='train',
            image='train-image',
            command=['python', 'train.py']
        )
        train_op.after(prepare_op)
        
        # Evaluation
        eval_op = dsl.ContainerOp(
            name='evaluate',
            image='eval-image',
            command=['python', 'evaluate.py']
        )
        eval_op.after(train_op)
    
    # Compile and run
    # kfp.Client().create_run_from_pipeline_func(ml_pipeline, arguments={})
except ImportError:
    print("Install: pip install kfp")
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Not Versioning Data

**Solution**: Use DVC or similar tool

### Pitfall 2: Not Tracking Experiments

**Solution**: Use MLflow or W&B from the start

### Pitfall 3: No Monitoring

**Solution**: Set up monitoring from day one

---

## Key Takeaways

1. **Advanced Tracking**: Hyperparameter tuning, experiment comparison
2. **Feature Stores**: Centralize feature management
3. **Monitoring**: Track model performance and drift
4. **Automation**: Automate retraining pipelines
5. **Kubeflow**: Orchestrate complex ML pipelines

---

**Remember**: Advanced MLOps requires proper tooling and practices!

