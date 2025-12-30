# MLOps Basics Complete Guide

Comprehensive guide to managing the complete ML lifecycle.

## Table of Contents

- [Version Control for ML](#version-control-for-ml)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [CI/CD for ML](#cicd-for-ml)
- [Practice Exercises](#practice-exercises)

---

## Version Control for ML

### DVC (Data Version Control)

```python
# Install: pip install dvc

# Initialize DVC
# dvc init

# Track data
# dvc add data/train.csv
# git add data/train.csv.dvc

# Reproduce pipeline
# dvc repro
```

### Git LFS

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.h5"
```

---

## Experiment Tracking

### MLflow

```python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("my_experiment")

# Log parameters and metrics
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="my-project")

# Log metrics
wandb.log({"accuracy": accuracy, "loss": loss})

# Log model
wandb.log_model(model, "model")
```

---

## Model Registry

### MLflow Model Registry

```python
# Register model
mlflow.register_model(
    "runs:/<run_id>/model",
    "MyModel"
)

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"
)
```

---

## CI/CD for ML

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
      - name: Train model
        run: python train.py
```

---

## Practice Exercises

### Exercise 1: Track Experiment with MLflow

**Task:** Log a training run with MLflow.

**Solution:**
```python
import mlflow
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(model, "model")
```

---

## Key Takeaways

1. **Version Control**: Track code, data, and models
2. **Experiment Tracking**: Learn from past experiments
3. **Model Registry**: Manage model versions
4. **CI/CD**: Automate workflows

---

## Next Steps

- Set up MLflow for your projects
- Implement CI/CD pipelines
- Practice with [16-projects-beginner](../16-projects-beginner/README.md)

**Remember**: MLOps makes ML production-ready!

