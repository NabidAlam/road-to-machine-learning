# MLOps Cheatsheet

Comprehensive quick reference for MLOps tools, practices, and workflows.

## Table of Contents

- [Version Control](#version-control)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [CI/CD](#cicd)
- [Model Monitoring](#model-monitoring)
- [Data Versioning](#data-versioning)
- [Pipeline Orchestration](#pipeline-orchestration)

---

## Version Control

### DVC (Data Version Control)

```bash
# Initialize DVC
dvc init

# Track large files
dvc add data/train.csv
dvc add models/model.pkl

# Create pipeline
dvc run -n prepare \
    -d data/raw.csv \
    -o data/processed.csv \
    python prepare.py

dvc run -n train \
    -d data/processed.csv \
    -d src/train.py \
    -o models/model.pkl \
    python src/train.py

# Reproduce pipeline
dvc repro

# Push to remote storage
dvc remote add -d myremote s3://mybucket/dvc
dvc push

# Pull from remote
dvc pull
```

### Git LFS

```bash
# Install Git LFS
git lfs install

# Track file types
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.csv"

# Add .gitattributes
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

---

## Experiment Tracking

### MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Set experiment
mlflow.set_experiment("my_experiment")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("plots/confusion_matrix.png")
    
    # Log tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "iris")
```

### MLflow UI

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Weights & Biases (wandb)

```python
import wandb

# Initialize
wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.01,
        "epochs": 10
    }
)

# Log metrics
wandb.log({"accuracy": 0.95, "loss": 0.05})

# Log images
wandb.log({"confusion_matrix": wandb.Image(cm_plot)})

# Log model
wandb.log_artifact(model_path, name="model")

# Finish
wandb.finish()
```

---

## Model Registry

### MLflow Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/<run_id>/model"
model_name = "IrisClassifier"
mv = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

# Transition stages
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging"
)

client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production"
)

# Load model from registry
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)
```

### Model Stages

- **None**: Newly registered
- **Staging**: Testing in staging environment
- **Production**: Live in production
- **Archived**: Deprecated

---

## CI/CD

### GitHub Actions

```yaml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

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
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run tests
        run: pytest tests/
      
      - name: Lint
        run: |
          pip install flake8
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Train model
        run: python train.py
      
      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/
```

### Model Testing

```python
# tests/test_model.py
import pytest
from sklearn.metrics import accuracy_score

def test_model_accuracy(model, X_test, y_test):
    """Test model meets minimum accuracy threshold"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy >= 0.90, f"Accuracy {accuracy} below threshold"

def test_model_output_shape(model, X_test):
    """Test model output shape"""
    predictions = model.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0]

def test_model_probabilities(model, X_test):
    """Test probability outputs sum to 1"""
    probabilities = model.predict_proba(X_test)
    assert (probabilities.sum(axis=1) - 1.0 < 1e-6).all()
```

---

## Model Monitoring

### Data Drift Detection

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Reference and current data
reference_data = train_df
current_data = production_df

# Create report
report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=ColumnMapping()
)

# Generate report
report.save_html("data_drift_report.html")
```

### Model Performance Monitoring

```python
import mlflow

# Log production metrics
with mlflow.start_run(run_name="production_monitoring"):
    mlflow.log_metric("production_accuracy", current_accuracy)
    mlflow.log_metric("prediction_latency", latency_ms)
    mlflow.log_metric("request_count", request_count)
```

---

## Data Versioning

### DVC Data Pipeline

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw.csv
      - src/prepare.py
    outs:
      - data/processed.csv
  
  train:
    cmd: python src/train.py
    deps:
      - data/processed.csv
      - src/train.py
    params:
      - config.yaml:model_params
    outs:
      - models/model.pkl
    metrics:
      - metrics/accuracy.json
```

### Data Validation

```python
import great_expectations as ge

# Create expectation suite
df = ge.read_csv("data/train.csv")
df.expect_column_values_to_not_be_null("target")
df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
df.save_expectation_suite("expectations.json")

# Validate new data
new_df = ge.read_csv("data/new_data.csv")
results = new_df.validate(expectation_suite="expectations.json")
```

---

## Pipeline Orchestration

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_model():
    # Training code
    pass

def evaluate_model():
    # Evaluation code
    pass

dag = DAG(
    'ml_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily'
)

train_task = PythonOperator(
    task_id='train',
    python_callable=train_model,
    dag=dag
)

eval_task = PythonOperator(
    task_id='evaluate',
    python_callable=evaluate_model,
    dag=dag
)

train_task >> eval_task
```

### Prefect

```python
from prefect import flow, task

@task
def prepare_data():
    # Data preparation
    return processed_data

@task
def train_model(data):
    # Model training
    return model

@flow
def ml_pipeline():
    data = prepare_data()
    model = train_model(data)
    return model

# Run pipeline
ml_pipeline()
```

---

## Best Practices

### Version Control Checklist

- [ ] Use DVC for large files
- [ ] Track data versions
- [ ] Version model artifacts
- [ ] Use Git LFS for binary files
- [ ] Document data lineage

### Experiment Tracking Checklist

- [ ] Log all hyperparameters
- [ ] Track all metrics
- [ ] Save model artifacts
- [ ] Log data versions
- [ ] Tag experiments meaningfully

### Model Registry Checklist

- [ ] Register all production models
- [ ] Use semantic versioning
- [ ] Document model metadata
- [ ] Track model lineage
- [ ] Implement approval workflow

### CI/CD Checklist

- [ ] Run tests automatically
- [ ] Validate data schemas
- [ ] Test model performance
- [ ] Deploy to staging first
- [ ] Monitor production metrics

---

## Quick Reference

### Common Commands

```bash
# DVC
dvc init                    # Initialize DVC
dvc add <file>              # Track file
dvc push                    # Push to remote
dvc pull                    # Pull from remote
dvc repro                   # Reproduce pipeline

# MLflow
mlflow ui                   # Start UI
mlflow run .                # Run MLflow project
mlflow models serve <path>  # Serve model

# Git LFS
git lfs install             # Install Git LFS
git lfs track "*.pkl"       # Track file type
git lfs ls-files            # List tracked files
```

### Tool Comparison

| Tool | Purpose | Best For |
|------|---------|----------|
| **DVC** | Data versioning | Large datasets, pipelines |
| **MLflow** | Experiment tracking | Model lifecycle management |
| **wandb** | Experiment tracking | Deep learning experiments |
| **Airflow** | Pipeline orchestration | Complex workflows |
| **Prefect** | Pipeline orchestration | Modern Python workflows |

---

**Remember**: MLOps ensures reproducibility, scalability, and reliability of ML systems!

