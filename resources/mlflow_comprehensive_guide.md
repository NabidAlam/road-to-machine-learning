# MLFlow Comprehensive Guide

Complete guide to MLFlow for experiment tracking, model registry, hyperparameter tuning, and MLOps lifecycle management.

## Table of Contents

- [Introduction to MLFlow](#introduction-to-mlflow)
- [What is Experiment Tracking?](#what-is-experiment-tracking)
- [MLFlow in MLOps Lifecycle](#mlflow-in-mlops-lifecycle)
- [MLFlow Unique Features](#mlflow-unique-features)
- [Why MLFlow is Widely Used](#why-mlflow-is-widely-used)
- [Recent MLFlow Updates](#recent-mlflow-updates)
- [Implementing Experiment Tracking with MLFlow](#implementing-experiment-tracking-with-mlflow)
- [MLFlow Server UI Walkthrough](#mlflow-server-ui-walkthrough)
- [Model Logging](#model-logging)
- [Model Evaluation through MLFlow](#model-evaluation-through-mlflow)
- [Hyperparameter Tuning with MLFlow](#hyperparameter-tuning-with-mlflow)
- [Model Registry and Version Tracking](#model-registry-and-version-tracking)
- [Model Deployment with Docker in MLFlow](#model-deployment-with-docker-in-mlflow)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to MLFlow

### What is MLFlow?

MLFlow is an open-source platform for managing the complete machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, sharing and deploying models, and managing model registry.

**Key Components:**
1. **MLFlow Tracking**: Log and query experiments
2. **MLFlow Projects**: Package ML code for reproducibility
3. **MLFlow Models**: Deploy models to various platforms
4. **MLFlow Model Registry**: Centralized model management

### Installation

```bash
# Install MLFlow
pip install mlflow

# For specific integrations
pip install mlflow[extras]  # Includes additional features
pip install mlflow[azure]   # Azure integration
pip install mlflow[aws]     # AWS integration
```

---

## What is Experiment Tracking?

### Definition

**Experiment tracking** is the process of logging and organizing machine learning experiments to:
- Compare different model configurations
- Reproduce past results
- Understand what works and what doesn't
- Share findings with team members
- Make data-driven decisions

### Where Does It Fit in MLOps Lifecycle?

```
Data Collection
    ↓
Data Preparation
    ↓
Feature Engineering
    ↓
Model Training → [EXPERIMENT TRACKING] ← Track parameters, metrics, artifacts
    ↓
Model Evaluation
    ↓
Model Registry → [EXPERIMENT TRACKING] ← Track model versions
    ↓
Model Deployment
    ↓
Model Monitoring → [EXPERIMENT TRACKING] ← Track production metrics
    ↓
Model Retraining
```

### Benefits of Experiment Tracking

**1. Reproducibility**
- Recreate exact experiments
- Same code + data + parameters = same results
- Track environment and dependencies

**2. Comparison**
- Compare multiple runs side-by-side
- Identify best hyperparameters
- Understand impact of changes

**3. Collaboration**
- Share experiments with team
- Centralized experiment history
- Avoid duplicate work

**4. Learning**
- Understand what improves performance
- Learn from failures
- Build institutional knowledge

**5. Debugging**
- Identify when performance degraded
- Track changes that caused issues
- Roll back to working versions

---

## MLFlow Architecture and Components

### MLFlow Architecture

**Four Main Components:**

**1. MLFlow Tracking:**
- Log and query experiments
- Track parameters, metrics, artifacts
- Compare runs
- Organize into experiments

**2. MLFlow Projects:**
- Package ML code for reproducibility
- Define dependencies (Conda, Docker)
- Run on different platforms
- Version control for code

**3. MLFlow Models:**
- Standardized model format
- Model Flavors (sklearn, tensorflow, pytorch, etc.)
- Cross-platform deployment
- Easy serving

**4. MLFlow Model Registry:**
- Centralized model repository
- Model versioning
- Staging workflow
- Model lineage

### Setting Up MLFlow Tracking Server

**Local Server:**
```bash
# Start local tracking server
mlflow ui

# Access UI at http://localhost:5000
```

**Remote/Cloud Server:**
```bash
# Start server with backend store
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# With S3 for artifacts
mlflow server \
    --backend-store-uri postgresql://user:pass@host:5432/mlflow \
    --default-artifact-root s3://bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

**Connect to Server:**
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Or set environment variable
# export MLFLOW_TRACKING_URI=http://localhost:5000
```

## MLFlow in MLOps Lifecycle

### MLFlow Components in MLOps

**1. Experiment Tracking (MLFlow Tracking)**
- Log parameters, metrics, artifacts
- Compare runs
- Search and filter experiments
- Organize runs into experiments

**2. Model Packaging (MLFlow Projects)**
- Package code for reproducibility
- Define dependencies (Conda, Docker)
- Run on different platforms
- Version control for code

**3. Model Deployment (MLFlow Models)**
- Deploy to various platforms
- Standardized model format (Model Flavors)
- Easy serving
- Cross-platform compatibility

**4. Model Management (MLFlow Model Registry)**
- Centralized model repository
- Version control for models
- Staging workflow (None → Staging → Production → Archived)
- Model lineage tracking

### MLOps Workflow with MLFlow

```
1. Data Versioning (DVC/Git LFS)
    ↓
2. Experiment Tracking (MLFlow Tracking)
    ↓
3. Model Training with Logging
    ↓
4. Model Evaluation (MLFlow Evaluation)
    ↓
5. Model Registry (MLFlow Registry)
    ↓
6. Model Deployment (MLFlow Models + Docker)
    ↓
7. Model Monitoring
    ↓
8. Retraining Loop
```

---

## MLFlow Unique Features

### 1. Open Source
- Free and open-source
- No vendor lock-in
- Community-driven

### 2. Framework Agnostic
- Works with any ML framework
- sklearn, TensorFlow, PyTorch, XGBoost, etc.
- Language support: Python, R, Java, REST API

### 3. Unified Interface
- Single API for all ML frameworks
- Consistent logging interface
- Easy to learn and use

### 4. Model Registry
- Centralized model management
- Version control
- Staging workflow

### 5. Model Serving
- Built-in model serving
- Docker support
- Cloud platform integration

### 6. Artifact Storage
- Store models, plots, data
- Support for local, S3, Azure, GCS
- Version artifacts

### 7. Search and Filter
- Powerful search capabilities
- Filter by parameters, metrics, tags
- Compare runs easily

### 8. REST API
- Programmatic access
- Integration with other tools
- Automation support

---

## Why MLFlow is Widely Used

### Industry Adoption

**1. Scalability**
- Handles small to enterprise-scale projects
- Distributed tracking server
- Cloud integration

**2. Flexibility**
- Works with any ML framework
- Custom integrations possible
- Extensible architecture

**3. Production Ready**
- Model serving capabilities
- Docker support
- Cloud deployment options

**4. Team Collaboration**
- Centralized tracking server
- Shared experiments
- Model registry for teams

**5. Cost Effective**
- Open source (free)
- Self-hosted option
- No per-user licensing

**6. Active Community**
- Regular updates
- Good documentation
- Community support

### Companies Using MLFlow

- Databricks (creators)
- Many Fortune 500 companies
- Startups and research institutions
- Widely adopted in industry

---

## Recent MLFlow Updates

### MLFlow 2.x Features

**1. Enhanced Model Registry**
- Better UI
- Improved versioning
- Enhanced search

**2. MLFlow Evaluation**
- Comprehensive model evaluation
- Built-in metrics
- Custom evaluation functions

**3. Prompt Engineering UI**
- Track LLM experiments
- Prompt versioning
- Response comparison

**4. Enhanced Tracking**
- Better visualization
- Improved search
- More artifact types

**5. Cloud Integrations**
- Better AWS/Azure/GCP support
- Managed MLFlow options
- Serverless deployment

**6. Performance Improvements**
- Faster tracking
- Better scalability
- Optimized queries

---

## Implementing Experiment Tracking with MLFlow

### Using MLFlow in Jupyter Environment

**Basic Setup:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("iris_classification")

# Start a run
with mlflow.start_run(run_name="random_forest_baseline"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
```

### Logging Key Parameters and Metrics

**Parameters:**
```python
# Single parameter
mlflow.log_param("learning_rate", 0.001)

# Multiple parameters
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
})

# Nested parameters (using dict)
mlflow.log_param("model_config.n_estimators", 100)
mlflow.log_param("model_config.max_depth", 10)
```

**Metrics:**
```python
# Single metric
mlflow.log_metric("accuracy", 0.95)

# Multiple metrics
mlflow.log_metrics({
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.94,
    "f1_score": 0.935
})

# Step-based metrics (for training curves)
for epoch in range(10):
    train_loss = train_one_epoch()
    val_loss = validate()
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

**Artifacts:**
```python
# Log files
mlflow.log_artifact("feature_importance.png")
mlflow.log_artifact("confusion_matrix.png")

# Log directory
mlflow.log_artifacts("plots/", artifact_path="visualizations")

# Log data
import pandas as pd
df.to_csv("predictions.csv", index=False)
mlflow.log_artifact("predictions.csv")
```

**Tags:**
```python
# Single tag
mlflow.set_tag("model_type", "RandomForest")
mlflow.set_tag("dataset", "iris")

# Multiple tags
mlflow.set_tags({
    "model_type": "RandomForest",
    "dataset": "iris",
    "experiment_type": "baseline"
})
```

---

## MLFlow Server UI Walkthrough

### Starting MLFlow UI

**Local Server:**
```bash
# Start MLFlow UI (default: http://localhost:5000)
mlflow ui

# Custom port
mlflow ui --port 5001

# Custom backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db

# With artifact storage
mlflow ui --default-artifact-root ./mlruns
```

**Remote Server:**
```bash
# With remote tracking server
mlflow ui --backend-store-uri postgresql://user:pass@host:5432/mlflowdb \
          --default-artifact-root s3://mlflow-artifacts
```

### UI Components

**1. Experiments List**
- View all experiments
- Filter by name
- See experiment metadata

**2. Runs Table**
- List of all runs in experiment
- Sortable columns (metrics, parameters)
- Filter by parameters/metrics
- Compare runs side-by-side

**3. Run Details**
- **Overview**: Parameters, metrics, tags
- **Artifacts**: Models, plots, files
- **Metrics**: Training curves
- **System**: Environment info

**4. Compare Runs**
- Side-by-side comparison
- Parallel coordinates plot
- Scatter plots
- Parameter importance

**5. Search and Filter**
- Search by run name
- Filter by parameters
- Filter by metrics
- Filter by tags

### Using MLFlow UI

**Viewing Experiments:**
1. Open browser: `http://localhost:5000`
2. Click on experiment name
3. View all runs in experiment

**Comparing Runs:**
1. Select multiple runs (checkboxes)
2. Click "Compare"
3. View side-by-side comparison
4. Analyze differences

**Viewing Run Details:**
1. Click on run name
2. View parameters, metrics, artifacts
3. Download model or artifacts
4. View training curves

**Searching Runs:**
1. Use search bar
2. Filter by parameters: `params.n_estimators = 100`
3. Filter by metrics: `metrics.accuracy > 0.9`
4. Filter by tags: `tags.model_type = "RandomForest"`

---

## MLFlow Models

### Saving Models in MLFlow Format

**Model Flavors:**
MLFlow supports multiple model formats (flavors) for cross-platform deployment.

**Supported Flavors:**
- **sklearn**: Scikit-learn models
- **tensorflow**: TensorFlow/Keras models
- **pytorch**: PyTorch models
- **xgboost**: XGBoost models
- **lightgbm**: LightGBM models
- **spark**: Spark ML models
- **python_function**: Generic Python models (pyfunc)

**Logging Models:**

**sklearn Models:**
```python
import mlflow.sklearn

# Log sklearn model
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="IrisClassifier"
)
```

**TensorFlow/Keras Models:**
```python
import mlflow.keras

# Log Keras model
mlflow.keras.log_model(
    model,
    "model",
    registered_model_name="ImageClassifier"
)
```

**PyTorch Models:**
```python
import mlflow.pytorch

# Log PyTorch model
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="NeuralNet"
)
```

**Generic Python Function (pyfunc):**
```python
import mlflow.pyfunc

# Custom model class
class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load dependencies
        self.model = joblib.load(context.artifacts["model_path"])
    
    def predict(self, context, model_input):
        # Custom prediction logic
        return self.model.predict(model_input)

# Log custom model
mlflow.pyfunc.log_model(
    "model",
    python_model=CustomModel(),
    artifacts={"model_path": "model.pkl"}
)
```

### Creating Reproducible Environment

**Using Conda:**
```python
# MLFlow automatically creates conda.yaml
mlflow.sklearn.log_model(
    model,
    "model",
    conda_env={
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "mlflow==2.8.0",
                    "scikit-learn==1.3.0",
                    "pandas==2.0.0",
                    "numpy==1.24.0"
                ]
            }
        ]
    }
)
```

**Using Docker:**
```python
# MLFlow Projects with Docker
# MLproject file
"""
name: iris-classification
docker_env:
  image: mlflow-docker-env:latest
entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
    command: "python train.py --n-estimators {n_estimators}"
"""

# Dockerfile
"""
FROM python:3.9-slim
RUN pip install mlflow scikit-learn pandas numpy
COPY . /app
WORKDIR /app
"""

# Run project
# mlflow run . --docker-args "-v /path/to/data:/data"
```

**Using requirements.txt:**
```python
# MLFlow can use requirements.txt
mlflow.sklearn.log_model(
    model,
    "model",
    pip_requirements="requirements.txt"
)

# requirements.txt
"""
mlflow==2.8.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
"""
```

## Model Logging

### Logging Models

**sklearn Models:**
```python
import mlflow.sklearn

# Log sklearn model
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="IrisClassifier"
)
```

**TensorFlow/Keras Models:**
```python
import mlflow.keras

# Log Keras model
mlflow.keras.log_model(
    model,
    "model",
    registered_model_name="ImageClassifier"
)
```

**PyTorch Models:**
```python
import mlflow.pytorch

# Log PyTorch model
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="TextClassifier"
)
```

**XGBoost Models:**
```python
import mlflow.xgboost

# Log XGBoost model
mlflow.xgboost.log_model(
    model,
    "model",
    registered_model_name="GradientBoosting"
)
```

**Custom Models (Python Function):**
```python
import mlflow.pyfunc

# Define custom model class
class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load artifacts
        self.model = joblib.load(context.artifacts["model_path"])
    
    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log custom model
mlflow.pyfunc.log_model(
    "model",
    python_model=CustomModel(),
    artifacts={"model_path": "model.pkl"}
)
```

### Model Signatures

**Define Input/Output Schema:**
```python
from mlflow.models import infer_signature

# Infer signature from data
signature = infer_signature(X_train, y_train)

# Log model with signature
mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature
)
```

**Manual Signature:**
```python
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

# Define input schema
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
    ColSpec("double", "feature3")
])

# Define output schema
output_schema = Schema([ColSpec("long", "prediction")])

# Create signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

---

## Model Evaluation through MLFlow

### MLFlow Evaluation API

**Basic Evaluation:**
```python
import mlflow
from mlflow.models import evaluate

# Evaluate model
results = evaluate(
    model="runs:/<run_id>/model",
    data=X_test,
    targets=y_test,
    model_type="classifier"
)

print(results)
```

**Comprehensive Evaluation:**
```python
from mlflow.models import evaluate
from mlflow.metrics import make_metric
import numpy as np

# Custom metric
def custom_metric(eval_df, builtin_metrics):
    return np.mean(eval_df["prediction"] == eval_df["target"])

custom_metric_fn = make_metric(
    eval_fn=custom_metric,
    greater_is_better=True,
    name="custom_accuracy"
)

# Evaluate with custom metrics
results = evaluate(
    model="runs:/<run_id>/model",
    data=X_test,
    targets=y_test,
    model_type="classifier",
    extra_metrics=[custom_metric_fn]
)
```

**Logging Evaluation Results:**
```python
with mlflow.start_run():
    # Train model
    model = train_model()
    mlflow.sklearn.log_model(model, "model")
    
    # Evaluate
    eval_results = evaluate(
        model=mlflow.get_artifact_uri("model"),
        data=X_test,
        targets=y_test,
        model_type="classifier"
    )
    
    # Log evaluation metrics
    for metric_name, metric_value in eval_results.metrics.items():
        mlflow.log_metric(f"eval_{metric_name}", metric_value)
    
    # Log evaluation artifacts
    mlflow.log_dict(eval_results.tags, "evaluation_tags.json")
```

### Evaluation Metrics

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Regression Metrics:**
- RMSE
- MAE
- R-squared
- Mean squared error

**Custom Evaluation:**
```python
from mlflow.models import evaluate
import pandas as pd

# Custom evaluation function
def custom_eval(eval_df, builtin_metrics):
    # Calculate custom metrics
    custom_metrics = {
        "custom_score": calculate_custom_score(eval_df),
        "business_metric": calculate_business_metric(eval_df)
    }
    return custom_metrics

# Evaluate
results = evaluate(
    model="runs:/<run_id>/model",
    data=X_test,
    targets=y_test,
    evaluators="default",
    custom_metrics=[custom_eval]
)
```

---

## Hyperparameter Tuning with MLFlow

### Grid Search with MLFlow

**Basic Grid Search:**
```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("hyperparameter_tuning")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

best_score = 0
best_params = None
best_run_id = None

for params in ParameterGrid(param_grid):
    with mlflow.start_run(nested=True):
        # Log all parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            best_run_id = mlflow.active_run().info.run_id
            mlflow.set_tag("best_model", "true")

print(f"Best accuracy: {best_score:.4f}")
print(f"Best parameters: {best_params}")
print(f"Best run ID: {best_run_id}")
```

### Hyperparameter Tuning with Hyperopt

**Installation:**
```bash
pip install hyperopt
```

**Hyperopt with MLFlow:**
```python
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("hyperopt_tuning")

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [5, 10, 20, None]),
    'min_samples_split': hp.uniform('min_samples_split', 2, 10)
}

def objective(params):
    """Objective function for hyperopt"""
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=params['max_depth'],
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Return for hyperopt (minimize, so use negative)
        return {'loss': -accuracy, 'status': STATUS_OK}

# Run optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

print(f"Best parameters: {best}")
print(f"Best accuracy: {-trials.best_trial['result']['loss']:.4f}")
```

### Hyperparameter Tuning with Optuna

**Installation:**
```bash
pip install optuna
```

**Optuna with MLFlow:**
```python
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("optuna_tuning")

def objective(trial):
    """Objective function for Optuna"""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Start MLFlow run
    with mlflow.start_run(nested=True):
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Report to Optuna
        trial.set_user_attr('run_id', mlflow.active_run().info.run_id)
        
        return accuracy

# Create study
study = optuna.create_study(direction='maximize')

# Run optimization
study.optimize(objective, n_trials=50)

# Get best trial
best_trial = study.best_trial
print(f"Best accuracy: {best_trial.value:.4f}")
print(f"Best parameters: {best_trial.params}")
print(f"Best run ID: {best_trial.user_attrs['run_id']}")
```

### Comparing Hyperparameter Tuning Methods

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **Grid Search** | Exhaustive, simple | Slow, limited to discrete values | Small search space |
| **Random Search** | Faster, simple | May miss optimal | Medium search space |
| **Hyperopt** | Bayesian optimization, efficient | More complex | Large search space |
| **Optuna** | Modern, efficient, pruning | Learning curve | Large search space, need pruning |

---

## Model Registry and Version Tracking

### Model Registry Overview

**Purpose:**
- Centralized model management
- Version control for models
- Staging workflow
- Model lineage tracking

### Registering Models

**From Run:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from run
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"
model_name = "IrisClassifier"

# Register model
mv = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

print(f"Model version {mv.version} registered")
```

**From Artifact URI:**
```python
# Register from artifact URI
model_uri = "runs:/<run_id>/model"
mv = client.create_model_version(
    name="IrisClassifier",
    source=model_uri
)
```

### Version Tracking

**List Model Versions:**
```python
# Get all versions of a model
versions = client.search_model_versions(f"name='{model_name}'")

for version in versions:
    print(f"Version {version.version}: {version.current_stage}")
    print(f"  Run ID: {version.run_id}")
    print(f"  Status: {version.status}")
```

**Get Model Version:**
```python
# Get specific version
version = client.get_model_version(
    name=model_name,
    version=1
)

print(f"Version: {version.version}")
print(f"Stage: {version.current_stage}")
print(f"Run ID: {version.run_id}")
```

### Staging Workflow

**Stages:**
- **None**: Newly registered model
- **Staging**: Testing in staging environment
- **Production**: Live in production
- **Archived**: Deprecated models

**Transition Model:**
```python
# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)

# Archive model
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Archived"
)
```

**Get Model by Stage:**
```python
# Get production model
production_model = client.get_latest_versions(
    name=model_name,
    stages=["Production"]
)[0]

# Load production model
import mlflow.pyfunc
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)
```

### Model Metadata

**Add Description:**
```python
# Add description to model version
client.update_model_version(
    name=model_name,
    version=1,
    description="Best performing model with 95% accuracy"
)
```

**Add Tags:**
```python
# Add tags to model version
client.set_model_version_tag(
    name=model_name,
    version=1,
    key="deployed_by",
    value="john_doe"
)
```

**Model Lineage:**
```python
# Get model lineage
version = client.get_model_version(name=model_name, version=1)
run = client.get_run(version.run_id)

print(f"Training run: {run.info.run_id}")
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")
print(f"Tags: {run.data.tags}")
```

---

## Model Deployment with Docker in MLFlow

### MLFlow Model Serving

**Basic Serving:**
```bash
# Serve model from run
mlflow models serve -m runs:/<run_id>/model -p 5000

# Serve model from registry
mlflow models serve -m models:/IrisClassifier/Production -p 5000
```

**Make Predictions:**
```python
import requests
import json

# Prepare input
data = {
    "dataframe_split": {
        "columns": ["feature1", "feature2", "feature3"],
        "data": [[5.1, 3.5, 1.4]]
    }
}

# Make prediction
response = requests.post(
    "http://localhost:5000/invocations",
    json=data,
    headers={"Content-Type": "application/json"}
)

prediction = response.json()
print(prediction)
```

### Docker Deployment

**Build Docker Image:**
```bash
# Build Docker image for model
mlflow models build-docker -m models:/IrisClassifier/Production \
                           -n iris-classifier:latest
```

**Run Docker Container:**
```bash
# Run container
docker run -p 5000:8080 iris-classifier:latest
```

**Dockerfile Example:**
```dockerfile
FROM python:3.9-slim

# Install MLFlow
RUN pip install mlflow

# Copy model
COPY model /model

# Expose port
EXPOSE 8080

# Run MLFlow serving
CMD ["mlflow", "models", "serve", "-m", "/model", "-p", "8080", "--host", "0.0.0.0"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  mlflow-model:
    image: iris-classifier:latest
    ports:
      - "5000:8080"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    depends_on:
      - mlflow-server
  
  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    ports:
      - "5001:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://user:pass@db:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
    depends_on:
      - db
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
```

### Cloud Deployment

**AWS SageMaker:**
```python
import mlflow.sagemaker

# Deploy to SageMaker
mlflow.sagemaker.deploy(
    app_name="iris-classifier",
    model_uri="models:/IrisClassifier/Production",
    execution_role_arn="arn:aws:iam::123456789012:role/MLFlowSageMaker",
    instance_type="ml.m5.large",
    region_name="us-east-1"
)
```

**Azure ML:**
```python
import mlflow.azureml

# Deploy to Azure ML
azure_model, azure_model_uri = mlflow.azureml.deploy(
    model_uri="models:/IrisClassifier/Production",
    workspace=workspace,
    model_name="iris-classifier",
    service_name="iris-classifier-service"
)
```

**Google Cloud Run:**
```bash
# Build and deploy to Cloud Run
mlflow models build-docker -m models:/IrisClassifier/Production -n gcr.io/project/iris-classifier
gcloud run deploy iris-classifier --image gcr.io/project/iris-classifier --platform managed
```

---

## Best Practices

### Experiment Tracking

1. **Log Everything**
   - All hyperparameters
   - All metrics
   - Environment info
   - Data versions

2. **Use Meaningful Names**
   - Descriptive experiment names
   - Clear run names
   - Organized tags

3. **Compare Systematically**
   - Use same evaluation metrics
   - Same test set
   - Fair comparisons

4. **Document Runs**
   - Add descriptions
   - Use tags for organization
   - Link to related runs

### Model Registry

1. **Version Everything**
   - Register all promising models
   - Track model lineage
   - Document changes

2. **Use Staging Workflow**
   - Test in staging first
   - Validate before production
   - Archive old models

3. **Monitor Models**
   - Track production metrics
   - Compare with training metrics
   - Set up alerts

### Hyperparameter Tuning

1. **Start Broad, Then Narrow**
   - Wide search space initially
   - Narrow based on results
   - Focus on promising regions

2. **Use Efficient Methods**
   - Optuna or Hyperopt for large spaces
   - Grid search for small spaces
   - Use pruning when possible

3. **Track All Runs**
   - Log every hyperparameter combination
   - Learn from failures
   - Build knowledge base

### Deployment

1. **Test Before Production**
   - Test in staging environment
   - Load testing
   - Validate predictions

2. **Monitor Deployed Models**
   - Track prediction latency
   - Monitor prediction distribution
   - Alert on anomalies

3. **Version Control**
   - Tag Docker images
   - Track model versions
   - Easy rollback

---

## Resources

### Official Documentation

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLFlow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLFlow Models Guide](https://mlflow.org/docs/latest/models.html)
- [MLFlow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

### Tutorials

- [MLFlow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- [MLFlow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
- [Hyperparameter Tuning with MLFlow](https://mlflow.org/docs/latest/tracking.html#hyperparameter-tuning)

### Books

- "MLOps: Continuous delivery and automation pipelines in machine learning" by Mark Treveil et al.

### Community

- [MLFlow GitHub](https://github.com/mlflow/mlflow)
- [MLFlow Discussions](https://github.com/mlflow/mlflow/discussions)

---

**Remember**: MLFlow is a powerful tool for managing the complete ML lifecycle. Start with basic experiment tracking, then gradually add model registry, hyperparameter tuning, and deployment capabilities. Focus on reproducibility and collaboration!

