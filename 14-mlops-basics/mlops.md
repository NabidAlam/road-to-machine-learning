# MLOps Basics Complete Guide

Comprehensive guide to managing the complete ML lifecycle.

## Table of Contents

- [Introduction to MLOps](#introduction-to-mlops)
- [Version Control for ML](#version-control-for-ml)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [CI/CD for ML](#cicd-for-ml)
- [Data Pipeline](#data-pipeline)
- [Reproducibility](#reproducibility)
- [Practice Exercises](#practice-exercises)

---

## Introduction to MLOps

### What is MLOps?

**MLOps (Machine Learning Operations)** is the practice of managing the complete ML lifecycle from development to deployment and monitoring.

**Definition and Importance:**
- **Bridging the gap** between **Data Science** and **Operations**
- Combines ML development with DevOps practices
- Ensures ML models are production-ready, reliable, and maintainable
- Enables collaboration between data scientists and engineers

**Key Principles:**
- **Reproducibility**: Same code + data = same results
- **Versioning**: Track code, data, and models
- **Automation**: Reduce manual work
- **Monitoring**: Track model performance
- **Collaboration**: Work effectively in teams

### MLOps vs DevOps

**DevOps**: Code → Build → Deploy
**MLOps**: Code + Data + Model → Train → Deploy → Monitor → Retrain

**Additional Challenges:**
- Data versioning
- Model versioning
- Experiment tracking
- Model monitoring
- Data drift detection
- Model retraining pipelines

### The MLOps Lifecycle

**Four Key Phases:**

1. **Experimentation**
   - Model development
   - Hyperparameter tuning
   - Feature engineering
   - Experiment tracking

2. **Data & Model Management**
   - Data versioning
   - Model versioning
   - Model registry
   - Feature stores

3. **CI/CD (Continuous Integration/Continuous Deployment)**
   - Automated testing
   - Automated deployment
   - Pipeline automation
   - Model validation

4. **Monitoring**
   - Model performance tracking
   - Data drift detection
   - Model drift detection
   - System health monitoring

### MLOps Maturity Levels

**Level 0: Manual**
- Manual processes
- No automation
- Ad-hoc deployments
- Limited tracking

**Level 1: Automated Pipeline**
- Automated training pipeline
- Basic CI/CD
- Experiment tracking
- Model versioning

**Level 2: CI/CD**
- Full CI/CD pipeline
- Automated testing
- Automated deployment
- Model registry integration

**Level 3: Continuous Training (CT)**
- Automated retraining
- Model monitoring
- Drift detection
- Auto-deployment on triggers

### MLOps Workflow

```
Data Ingestion → EDA → Feature Engineering → Model Training → 
Hyperparameter Tuning → Evaluation → Model Registry → 
Deployment → Monitoring → Retraining
```

---

## Version Control for ML

### Why Version Control for ML?

Traditional Git tracks code, but ML needs:
- **Data Versioning**: Track dataset versions
- **Model Versioning**: Track trained models
- **Experiment Tracking**: Track hyperparameters and results
- **Reproducibility**: Recreate exact experiments

### Git and GitHub/GitLab Essentials

**Git Basics for ML Projects:**
```bash
# Initialize repository
git init

# Add files
git add .

# Commit changes
git commit -m "Add training script"

# Create branch
git checkout -b feature/new-model

# Merge branch
git checkout main
git merge feature/new-model

# Push to remote
git push origin main
```

**GitHub/GitLab for Collaborative ML Projects:**
- **Pull Requests**: Review code changes
- **Issues**: Track bugs and features
- **Actions/CI**: Automate testing and deployment
- **Wiki**: Document projects
- **Releases**: Tag model versions

### Project Structure and Best Practices

**Recommended ML Project Structure:**
```
project/
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Processed data
│   └── external/      # External data sources
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data/         # Data processing
│   ├── features/     # Feature engineering
│   ├── models/       # Model training
│   └── visualization/ # Visualizations
├── tests/            # Unit tests
├── models/           # Trained models
├── configs/         # Configuration files
├── scripts/         # Utility scripts
├── requirements.txt # Python dependencies
├── environment.yml  # Conda environment
├── Dockerfile       # Docker configuration
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```

**Best Practices:**
- Keep data out of Git (use DVC or Git LFS)
- Version control all code
- Document everything
- Use meaningful commit messages
- Tag releases
- Maintain clean project structure

### DVC (Data Version Control)

DVC extends Git to handle large files and data pipelines.

```python
# Install: pip install dvc dvc-s3  # or dvc-gdrive, dvc-azure

# Initialize DVC
# dvc init

# Track data files
# dvc add data/train.csv data/test.csv
# git add data/train.csv.dvc data/test.csv.dvc .gitignore
# git commit -m "Add training data"

# Track models
# dvc add models/model.pkl
# git add models/model.pkl.dvc

# Define pipeline
# dvc run -n prepare -d data/raw -o data/prepared python prepare.py
# dvc run -n train -d data/prepared -o models/model.pkl python train.py

# Reproduce pipeline
# dvc repro

# Push to remote storage
# dvc remote add -d myremote s3://mybucket/dvc
# dvc push

# Pull data
# dvc pull
```

### Git LFS

Git Large File Storage for large files.

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.pth"
git lfs track "data/*.csv"

# Add .gitattributes
git add .gitattributes

# Use normally
git add model.pkl
git commit -m "Add model"
git push
```

### Best Practices

```python
# .gitignore for ML projects
"""
# Data
data/raw/
data/processed/
*.csv
*.parquet

# Models
models/*.pkl
models/*.h5
models/*.pth
*.joblib

# Experiments
mlruns/
wandb/
*.log

# Environment
venv/
env/
.venv/
"""
```

---

## Experiment Tracking

### Why Track Experiments?

- **Compare Runs**: See what works best
- **Reproducibility**: Recreate past experiments
- **Collaboration**: Share results with team
- **Learning**: Understand what improves performance

### MLflow

Open-source platform for managing ML lifecycle.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set experiment
mlflow.set_experiment("iris_classification")

# Start run
with mlflow.start_run(run_name="random_forest_baseline"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
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
    
    # Log artifacts (plots, data samples)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    # ... create plot ...
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    
    # Log tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "iris")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# View results: mlflow ui
```

### Comparing Experiments

```python
# Search runs
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("iris_classification")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

print("Top 10 runs:")
for run in runs:
    print(f"Run {run.info.run_id}: accuracy={run.data.metrics['accuracy']:.4f}")
```

### MLflow with TensorFlow/Keras

```python
import mlflow.keras

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    
    # Build and train model
    model = create_keras_model()
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    # Log metrics
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    # Log model
    mlflow.keras.log_model(model, "model")
```

### MLflow with PyTorch

```python
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)
    
    # Train model
    model = train_pytorch_model()
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Weights & Biases

Beautiful UI for experiment tracking.

```python
try:
    import wandb
    
    # Initialize
    wandb.init(
        project="my-project",
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32
        }
    )
    
    # Log metrics during training
    for epoch in range(10):
        train_loss = train_one_epoch()
        val_loss = validate()
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # Log model
    wandb.log_model(model, "model")
    
    # Log plots
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(history['loss'])
    wandb.log({"loss_plot": wandb.Image(fig)})
    
    wandb.finish()
except ImportError:
    print("Install wandb: pip install wandb")
```

---

## Data and Dependency Versioning

### Data Version Control (DVC)

**Introduction to DVC:**
DVC extends Git to handle large files and data pipelines.

**Basic DVC Usage:**
```bash
# Initialize DVC
dvc init

# Track data files
dvc add data/train.csv data/test.csv

# Commit DVC files (not data itself)
git add data/train.csv.dvc data/test.csv.dvc .gitignore
git commit -m "Add training data"

# Push data to remote storage
dvc remote add -d myremote s3://mybucket/dvc
dvc push

# Pull data
dvc pull
```

**DVC Pipelines:**
```bash
# Define pipeline stages
dvc run -n prepare \
    -d data/raw \
    -o data/prepared \
    python prepare.py

dvc run -n train \
    -d data/prepared \
    -d train.py \
    -o models/model.pkl \
    -M metrics/accuracy.json \
    python train.py

# Reproduce pipeline
dvc repro
```

### Feature Store Concept

**What is a Feature Store?**
Centralized repository for storing and serving features.

**Benefits:**
- Reuse features across projects
- Consistent feature definitions
- Real-time feature serving
- Feature versioning

**Popular Feature Stores:**
- **Feast**: Open-source feature store
- **Tecton**: Managed feature store
- **Hopsworks**: Open-source platform
- **AWS SageMaker Feature Store**: Managed service

### Managing Project Dependencies

**requirements.txt:**
```txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
tensorflow==2.13.0
mlflow==2.8.0
```

**environment.yml (Conda):**
```yaml
name: ml-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - numpy=1.24.0
  - pandas=2.0.0
  - pip
  - pip:
    - scikit-learn==1.3.0
    - tensorflow==2.13.0
    - mlflow==2.8.0
```

**Best Practices:**
- Pin exact versions for reproducibility
- Update dependencies carefully
- Test after dependency updates
- Document dependency changes

---

## Model Registry

### Why Model Registry?

- **Centralized Management**: All models in one place
- **Versioning**: Track model versions
- **Staging**: Dev → Staging → Production workflow
- **Lineage**: Track model history and metadata

### MLflow Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from a run
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

# Transition to staging
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging",
    archive_existing_versions=True
)

# Get model for deployment
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Staging"
)

# Transition to production
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production"
)

# List all model versions
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

### Model Stages

```
None → Staging → Production → Archived
```

**Workflow:**
1. Register model (None stage)
2. Test in Staging
3. Promote to Production
4. Archive old versions

---

## Machine Learning Project Basics (Refresher)

### Standard ML Workflow

**1. Data Ingestion**
- Load data from various sources
- Handle different formats
- Validate data quality

**2. Exploratory Data Analysis (EDA)**
- Understand data distribution
- Identify patterns
- Detect outliers
- Visualize relationships

**3. Feature Engineering**
- Create new features
- Transform features
- Select important features
- Handle missing values

**4. Model Training**
- Split data (train/validation/test)
- Train multiple models
- Compare performance
- Select best model

**5. Hyperparameter Tuning**
- Grid search
- Random search
- Bayesian optimization
- Find optimal hyperparameters

**6. Evaluation Metrics**
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Regression**: MSE, RMSE, MAE, R² Score
- **Cross-validation**: K-fold, Stratified

### Serializing Models

**Using Pickle:**
```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

**Using Joblib (Recommended for scikit-learn):**
```python
import joblib

# Save model (faster for NumPy arrays)
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')
```

**Why Joblib over Pickle?**
- Faster for NumPy arrays
- Better compression
- More efficient for scikit-learn models

### Key Concept: Reproducibility in ML

**Understanding Non-Determinism:**
- Random seeds in algorithms
- Data shuffling
- Parallel processing
- GPU operations

**Version Control for Reproducibility:**
- **Code**: Git versioning
- **Data**: DVC or Git LFS
- **Dependencies**: requirements.txt, environment.yml
- **Configuration**: Config files, hyperparameters

**Reproducibility Checklist:**
```python
# Set random seeds
import numpy as np
import random
import tensorflow as tf

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Version data
# Use DVC or Git LFS

# Version dependencies
# requirements.txt with exact versions

# Version configuration
# config.yaml with all hyperparameters
```

---

## CI/CD for ML

### DevOps Tools Foundation

**Introduction to CI/CD:**
- **Continuous Integration (CI)**: Automatically test code changes
- **Continuous Deployment (CD)**: Automatically deploy to production
- **Pipelines**: Automated workflows

**CI/CD Concepts:**
```
Code Commit → Build → Test → Deploy → Monitor
```

**CI/CD Tools:**
- **GitHub Actions**: Integrated with GitHub
- **GitLab CI**: Integrated with GitLab
- **Jenkins**: Self-hosted, flexible
- **CircleCI**: Cloud-based
- **Travis CI**: Cloud-based

### Code Testing for ML

**Unit Testing:**
Test individual functions and components.

```python
import pytest
import numpy as np
from src.features import preprocess_data, create_features

def test_preprocess_data():
    """Test data preprocessing function"""
    input_data = np.array([1, 2, 3, None, 5])
    result = preprocess_data(input_data)
    assert result is not None
    assert len(result) == 5
    assert np.isnan(result[3]) == False  # Missing value handled

def test_create_features():
    """Test feature engineering function"""
    data = np.array([[1, 2], [3, 4]])
    features = create_features(data)
    assert features.shape[0] == 2
    assert features.shape[1] > 2  # More features created
```

**Integration Testing:**
Test pipeline components work together.

```python
def test_training_pipeline():
    """Test complete training pipeline"""
    # Load data
    X, y = load_data()
    
    # Preprocess
    X_processed = preprocess_data(X)
    
    # Train
    model = train_model(X_processed, y)
    
    # Evaluate
    score = evaluate_model(model, X_processed, y)
    
    assert score > 0.8  # Minimum performance threshold
```

### Continuous Integration (CI) Implementation

**GitHub Actions Example:**
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Code quality checks
        run: |
          # Linting
          pip install flake8 black
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          black --check .
      
      - name: Format check
        run: |
          black --check .
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=xml
      
      - name: Run integration tests
        run: |
          pytest tests/integration/
```

**GitLab CI Example:**
```yaml
# .gitlab-ci.yml
stages:
  - test
  - validate
  - train

test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest tests/

validate:
  stage: validate
  script:
    - python scripts/validate_data.py
    - python scripts/validate_model.py

train:
  stage: train
  script:
    - python train.py
  only:
    - main
```

### ML-Specific CI: Testing the Model and Data

**Data Validation with Great Expectations:**
```python
import great_expectations as ge

# Create expectation suite
df = ge.read_csv('data/train.csv')

# Define expectations
df.expect_column_values_to_not_be_null('target')
df.expect_column_mean_to_be_between('feature1', min_value=0, max_value=100)
df.expect_column_values_to_be_in_set('category', ['A', 'B', 'C'])

# Save suite
df.save_expectation_suite('expectations.json')

# Validate new data
new_df = ge.read_csv('data/new_data.csv')
results = new_df.validate(expectation_suite='expectations.json')

if not results['success']:
    raise ValueError("Data validation failed!")
```

**Model Validation/Drift Tests:**
```python
def validate_model_performance(new_model, baseline_model, test_data):
    """Compare new model to baseline"""
    # Evaluate new model
    new_score = evaluate_model(new_model, test_data)
    
    # Evaluate baseline
    baseline_score = evaluate_model(baseline_model, test_data)
    
    # Check if new model is better
    if new_score < baseline_score * 0.95:  # 5% degradation threshold
        raise ValueError(f"New model performance ({new_score:.3f}) "
                        f"is worse than baseline ({baseline_score:.3f})")
    
    return new_score
```

**Using CML (Continuous Machine Learning):**
```yaml
# .github/workflows/cml.yml
name: CML
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: iterative/setup-cml@v1
      
      - name: Train model
        run: |
          pip install -r requirements.txt
          python train.py
      
      - name: Create CML report
        uses: iterative/action-cml@v0
        with:
          file: report.md
          token: ${{ secrets.GITHUB_TOKEN }}
```

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
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
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
      
      - name: Lint code
        run: |
          pip install flake8 black
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          black --check .
      
      - name: Data validation
        run: python scripts/validate_data.py
      
      - name: Train model
        run: python train.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      
      - name: Upload model artifacts
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/
```

### ML-Specific CI/CD

```yaml
# Additional ML-specific steps
- name: Model validation
  run: python scripts/validate_model.py

- name: Performance check
  run: python scripts/check_performance.py

- name: Register model
  run: python scripts/register_model.py
  env:
    MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

## Data Pipeline

### Data Validation

```python
try:
    import great_expectations as ge
    
    # Create expectation suite
    df = ge.read_csv('data/train.csv')
    
    # Define expectations
    df.expect_column_values_to_not_be_null('target')
    df.expect_column_mean_to_be_between('feature1', min_value=0, max_value=100)
    df.expect_column_values_to_be_in_set('category', ['A', 'B', 'C'])
    
    # Save suite
    df.save_expectation_suite('expectations.json')
    
    # Validate new data
    new_df = ge.read_csv('data/new_data.csv')
    results = new_df.validate(expectation_suite='expectations.json')
    print(results)
except ImportError:
    print("Install: pip install great-expectations")
```

## Reproducibility

### Environment Management

```python
# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
tensorflow==2.13.0

# Or use conda
# environment.yml
"""
name: ml-env
dependencies:
  - python=3.9
  - numpy=1.24.0
  - pandas=2.0.0
  - scikit-learn=1.3.0
"""
```

### Random Seeds

```python
import numpy as np
import random
import tensorflow as tf

# Set all random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
```

### Reproducible Pipeline

```python
# dvc.yaml (DVC pipeline)
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - data/raw
      - prepare.py
    outs:
      - data/prepared
    
  train:
    cmd: python train.py
    deps:
      - data/prepared
      - train.py
    params:
      - train.epochs
      - train.lr
    outs:
      - models/model.pkl
    metrics:
      - metrics/accuracy.json
```

### Cookiecutter for Data Science

Cookiecutter provides standardized project templates for data science projects.

**Installation:**
```bash
pip install cookiecutter
```

**Create Project:**
```bash
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

**Project Structure:**
```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── reports/
│   └── figures/
├── requirements.txt
└── README.md
```

**Benefits:**
- Standardized structure
- Best practices built-in
- Easy collaboration
- Quick project setup

---

## Practice Exercises

### Exercise 1: Track Experiment with MLflow

**Task:** Log a complete training run with MLflow.

**Solution:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="random_forest_experiment"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# View: mlflow ui
```

### Exercise 2: Set Up DVC Pipeline

**Task:** Create reproducible data pipeline with DVC.

**Solution:**
```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw/train.csv

# Create pipeline
dvc run -n prepare -d data/raw -o data/prepared python prepare.py
dvc run -n train -d data/prepared -o models/model.pkl python train.py

# Reproduce
dvc repro
```

---

## Key Takeaways

1. **Version Control**: Track code, data, and models (Git + DVC)
2. **Experiment Tracking**: Learn from past experiments (MLflow, W&B)
3. **Model Registry**: Manage model versions and stages
4. **CI/CD**: Automate testing and deployment
5. **Reproducibility**: Same inputs = same outputs
6. **Data Validation**: Ensure data quality
7. **Monitoring**: Track model performance in production

---

## Best Practices

### Version Control
- Use Git for code
- Use DVC for data and models
- Commit frequently with clear messages
- Tag releases

### Experiment Tracking
- Log all hyperparameters
- Log all metrics
- Log artifacts (plots, data samples)
- Use descriptive run names

### Reproducibility
- Set random seeds
- Version data
- Document environment
- Use containers (Docker)

### CI/CD
- Test code changes
- Validate data
- Check model performance
- Automate deployment

---

## Next Steps

- Set up MLflow for your projects
- Implement DVC for data versioning
- Create CI/CD pipelines
- Practice with real projects
- Explore advanced MLOps tools
- Practice with [16-projects-beginner](../16-projects-beginner/README.md)

**Remember**: MLOps makes ML production-ready! Invest time in proper tooling and practices.

