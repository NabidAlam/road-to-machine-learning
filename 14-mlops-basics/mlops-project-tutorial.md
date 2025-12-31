# Complete MLOps Project Tutorial

Step-by-step walkthrough of setting up a complete MLOps pipeline.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Set Up Version Control](#step-1-set-up-version-control)
- [Step 2: Set Up Experiment Tracking](#step-2-set-up-experiment-tracking)
- [Step 3: Create Reproducible Pipeline](#step-3-create-reproducible-pipeline)
- [Step 4: Set Up CI/CD](#step-4-set-up-cicd)
- [Step 5: Model Registry](#step-5-model-registry)

---

## Project Overview

**Project**: Complete MLOps Pipeline Setup

**Goals**: Set up version control, tracking, and CI/CD

---

## Step 1: Set Up Version Control

```bash
# Initialize Git
git init

# Initialize DVC
dvc init

# Track data
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data"
```

---

## Step 2: Set Up Experiment Tracking

```python
import mlflow
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    model = train_model()
    accuracy = evaluate_model(model)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
```

---

## Step 3: Create Reproducible Pipeline

```bash
# Create DVC pipeline
dvc run -n prepare -d data/raw -o data/prepared python prepare.py
dvc run -n train -d data/prepared -o models/model.pkl python train.py
```

---

## Step 4: Set Up CI/CD

```yaml
# .github/workflows/ml.yml
name: ML Pipeline
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest
      - run: python train.py
```

---

## Step 5: Model Registry

```python
# Register model
mlflow.register_model("runs:/<run_id>/model", "MyModel")

# Transition to production
client.transition_model_version_stage("MyModel", 1, "Production")
```

---

**Congratulations!** You've set up a complete MLOps pipeline!

