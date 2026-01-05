# MLOps Quick Reference Guide

Quick reference for MLOps tools and practices.

## Table of Contents

- [Version Control](#version-control)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [CI/CD](#cicd)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Version Control

### DVC Commands

```bash
# Initialize
dvc init

# Track files
dvc add data/train.csv

# Create pipeline
dvc run -n step -d input -o output python script.py

# Reproduce
dvc repro
```

### Git LFS

```bash
# Install
git lfs install

# Track
git lfs track "*.pkl"
```

---

## Experiment Tracking

### MLflow

```python
import mlflow

mlflow.set_experiment("experiment")
with mlflow.start_run():
    mlflow.log_param("param", value)
    mlflow.log_metric("metric", value)
    mlflow.sklearn.log_model(model, "model")
```

### Weights & Biases

```python
import wandb
wandb.init(project="project")
wandb.log({"metric": value})
```

---

## Model Registry

```python
# Register
mlflow.register_model("runs:/<id>/model", "ModelName")

# Transition
client.transition_model_version_stage("ModelName", 1, "Production")
```

---

## CI/CD

### GitHub Actions

```yaml
- name: Run tests
  run: pytest

- name: Train model
  run: python train.py
```

---

## Common Issues & Solutions

### Issue 1: Large Files in Git

**Solution**: Use DVC or Git LFS

### Issue 2: Lost Experiments

**Solution**: Use MLflow from the start

---

## Best Practices Checklist

- [ ] Version control code, data, models
- [ ] Track all experiments
- [ ] Use model registry
- [ ] Set up CI/CD
- [ ] Document everything
- [ ] Monitor models in production

---

**Remember**: MLOps makes ML production-ready!

