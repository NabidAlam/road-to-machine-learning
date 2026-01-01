# Advanced Model Deployment Topics

Comprehensive guide to advanced deployment techniques and best practices.

## Table of Contents

- [Model Optimization for Deployment](#model-optimization-for-deployment)
- [Advanced API Patterns](#advanced-api-patterns)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Edge Deployment](#edge-deployment)
- [A/B Testing](#ab-testing)
- [Model Versioning](#model-versioning)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Model Optimization for Deployment

### Model Quantization

```python
# TensorFlow Lite quantization
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# PyTorch quantization
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### Model Pruning

```python
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

---

## Advanced API Patterns

### Async Processing

```python
from fastapi import BackgroundTasks
import asyncio

@app.post("/predict/async")
async def predict_async(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Async prediction with background processing"""
    task_id = str(uuid.uuid4())
    
    # Process in background
    background_tasks.add_task(process_prediction, task_id, request.features)
    
    return {"task_id": task_id, "status": "processing"}

async def process_prediction(task_id: str, features: List[float]):
    # Long-running prediction
    await asyncio.sleep(1)
    prediction = model.predict([features])[0]
    # Store result
    results[task_id] = prediction
```

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(features_hash: str):
    """Cache predictions"""
    # Decode features from hash
    features = decode_features(features_hash)
    return model.predict([features])[0]

def hash_features(features: List[float]) -> str:
    """Create hash of features for caching"""
    return hashlib.md5(str(features).encode()).hexdigest()
```

---

## Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Edge Deployment

### TensorFlow Lite

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use on mobile/edge devices
```

### ONNX Runtime

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
inputs = {session.get_inputs()[0].name: input_data}
outputs = session.run(None, inputs)
```

---

## A/B Testing

### Introduction to A/B Testing for ML

**A/B Testing** (also called split testing) compares two versions of a model to determine which performs better. It's essential for safely deploying new models and measuring their impact.

**Key Concepts:**
- **Control (A)**: Current model in production
- **Treatment (B)**: New model to test
- **Traffic Split**: Percentage of traffic to each model
- **Statistical Significance**: Confidence that results are real, not random

### Why A/B Test ML Models?

1. **Risk Reduction**: Test new models on subset of users
2. **Performance Validation**: Verify improvements in production
3. **Business Impact**: Measure actual business metrics, not just accuracy
4. **Gradual Rollout**: Deploy safely with ability to rollback

### Basic A/B Test Setup

```python
import numpy as np
import pandas as pd
from scipy import stats

def ab_test_ml_models(control_predictions, treatment_predictions, 
                      control_actuals, treatment_actuals, 
                      metric='accuracy', alpha=0.05):
    """
    Perform A/B test comparing two ML models
    
    Parameters:
    - control_predictions: Predictions from control model (A)
    - treatment_predictions: Predictions from treatment model (B)
    - control_actuals: Actual values for control group
    - treatment_actuals: Actual values for treatment group
    - metric: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    - alpha: Significance level (default 0.05)
    """
    from sklearn.metrics import (accuracy_score, precision_score, 
                                recall_score, f1_score, roc_auc_score)
    
    # Calculate metrics
    metric_funcs = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }
    
    metric_func = metric_funcs[metric]
    
    control_score = metric_func(control_actuals, control_predictions)
    treatment_score = metric_func(treatment_actuals, treatment_predictions)
    
    # Statistical test (t-test for means)
    # For binary classification, use proportion test
    if metric == 'accuracy':
        from statsmodels.stats.proportion import proportions_ztest
        
        # Count successes
        control_success = np.sum(control_predictions == control_actuals)
        treatment_success = np.sum(treatment_predictions == treatment_actuals)
        
        # Z-test for proportions
        count = np.array([control_success, treatment_success])
        nobs = np.array([len(control_predictions), len(treatment_predictions)])
        z_stat, p_value = proportions_ztest(count, nobs)
    else:
        # For other metrics, use t-test
        # Note: This is simplified - in practice, you'd need proper sampling
        t_stat, p_value = stats.ttest_ind(
            [control_score], [treatment_score]
        )
    
    # Results
    improvement = treatment_score - control_score
    improvement_pct = (improvement / control_score) * 100
    
    is_significant = p_value < alpha
    
    results = {
        'control_score': control_score,
        'treatment_score': treatment_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        'is_significant': is_significant,
        'recommendation': 'Use treatment' if (is_significant and improvement > 0) else 'Keep control'
    }
    
    return results

# Example usage
# control_pred = model_a.predict(X_test)
# treatment_pred = model_b.predict(X_test)
# results = ab_test_ml_models(control_pred, treatment_pred, y_test, y_test)
# print(results)
```

### Traffic Splitting

```python
import random
import hashlib

def assign_to_group(user_id, split_ratio=0.5, seed=42):
    """
    Deterministically assign user to A or B group
    
    Parameters:
    - user_id: Unique identifier for user/request
    - split_ratio: Percentage of traffic to treatment (B)
    - seed: Random seed for reproducibility
    """
    # Create deterministic hash
    hash_obj = hashlib.md5(f"{user_id}_{seed}".encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Assign based on hash
    if (hash_int % 100) < (split_ratio * 100):
        return 'B'  # Treatment
    else:
        return 'A'  # Control

# Example
user_ids = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
for user_id in user_ids:
    group = assign_to_group(user_id, split_ratio=0.5)
    print(f"{user_id}: {group}")
```

### Statistical Significance

**Key Concepts:**

1. **Null Hypothesis (H₀)**: No difference between models
2. **Alternative Hypothesis (H₁)**: Models are different
3. **P-value**: Probability of observing results if H₀ is true
4. **Alpha (α)**: Significance level (typically 0.05)
5. **Power**: Probability of detecting a real difference

```python
def calculate_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test
    
    Parameters:
    - effect_size: Minimum detectable effect (e.g., 0.05 for 5% improvement)
    - alpha: Significance level
    - power: Statistical power (1 - beta)
    """
    from statsmodels.stats.power import TTestIndPower
    
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0  # Equal group sizes
    )
    
    return int(np.ceil(sample_size))

# Example: How many samples needed to detect 5% improvement?
sample_size = calculate_sample_size(effect_size=0.05)
print(f"Required sample size per group: {sample_size}")
```

### Multi-Armed Bandits

**Problem with Traditional A/B Testing:**
- Fixed traffic split (50/50) even if one model is clearly better
- Wastes traffic on inferior model

**Solution: Multi-Armed Bandits**
- Dynamically adjust traffic based on performance
- Allocate more traffic to better-performing model
- Still maintains statistical validity

```python
class EpsilonGreedyBandit:
    """
    Epsilon-greedy multi-armed bandit for ML model selection
    """
    def __init__(self, n_arms=2, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon  # Exploration rate
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
    
    def select_arm(self):
        """Select which model to use"""
        if random.random() < self.epsilon:
            # Explore: random selection
            return random.randint(0, self.n_arms - 1)
        else:
            # Exploit: select best performing
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """Update model performance estimate"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Running average
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# Example usage
bandit = EpsilonGreedyBandit(n_arms=2, epsilon=0.1)

for request in range(1000):
    # Select model
    model_idx = bandit.select_arm()
    
    # Get prediction and actual outcome
    prediction = models[model_idx].predict([features])
    actual = get_actual_outcome()
    
    # Calculate reward (e.g., accuracy)
    reward = 1.0 if prediction == actual else 0.0
    
    # Update bandit
    bandit.update(model_idx, reward)
    
    # Periodically check performance
    if request % 100 == 0:
        print(f"Model 0 value: {bandit.values[0]:.3f}")
        print(f"Model 1 value: {bandit.values[1]:.3f}")
```

### Sequential Testing

**Problem**: Traditional tests require fixed sample size upfront  
**Solution**: Sequential testing allows early stopping when significance is reached

```python
from scipy.stats import norm

def sequential_test(control_successes, control_total,
                   treatment_successes, treatment_total,
                   alpha=0.05):
    """
    Sequential A/B test with early stopping
    
    Returns: (is_significant, should_continue)
    """
    # Calculate proportions
    p_control = control_successes / control_total
    p_treatment = treatment_successes / treatment_total
    
    # Calculate standard error
    se = np.sqrt(
        (p_control * (1 - p_control) / control_total) +
        (p_treatment * (1 - p_treatment) / treatment_total)
    )
    
    # Z-score
    z_score = (p_treatment - p_control) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    is_significant = p_value < alpha
    
    # Decision: Stop if significant or if we have enough data
    min_samples = 100  # Minimum samples before stopping
    should_continue = not (is_significant and 
                          control_total >= min_samples and 
                          treatment_total >= min_samples)
    
    return {
        'is_significant': is_significant,
        'p_value': p_value,
        'should_continue': should_continue,
        'control_rate': p_control,
        'treatment_rate': p_treatment
    }
```

### Interpreting A/B Test Results

**Common Pitfalls:**

1. **Stopping Too Early**: Need sufficient sample size
2. **Multiple Comparisons**: Testing many variants increases false positives
3. **Peeking**: Checking results repeatedly inflates p-values
4. **Ignoring Business Metrics**: Technical metrics may not reflect business impact

**Best Practices:**

```python
def interpret_ab_test_results(results):
    """
    Interpret and present A/B test results
    """
    print("A/B Test Results")
    print("=" * 50)
    print(f"Control Score: {results['control_score']:.4f}")
    print(f"Treatment Score: {results['treatment_score']:.4f}")
    print(f"Improvement: {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"Statistically Significant: {results['is_significant']}")
    print(f"\nRecommendation: {results['recommendation']}")
    
    if results['is_significant']:
        if results['improvement'] > 0:
            print("\nTreatment model is significantly better!")
            print("   Safe to deploy to all traffic.")
        else:
            print("\nTreatment model is significantly worse!")
            print("   Do not deploy. Investigate issues.")
    else:
        print("\nNo significant difference detected.")
        print("   May need more data or larger effect size.")
        print("   Consider continuing test or keeping control.")
```

### A/B Testing Best Practices

1. **Define Success Metrics**: Before starting, define what success looks like
2. **Calculate Sample Size**: Ensure sufficient power to detect effects
3. **Random Assignment**: Use deterministic hashing for consistent assignment
4. **Monitor Continuously**: Watch for anomalies, but don't peek too early
5. **Set Duration**: Pre-determine test duration or use sequential testing
6. **Document Everything**: Record configuration, results, and decisions
7. **Consider Business Context**: Technical improvements may not translate to business value

### Example: Complete A/B Test Workflow

```python
# 1. Setup
control_model = load_model('model_v1.pkl')
treatment_model = load_model('model_v2.pkl')
split_ratio = 0.5

# 2. Assign traffic
def route_request(request_id, features):
    group = assign_to_group(request_id, split_ratio)
    
    if group == 'A':
        prediction = control_model.predict([features])[0]
        model_used = 'control'
    else:
        prediction = treatment_model.predict([features])[0]
        model_used = 'treatment'
    
    return prediction, model_used

# 3. Collect results
results_control = []
results_treatment = []

for request in requests:
    pred, model = route_request(request.id, request.features)
    actual = get_actual_outcome(request.id)
    
    if model == 'control':
        results_control.append((pred, actual))
    else:
        results_treatment.append((pred, actual))

# 4. Analyze
control_preds, control_actuals = zip(*results_control)
treatment_preds, treatment_actuals = zip(*results_treatment)

ab_results = ab_test_ml_models(
    control_preds, treatment_preds,
    control_actuals, treatment_actuals
)

# 5. Make decision
interpret_ab_test_results(ab_results)

if ab_results['recommendation'] == 'Use treatment':
    deploy_model(treatment_model, traffic_percentage=1.0)
else:
    keep_control_model()
```

---

## A/B Testing

### Model Versioning

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            'v1': joblib.load('model_v1.joblib'),
            'v2': joblib.load('model_v2.joblib')
        }
        self.traffic_split = {'v1': 0.5, 'v2': 0.5}
    
    def route(self, features):
        import random
        version = random.choices(
            list(self.traffic_split.keys()),
            weights=list(self.traffic_split.values())
        )[0]
        return self.models[version].predict([features])[0]
```

---

## Model Versioning

### Version Management

```python
import os
from datetime import datetime

def save_model_version(model, version=None):
    """Save model with versioning"""
    if version is None:
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = f'models/v{version}/model.joblib'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'accuracy': model.score(X_test, y_test)
    }
    
    with open(f'models/v{version}/metadata.json', 'w') as f:
        json.dump(metadata, f)
```

---

## AWS SageMaker Comprehensive Guide

### Introduction to AWS SageMaker

AWS SageMaker is a fully managed machine learning service that provides tools to build, train, and deploy ML models at scale.

### Key Features

- **Managed Infrastructure**: No server management
- **Built-in Algorithms**: Pre-optimized ML algorithms
- **AutoML**: Automated model building
- **Model Training**: Distributed training
- **Model Deployment**: One-click deployment
- **Model Monitoring**: Track model performance

### SageMaker Components

#### 1. SageMaker Studio

Integrated development environment for ML.

**Features:**
- Jupyter notebooks
- Data preparation
- Model building
- Training and deployment
- Monitoring

#### 2. SageMaker Notebooks

Managed Jupyter notebooks with pre-configured environments.

```python
# Example: Training a model in SageMaker
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

# Get execution role
role = get_execution_role()

# Create estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3'
)

# Train model
sklearn_estimator.fit({'training': 's3://bucket/training-data'})
```

#### 3. SageMaker Training

Managed training infrastructure.

**Training Job:**
```python
from sagemaker.tensorflow import TensorFlow

# Create TensorFlow estimator
tf_estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.8',
    py_version='py39'
)

# Start training
tf_estimator.fit({'training': 's3://bucket/data'})
```

#### 4. SageMaker Endpoints

Deploy models for real-time inference.

```python
# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Make predictions
result = predictor.predict(data)
print(result)

# Delete endpoint
predictor.delete_endpoint()
```

#### 5. SageMaker Batch Transform

Batch inference for large datasets.

```python
# Create transformer
transformer = sklearn_estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.large'
)

# Run batch transform
transformer.transform(
    data='s3://bucket/input-data',
    content_type='text/csv',
    split_type='Line'
)
```

### SageMaker Built-in Algorithms

**Supervised Learning:**
- Linear Learner
- XGBoost
- Factorization Machines
- Neural Topic Model

**Unsupervised Learning:**
- K-Means
- Principal Component Analysis
- Latent Dirichlet Allocation

**Deep Learning:**
- Image Classification
- Object Detection
- Semantic Segmentation

### SageMaker AutoML

Automated machine learning with AutoPilot.

```python
from sagemaker.automl.automl import AutoML

# Create AutoML job
automl = AutoML(
    role=role,
    target_attribute_name='target',
    output_path='s3://bucket/output',
    problem_type='BinaryClassification',
    max_candidates=10
)

# Start AutoML
automl.fit({'training': 's3://bucket/training-data'})
```

### SageMaker Model Registry

Manage model versions and metadata.

```python
from sagemaker.model_registry import ModelRegistry

# Register model
model_package = model_registry.register_model(
    model_package_group_name='my-models',
    model_artifact=model_artifact,
    inference_specification=inference_spec
)

# Approve model
model_registry.approve_model_package(
    model_package_arn=model_package['ModelPackageArn']
)
```

### Cost Optimization

**Strategies:**
1. Use Spot Instances for training
2. Right-size instances
3. Use batch transform for non-real-time
4. Monitor and stop unused endpoints
5. Use SageMaker Serverless Inference

**Spot Instances:**
```python
# Use spot instances (up to 90% savings)
estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    use_spot_instances=True,
    max_wait=3600,  # Max wait time
    max_run=7200     # Max training time
)
```

### Best Practices

1. **Data Preparation**: Use SageMaker Processing
2. **Feature Store**: Centralize features
3. **Experiments**: Track experiments with SageMaker Experiments
4. **Monitoring**: Use SageMaker Model Monitor
5. **Security**: Use IAM roles and VPC

### SageMaker vs Other Platforms

| Feature | SageMaker | GCP Vertex AI | Azure ML |
|---------|-----------|---------------|----------|
| **Ease of Use** | High | High | Medium |
| **Cost** | Pay-per-use | Pay-per-use | Pay-per-use |
| **Integration** | AWS ecosystem | GCP ecosystem | Azure ecosystem |
| **AutoML** | Yes | Yes | Yes |

---

## Common Pitfalls and Solutions

### Pitfall 1: Model Size Too Large

**Solution**: Quantization, pruning, use smaller models

### Pitfall 2: High Latency

**Solution**: Optimize model, use caching, batch processing

### Pitfall 3: Memory Issues

**Solution**: Limit batch size, use streaming, optimize model

---

## Key Takeaways

1. **Optimization**: Quantize and prune for deployment
2. **Caching**: Cache predictions for performance
3. **Versioning**: Manage model versions properly
4. **A/B Testing**: Compare model versions
5. **Edge Deployment**: Use TFLite/ONNX for mobile

---

**Remember**: Production deployment requires optimization and monitoring!

