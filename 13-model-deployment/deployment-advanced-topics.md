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

