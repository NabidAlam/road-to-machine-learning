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

