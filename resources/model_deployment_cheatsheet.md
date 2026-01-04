# Model Deployment Cheatsheet

Comprehensive quick reference for deploying ML models to production.

## Table of Contents

- [Model Serialization](#model-serialization)
- [API Development](#api-development)
- [Containerization](#containerization)
- [Cloud Deployment](#cloud-deployment)
- [A/B Testing](#ab-testing)
- [Model Serving](#model-serving)
- [Monitoring](#monitoring)

---

## Model Serialization

### Scikit-learn

```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')

# Save with metadata
joblib.dump({
    'model': model,
    'scaler': scaler,
    'version': '1.0.0',
    'features': feature_names
}, 'model_package.joblib')
```

### TensorFlow/Keras

```python
# Save entire model
model.save('model.h5')
model.save('model/')  # SavedModel format

# Load model
model = tf.keras.models.load_model('model.h5')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

### PyTorch

```python
import torch

# Save model state
torch.save(model.state_dict(), 'model.pth')

# Load model state
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```

### ONNX (Cross-platform)

```python
# Convert to ONNX
import onnx
from skl2onnx import convert_sklearn

onnx_model = convert_sklearn(model, 'model.onnx')

# Load ONNX
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

---

## API Development

### FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML API")

# Load model
model = joblib.load('model.joblib')

# Request model
class PredictionRequest(BaseModel):
    features: list[float]

# Response model
class PredictionResponse(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess
        features = np.array(request.features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn main:app --reload
```

### Flask

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Streamlit

```python
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model.joblib')

st.title("ML Prediction App")

# Input form
features = []
for i in range(4):
    feature = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(feature)

# Predict button
if st.button("Predict"):
    prediction = model.predict([features])[0]
    st.success(f"Prediction: {prediction}")

# Run: streamlit run app.py
```

---

## Containerization

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Commands

```bash
# Build image
docker build -t ml-api:latest .

# Run container
docker run -p 8000:8000 ml-api:latest

# Run in background
docker run -d -p 8000:8000 --name ml-api ml-api:latest

# View logs
docker logs ml-api

# Stop container
docker stop ml-api

# Remove container
docker rm ml-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.joblib
    volumes:
      - ./models:/models
    restart: unless-stopped
```

---

## Cloud Deployment

### AWS SageMaker

```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Create model
sklearn_model = SKLearnModel(
    model_data='s3://bucket/model.tar.gz',
    role=sagemaker.get_execution_role(),
    entry_point='inference.py',
    framework_version='0.23-1'
)

# Deploy
predictor = sklearn_model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1
)

# Predict
prediction = predictor.predict(data)
```

### Google Cloud AI Platform

```python
from google.cloud import aiplatform

# Deploy model
endpoint = aiplatform.Endpoint.create(
    display_name="my-endpoint"
)

# Upload model
model = aiplatform.Model.upload(
    display_name="my-model",
    artifact_uri="gs://bucket/model"
)

# Deploy to endpoint
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4"
)
```

### Azure ML

```python
from azureml.core import Model, Workspace
from azureml.core.webservice import AciWebservice, Webservice

# Register model
model = Model.register(
    workspace=ws,
    model_path="model.pkl",
    model_name="my-model"
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name="my-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
```

---

## A/B Testing

### Statistical Significance

```python
from scipy import stats
import numpy as np

def analyze_ab_test(conversions_a, total_a, conversions_b, total_b):
    """Z-test for two proportions"""
    rate_a = conversions_a / total_a
    rate_b = conversions_b / total_b
    
    # Pooled proportion
    pooled = (conversions_a + conversions_b) / (total_a + total_b)
    se = np.sqrt(pooled * (1 - pooled) * (1/total_a + 1/total_b))
    
    # Z-score and p-value
    z = (rate_b - rate_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'rate_a': rate_a,
        'rate_b': rate_b,
        'lift': (rate_b - rate_a) / rate_a,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Multi-Armed Bandit

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
```

---

## Model Serving

### TensorFlow Serving

```bash
# Install TensorFlow Serving
docker pull tensorflow/serving

# Serve model
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving

# Predict
curl -d '{"instances": [[1, 2, 3]]}' \
  -X POST http://localhost:8501/v1/models/my_model:predict
```

### TorchServe

```python
# Create model archive
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model.pth \
  --handler handler.py \
  --export-path model_store

# Start server
torchserve --start \
  --model-store model_store \
  --models my_model=my_model.mar
```

---

## Monitoring

### Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request: {request.features}")
    try:
        prediction = model.predict([request.features])[0]
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    with prediction_latency.time():
        prediction = model.predict([request.features])[0]
        prediction_counter.inc()
    return {"prediction": prediction}

# Start metrics server
start_http_server(8000)
```

---

## Best Practices

### Deployment Checklist

- [ ] Serialize model properly
- [ ] Version model artifacts
- [ ] Include preprocessing in API
- [ ] Add input validation
- [ ] Implement error handling
- [ ] Add health checks
- [ ] Set up logging
- [ ] Configure monitoring
- [ ] Test API endpoints
- [ ] Document API

### Security Checklist

- [ ] Validate all inputs
- [ ] Rate limit API calls
- [ ] Use authentication
- [ ] Encrypt sensitive data
- [ ] Keep dependencies updated
- [ ] Scan for vulnerabilities
- [ ] Use HTTPS
- [ ] Implement CORS properly

---

## Quick Reference

### Model Formats

| Format | Library | Use Case |
|--------|---------|----------|
| **joblib** | scikit-learn | Python-only, fast |
| **pickle** | Python | General Python objects |
| **H5** | Keras | TensorFlow models |
| **ONNX** | Cross-platform | Multi-language deployment |
| **SavedModel** | TensorFlow | TensorFlow serving |

### Deployment Options

| Option | Best For | Complexity |
|--------|----------|------------|
| **FastAPI/Flask** | Custom APIs | Medium |
| **Docker** | Containerized apps | Medium |
| **AWS SageMaker** | AWS ecosystem | High |
| **Google Cloud AI** | GCP ecosystem | High |
| **Azure ML** | Azure ecosystem | High |
| **Kubernetes** | Scalable deployments | High |

---

**Remember**: Deploy models that are tested, monitored, and secure!

