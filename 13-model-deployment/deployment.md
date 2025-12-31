# Model Deployment Complete Guide

Comprehensive guide to deploying machine learning models to production.

## Table of Contents

- [Introduction to Model Deployment](#introduction-to-model-deployment)
- [Model Serialization](#model-serialization)
- [REST APIs](#rest-apis)
- [Docker](#docker)
- [Cloud Deployment](#cloud-deployment)
- [Model Monitoring](#model-monitoring)
- [Best Practices](#best-practices)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Model Deployment

### Why Deploy Models?

- **Make Models Useful**: Models need to serve predictions
- **Real-World Impact**: Deploy to help users
- **Production Challenges**: Different from development
- **Scalability**: Handle multiple requests

### Deployment Considerations

- **Latency**: Response time requirements
- **Throughput**: Requests per second
- **Reliability**: Uptime and error handling
- **Scalability**: Handle load spikes
- **Monitoring**: Track performance and errors
- **Versioning**: Manage model updates

### Deployment Options

1. **REST API**: Most common, flexible
2. **Batch Processing**: For non-real-time predictions
3. **Edge Deployment**: On-device inference
4. **Serverless**: AWS Lambda, Cloud Functions
5. **Containerized**: Docker, Kubernetes

---

## Model Serialization

### Saving Models

Proper serialization is crucial for deployment.

```python
import pickle
import joblib
from tensorflow import keras
import json

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Method 1: Pickle (Python native)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 2: Joblib (better for NumPy arrays, faster)
joblib.dump(model, 'model.joblib')
joblib.dump(model, 'model.joblib', compress=3)  # With compression

# Method 3: ONNX (cross-platform)
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open('model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())
except ImportError:
    print("Install: pip install skl2onnx onnx")

# Keras models
keras_model = keras.Sequential([...])
keras_model.save('model.h5')  # HDF5 format
keras_model.save('model.keras')  # New Keras format
keras_model.save_weights('weights.h5')  # Only weights

# PyTorch models
import torch
torch.save(model.state_dict(), 'model.pth')  # Recommended
torch.save(model, 'model_full.pth')  # Not recommended

# Loading
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

model = joblib.load('model.joblib')
keras_model = keras.models.load_model('model.h5')

# PyTorch
model = ModelClass()  # Create architecture
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Model Metadata

```python
# Save model metadata
metadata = {
    'model_version': '1.0',
    'training_date': '2024-01-01',
    'features': list(feature_names),
    'accuracy': 0.95,
    'framework': 'scikit-learn',
    'python_version': '3.9'
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## REST APIs

### FastAPI Example

FastAPI is modern, fast, and provides automatic documentation.

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model = joblib.load('model.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Create app
app = FastAPI(
    title="ML Model API",
    description="API for machine learning model predictions",
    version="1.0.0"
)

# Define request/response models
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature values", min_items=1)
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: List[float]
    confidence: float

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on input features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = np.array(request.features).reshape(1, -1)
        
        # Validate feature count
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(request.features)
        if len(request.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {len(request.features)}"
            )
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        confidence = max(probabilities)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probabilities=probabilities,
            confidence=float(confidence)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Health check
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    }

# Batch prediction
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_list = [np.array(req.features) for req in requests]
        features_array = np.array(features_list)
        
        predictions = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "probabilities": prob.tolist(),
                "confidence": float(max(prob))
            })
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### Flask Example

Flask is simpler but less feature-rich than FastAPI.

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### API Testing

```python
import requests

# Test prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={'features': [1.0, 2.0, 3.0, 4.0]}
)
print(response.json())

# Test health
response = requests.get('http://localhost:8000/health')
print(response.json())
```

---

## Docker

### Dockerfile

Containerization ensures consistent environments.

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.joblib
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Building and Running

```bash
# Build image
docker build -t ml-api:latest .

# Run container
docker run -d -p 8000:8000 --name ml-api ml-api:latest

# View logs
docker logs ml-api

# Stop container
docker stop ml-api

# With docker-compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Cloud Deployment

### Heroku

Simple deployment platform.

```python
# Procfile
web: uvicorn main:app --host=0.0.0.0 --port=$PORT

# runtime.txt
python-3.9.16

# Deploy
# heroku create ml-api
# git push heroku main
# heroku logs --tail
```

### AWS Lambda

Serverless deployment.

```python
# lambda_handler.py
import json
import joblib
import numpy as np

# Load model (at module level for reuse)
model = joblib.load('model.joblib')

def lambda_handler(event, context):
    """AWS Lambda handler"""
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        features = np.array(body['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction),
                'probabilities': probability
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Run

Container-based serverless.

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-api
gcloud run deploy ml-api --image gcr.io/PROJECT_ID/ml-api --platform managed
```

### Azure Container Instances

```bash
# Build and deploy
az acr build --registry myregistry --image ml-api:latest .
az container create --resource-group mygroup --name ml-api --image myregistry.azurecr.io/ml-api:latest
```

## Model Monitoring

### Logging Predictions

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_prediction(features, prediction, probability):
    """Log prediction for monitoring"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': features.tolist(),
        'prediction': int(prediction),
        'probability': float(probability)
    }
    logger.info(json.dumps(log_entry))
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor API performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

# Use decorator
@app.post("/predict")
@monitor_performance
async def predict(request: PredictionRequest):
    # ... prediction code ...
```

### Data Drift Detection

```python
def detect_data_drift(new_features, training_features):
    """Detect if new data distribution differs from training"""
    from scipy import stats
    
    drift_scores = []
    for i in range(new_features.shape[1]):
        ks_statistic, p_value = stats.ks_2samp(
            training_features[:, i],
            new_features[:, i]
        )
        drift_scores.append(p_value)
    
    return np.mean(drift_scores) < 0.05  # Significant drift
```

---

## Best Practices

### API Design
- Use RESTful conventions
- Version your API (`/v1/predict`)
- Provide clear error messages
- Include health checks
- Document with OpenAPI/Swagger

### Security
- Validate all inputs
- Rate limiting
- Authentication/Authorization
- HTTPS in production
- Input sanitization

### Performance
- Cache predictions when possible
- Use async for I/O operations
- Batch processing for multiple requests
- Optimize model size
- Use model quantization

### Reliability
- Health checks
- Graceful degradation
- Error handling
- Retry logic
- Circuit breakers

## Practice Exercises

### Exercise 1: Deploy Model with FastAPI

**Task:** Create complete FastAPI service for a trained model.

**Solution:**
```python
# See FastAPI example above
# Test with:
# curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"features": [1,2,3,4]}'

# View docs: http://localhost:8000/docs
```

### Exercise 2: Containerize with Docker

**Task:** Create Docker image for ML API.

**Solution:**
```dockerfile
# See Dockerfile example above
# Build: docker build -t ml-api .
# Run: docker run -p 8000:8000 ml-api
```

---

## Key Takeaways

1. **Serialization**: Save models properly (pickle, joblib, H5, ONNX)
2. **APIs**: Expose models as REST services (FastAPI, Flask)
3. **Docker**: Containerize for consistency and portability
4. **Cloud**: Deploy to production platforms (AWS, GCP, Azure)
5. **Monitoring**: Track performance and detect issues
6. **Security**: Validate inputs, use authentication
7. **Performance**: Optimize for latency and throughput

---

## Next Steps

- Practice deploying models to different platforms
- Experiment with containerization
- Set up monitoring and logging
- Learn about model versioning
- Explore edge deployment
- Move to [14-mlops-basics](../14-mlops-basics/README.md)

**Remember**: Deployment is as important as training! A great model is useless if it can't serve predictions.

