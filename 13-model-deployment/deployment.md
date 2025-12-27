# Model Deployment Complete Guide

Comprehensive guide to deploying machine learning models to production.

## Table of Contents

- [Model Serialization](#model-serialization)
- [REST APIs](#rest-apis)
- [Docker](#docker)
- [Cloud Deployment](#cloud-deployment)
- [Practice Exercises](#practice-exercises)

---

## Model Serialization

### Saving Models

```python
import pickle
import joblib
from tensorflow import keras

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Method 1: Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 2: Joblib (better for NumPy arrays)
joblib.dump(model, 'model.joblib')

# Keras models
keras_model = keras.Sequential([...])
keras_model.save('model.h5')
keras_model.save_weights('weights.h5')

# Loading
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

model = joblib.load('model.joblib')
keras_model = keras.models.load_model('model.h5')
```

---

## REST APIs

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load('model.joblib')

# Create app
app = FastAPI()

# Define request model
class PredictionRequest(BaseModel):
    features: list[float]

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        return {
            "prediction": int(prediction),
            "probabilities": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn main:app --reload
```

### Flask Example

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Docker

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Building and Running

```bash
# Build image
docker build -t ml-api .

# Run container
docker run -p 8000:8000 ml-api

# With docker-compose
docker-compose up
```

---

## Cloud Deployment

### Heroku

```python
# Procfile
web: uvicorn main:app --host=0.0.0.0 --port=$PORT

# Deploy
git push heroku main
```

### AWS Lambda

```python
# Lambda handler
def lambda_handler(event, context):
    features = event['features']
    prediction = model.predict([features])[0]
    return {'prediction': int(prediction)}
```

---

## Practice Exercises

### Exercise 1: Deploy Model with FastAPI

**Task:** Create FastAPI service for a trained model.

**Solution:**
```python
# See FastAPI example above
# Test with: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [1,2,3,4]}'
```

---

## Key Takeaways

1. **Serialization**: Save models properly
2. **APIs**: Expose models as services
3. **Docker**: Containerize applications
4. **Cloud**: Deploy to production

---

## Next Steps

- Practice deploying models
- Experiment with different platforms
- Move to [14-mlops-basics](../14-mlops-basics/README.md)

**Remember**: Deployment is as important as training!

