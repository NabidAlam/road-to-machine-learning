# Complete Model Deployment Project Tutorial

Step-by-step walkthrough of deploying a machine learning model to production.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Prepare Model](#step-1-prepare-model)
- [Step 2: Create FastAPI Service](#step-2-create-fastapi-service)
- [Step 3: Containerize with Docker](#step-3-containerize-with-docker)
- [Step 4: Deploy to Cloud](#step-4-deploy-to-cloud)
- [Step 5: Monitor and Test](#step-5-monitor-and-test)

---

## Project Overview

**Project**: Deploy ML Model as REST API

**Goals**: Create production-ready API service

---

## Step 1: Prepare Model

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')
```

---

## Step 2: Create FastAPI Service

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.joblib')

@app.post("/predict")
async def predict(features: list[float]):
    prediction = model.predict([features])[0]
    return {"prediction": int(prediction)}
```

---

## Step 3: Containerize with Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 4: Deploy to Cloud

```bash
# Build and push
docker build -t ml-api .
docker push ml-api

# Deploy (platform-specific)
```

---

## Step 5: Monitor and Test

```python
import requests

response = requests.post('http://localhost:8000/predict', json={'features': [1,2,3,4]})
print(response.json())
```

---

**Congratulations!** You've deployed a model to production!

