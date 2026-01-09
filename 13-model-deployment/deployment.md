# Model Deployment Complete Guide

Comprehensive guide to deploying machine learning models to production.

## Table of Contents

- [Introduction to Model Deployment](#introduction-to-model-deployment)
- [Model Serialization](#model-serialization)
- [REST APIs](#rest-apis)
- [Docker](#docker)
- [Cloud Deployment](#cloud-deployment)
  - [AWS ECS Fargate for ML Model Serving](#aws-ecs-fargate-for-ml-model-serving)
  - [AWS Infrastructure as Code for ML](#aws-infrastructure-as-code-for-ml)
  - [AWS Observability for ML Systems](#aws-observability-for-ml-systems)
  - [AWS Security Best Practices for ML](#aws-security-best-practices-for-ml)
  - [AWS Cost Management for ML Workloads](#aws-cost-management-for-ml-workloads)
- [Orchestration and Scaling with Kubernetes](#orchestration-and-scaling-with-kubernetes)
- [Production Server Setup](#production-server-setup)
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

### Model Serving Architectures

**Online/Real-time Serving (REST APIs):**
- Immediate predictions
- Low latency requirements
- Interactive applications
- Example: User-facing applications, chatbots

**Batch Serving:**
- Process multiple predictions at once
- Scheduled or on-demand
- Higher throughput
- Example: Daily reports, bulk processing

**Deployment Patterns:**

**1. Shadow Deployment:**
- Deploy new model alongside production
- Route traffic to both models
- Compare performance
- No impact on users

**2. Canary Rollouts:**
- Gradually increase traffic to new model
- Start with small percentage (e.g., 5%)
- Monitor performance
- Increase if successful

**3. A/B Testing:**
- Split traffic between models
- Compare performance metrics
- Statistical significance testing
- Choose best performing model

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

### FastAPI Advanced Features

**1. Type Checking and Validation:**
```python
from fastapi import FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

app = FastAPI()

# Enum for validation
class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

# Advanced Pydantic model with validators
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=1, max_items=100)
    model_type: ModelType = ModelType.CLASSIFICATION
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) < 1:
            raise ValueError('At least one feature required')
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numbers')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0],
                "model_type": "classification",
                "threshold": 0.5
            }
        }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction with validated input"""
    # Pydantic automatically validates and converts types
    features = np.array(request.features)
    # ... prediction logic
    return {"prediction": 1}
```

**2. Query Parameters and Path Parameters:**
```python
from fastapi import Query, Path

@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(..., description="Item ID", gt=0),
    skip: int = Query(0, ge=0, description="Skip items"),
    limit: int = Query(10, ge=1, le=100, description="Limit items"),
    q: Optional[str] = Query(None, min_length=3, max_length=50)
):
    """Get item with path and query parameters"""
    return {
        "item_id": item_id,
        "skip": skip,
        "limit": limit,
        "q": q
    }
```

**3. Dependency Injection:**
```python
from fastapi import Depends

def get_model():
    """Dependency to load model"""
    return joblib.load('model.joblib')

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    model = Depends(get_model)
):
    """Predict using injected model"""
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}
```

**4. Background Tasks:**
```python
from fastapi import BackgroundTasks

def log_prediction(features, prediction):
    """Background task to log prediction"""
    with open('predictions.log', 'a') as f:
        f.write(f"{features},{prediction}\n")

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model)
):
    """Predict with background logging"""
    prediction = model.predict([request.features])[0]
    
    # Add background task
    background_tasks.add_task(log_prediction, request.features, prediction)
    
    return {"prediction": int(prediction)}
```

**5. Error Handling:**
```python
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Custom validation error handler"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Prediction logic
        pass
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
```

**6. CORS and Security:**
```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)
```

**7. Automatic API Documentation:**
FastAPI automatically generates OpenAPI/Swagger documentation:
- Interactive docs: `http://localhost:8000/docs` (Swagger UI)
- Alternative docs: `http://localhost:8000/redoc` (ReDoc)
- OpenAPI schema: `http://localhost:8000/openapi.json`

**8. Testing FastAPI:**
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
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

### Hugging Face Spaces (Free Model Hosting)

**What are Hugging Face Spaces?**

Hugging Face Spaces provides free hosting for ML model demos and applications. It's perfect for:
- **Portfolio projects**: Showcase your models
- **Quick demos**: Share models with others
- **Learning**: Practice deployment without costs
- **Community**: Share with the ML community

**Key Features:**
- **Free**: No credit card required
- **Easy**: Git push to deploy
- **Automatic**: Auto-deploys on push
- **Built-in UI**: Gradio or Streamlit interfaces
- **Public**: Share with anyone via URL

**Why Use Hugging Face Spaces?**
- Fastest way to deploy ML models
- Great for building portfolio
- Free hosting (no costs)
- Easy sharing and collaboration
- Industry-recognized platform

**Step 1: Install Hugging Face Hub**

```bash
pip install huggingface_hub
```

**Step 2: Create a Space**

```bash
# Login to Hugging Face
huggingface-cli login

# Create a new Space
huggingface-cli repo create my-sentiment-model --type space --space-sdk gradio
```

Or create via web interface:
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose SDK: Gradio or Streamlit
4. Set visibility: Public or Private

**Step 3: Create Your App**

**Option A: Gradio (Recommended for beginners)**

```python
# app.py
import gradio as gr
import joblib
import numpy as np

# Load your model
model = joblib.load("model.pkl")

def predict(text):
    """Make prediction"""
    # Preprocess text (example)
    features = preprocess_text(text)
    
    # Predict
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    # Format output
    labels = ["Negative", "Positive"]
    result = {
        labels[i]: float(prob) 
        for i, prob in enumerate(probability)
    }
    
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter text", placeholder="Type your text here..."),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis Model",
    description="Classify text as positive or negative sentiment",
    examples=[
        ["I love this product!"],
        ["This is terrible."],
        ["It's okay, not great."]
    ]
)

iface.launch()
```

**Option B: Streamlit**

```python
# app.py
import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Sentiment Analysis Model")
st.write("Enter text to classify sentiment")

# Input
text = st.text_area("Text input", height=100)

if st.button("Predict"):
    if text:
        # Preprocess and predict
        features = preprocess_text(text)
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Display results
        st.success(f"Prediction: {prediction}")
        st.bar_chart({
            "Negative": probability[0],
            "Positive": probability[1]
        })
    else:
        st.warning("Please enter some text")
```

**Step 4: Add Requirements**

```txt
# requirements.txt
gradio>=4.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
```

**Step 5: Deploy**

```bash
# Clone your space
git clone https://huggingface.co/spaces/your-username/my-sentiment-model
cd my-sentiment-model

# Add files
# - app.py
# - requirements.txt
# - model.pkl (or load from Hugging Face Hub)

# Commit and push
git add .
git commit -m "Add sentiment analysis model"
git push

# Auto-deploys in 2-3 minutes!
# Your app will be live at:
# https://your-username-my-sentiment-model.hf.space
```

**Advanced: Loading Models from Hugging Face Hub**

```python
# app.py
from transformers import pipeline
import gradio as gr

# Load model from Hugging Face Hub
classifier = pipeline("sentiment-analysis", 
                     model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def predict(text):
    result = classifier(text)[0]
    return {
        result['label']: result['score']
    }

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="label",
    title="Sentiment Analysis"
)

iface.launch()
```

**Advanced: Multi-Model Space**

```python
# app.py
import gradio as gr
import joblib

# Load multiple models
sentiment_model = joblib.load("sentiment_model.pkl")
topic_model = joblib.load("topic_model.pkl")

def sentiment_analysis(text):
    pred = sentiment_model.predict([text])[0]
    return pred

def topic_classification(text):
    topic = topic_model.predict([text])[0]
    return topic

# Create tabbed interface
with gr.Blocks() as demo:
    gr.Markdown("# NLP Models Demo")
    
    with gr.Tabs():
        with gr.TabItem("Sentiment Analysis"):
            text1 = gr.Textbox(label="Text")
            output1 = gr.Label(label="Sentiment")
            btn1 = gr.Button("Analyze")
            btn1.click(sentiment_analysis, text1, output1)
        
        with gr.TabItem("Topic Classification"):
            text2 = gr.Textbox(label="Text")
            output2 = gr.Label(label="Topic")
            btn2 = gr.Button("Classify")
            btn2.click(topic_classification, text2, output2)

demo.launch()
```

**Best Practices:**

1. **Keep models small**: Use quantized models (< 1GB recommended)
2. **Add examples**: Help users understand how to use your model
3. **Error handling**: Handle edge cases gracefully
4. **Documentation**: Add README.md explaining your model
5. **Version control**: Use Git to track changes

**Benefits for Portfolio:**

- **Live demos**: Show working models to employers
- **Easy sharing**: Share via URL
- **Professional**: Industry-recognized platform
- **Free**: No hosting costs
- **Community**: Get feedback and stars

**Limitations:**

- **Free tier**: Limited resources (CPU, memory)
- **Public by default**: Models are publicly accessible
- **File size**: Limited storage for large models
- **No custom domains**: Must use hf.space domain

**When to Use Hugging Face Spaces:**

- Portfolio projects
- Model demos and prototypes
- Sharing with community
- Learning deployment
- Quick proof-of-concepts

**When NOT to Use:**

- Production applications (use AWS/GCP/Azure)
- Large models (> 5GB)
- High-traffic applications
- Private/internal tools
- Custom domain requirements

---

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

### AWS ECS Fargate for ML Model Serving

**What is ECS Fargate?**
AWS Elastic Container Service (ECS) Fargate is a serverless container platform that eliminates the need to manage servers. Perfect for ML model serving with automatic scaling.

**Why ECS Fargate for ML?**
- **No Server Management**: Focus on your models, not infrastructure
- **Automatic Scaling**: Handles traffic spikes automatically
- **Cost Effective**: Pay only for running containers
- **Security**: Built-in IAM integration and VPC support
- **Container Insights**: Built-in monitoring and logging

**Setting Up ECS Fargate for ML API:**

```python
# Dockerfile (same as before)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**1. Create ECR Repository (Container Registry):**

```bash
# Create repository
aws ecr create-repository --repository-name ml-api

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t ml-api .
docker tag ml-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
```

**2. Create Task Definition:**

```json
{
  "family": "ml-api-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "ml-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "s3://my-ml-bucket/models/model.pkl"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

**3. Create ECS Service:**

```bash
# Create service
aws ecs create-service \
  --cluster ml-api-cluster \
  --service-name ml-api-service \
  --task-definition ml-api-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:xxx:targetgroup/ml-api-tg/xxx,containerName=ml-api,containerPort=8000"
```

**4. Enable Container Insights:**

```bash
# Enable Container Insights for monitoring
aws ecs update-cluster \
  --cluster ml-api-cluster \
  --settings name=containerInsights,value=enabled
```

**Benefits for ML:**
- **Auto-scaling**: Automatically scale based on CPU/memory usage
- **Blue/Green Deployments**: Zero-downtime model updates
- **Health Checks**: Automatic container replacement on failures
- **Integration**: Works seamlessly with ALB, CloudWatch, X-Ray

---

### AWS Infrastructure as Code for ML

**Why Infrastructure as Code (IaC)?**
- **Reproducibility**: Deploy identical infrastructure across environments
- **Version Control**: Track infrastructure changes in Git
- **Collaboration**: Team members can review and approve changes
- **Disaster Recovery**: Quickly recreate infrastructure
- **Cost Management**: Track and optimize infrastructure costs

**AWS CloudFormation for ML Infrastructure:**

```yaml
# ml-infrastructure.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ML Model Serving Infrastructure'

Parameters:
  ModelBucket:
    Type: String
    Description: S3 bucket for model artifacts
  
Resources:
  # VPC for ML services
  MLVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: ML-VPC

  # ECS Cluster
  MLCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: ml-api-cluster
      ClusterSettings:
        - Name: containerInsights
          Value: enabled

  # Application Load Balancer
  MLALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: ml-api-alb
      Type: application
      Scheme: internet-facing
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # ECS Service
  MLService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: ml-api-service
      Cluster: !Ref MLCluster
      TaskDefinition: !Ref MLTaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: ENABLED
          Subnets:
            - !Ref PublicSubnet1
            - !Ref PublicSubnet2
          SecurityGroups:
            - !Ref ECSSecurityGroup
      LoadBalancers:
        - ContainerName: ml-api
          ContainerPort: 8000
          TargetGroupArn: !Ref MLTargetGroup

  # S3 Bucket for Models
  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Ref ModelBucket
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt MLALB.DNSName
    Export:
      Name: ML-API-DNS
```

**Deploy with CloudFormation:**

```bash
# Create stack
aws cloudformation create-stack \
  --stack-name ml-infrastructure \
  --template-body file://ml-infrastructure.yaml \
  --parameters ParameterKey=ModelBucket,ParameterValue=my-ml-models-bucket

# Update stack
aws cloudformation update-stack \
  --stack-name ml-infrastructure \
  --template-body file://ml-infrastructure.yaml

# Delete stack
aws cloudformation delete-stack --stack-name ml-infrastructure
```

**AWS SAM (Serverless Application Model) for ML APIs:**

SAM simplifies serverless ML deployments.

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Parameters:
  ModelBucket:
    Type: String
    Default: my-ml-models

Resources:
  # Lambda function for ML inference
  MLInferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: ml-inference-api
      Runtime: python3.9
      Handler: lambda_handler.handler
      CodeUri: inference/
      Timeout: 30
      MemorySize: 3008  # For larger models
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
          MODEL_KEY: models/model.pkl
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /predict
            Method: post
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket
        - CloudWatchLogsFullAccess

  # API Gateway
  MLAPI:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Cors:
        AllowMethods: "'POST,OPTIONS'"
        AllowHeaders: "'Content-Type,X-Amz-Date,Authorization'"
        AllowOrigin: "'*'"

Outputs:
  MLAPIEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub https://${MLAPI}.execute-api.${AWS::Region}.amazonaws.com/prod/predict
```

**Deploy with SAM:**

```bash
# Build
sam build

# Deploy
sam deploy --guided

# Test
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

---

### AWS Observability for ML Systems

**Observability vs Monitoring:**

**Monitoring**: Collecting metrics and logs (what happened)
**Observability**: Understanding system behavior through metrics, logs, and traces (why it happened)

**Three Pillars of Observability:**
1. **Metrics**: Numerical measurements over time (latency, error rate)
2. **Logs**: Discrete events with timestamps
3. **Traces**: Request flow through distributed systems

**AWS X-Ray for Distributed Tracing:**

X-Ray helps trace requests through your ML pipeline.

```python
# Install
# pip install aws-xray-sdk

from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
import boto3

# Patch all AWS SDK clients
patch_all()

@xray_recorder.capture('predict')
def predict(features):
    """ML prediction with X-Ray tracing"""
    # This segment will appear in X-Ray
    subsegment = xray_recorder.begin_subsegment('model_loading')
    model = load_model()
    xray_recorder.end_subsegment()
    
    subsegment = xray_recorder.begin_subsegment('prediction')
    result = model.predict(features)
    xray_recorder.end_subsegment()
    
    return result

# Add metadata
xray_recorder.put_metadata('model_version', 'v1.2.3')
xray_recorder.put_metadata('features_count', len(features))
```

**X-Ray Configuration for ECS:**

```json
{
  "containerDefinitions": [
    {
      "name": "ml-api",
      "image": "ml-api:latest",
      "environment": [
        {
          "name": "_X_AMZN_TRACE_ID",
          "value": ""
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-api",
          "awslogs-region": "us-east-1"
        }
      }
    },
    {
      "name": "xray-daemon",
      "image": "amazon/aws-xray-daemon:latest",
      "cpu": 32,
      "memoryReservation": 256,
      "command": ["-o"]
    }
  ]
}
```

**CloudWatch for ML Monitoring:**

```python
import boto3
import time
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def log_prediction_metric(prediction_time, model_version):
    """Log custom metrics to CloudWatch"""
    cloudwatch.put_metric_data(
        Namespace='ML/API',
        MetricData=[
            {
                'MetricName': 'PredictionLatency',
                'Value': prediction_time,
                'Unit': 'Milliseconds',
                'Timestamp': datetime.utcnow(),
                'Dimensions': [
                    {
                        'Name': 'ModelVersion',
                        'Value': model_version
                    }
                ]
            },
            {
                'MetricName': 'PredictionCount',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            }
        ]
    )

# Create CloudWatch alarms
def create_latency_alarm():
    cloudwatch.put_metric_alarm(
        AlarmName='ML-API-High-Latency',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=2,
        MetricName='PredictionLatency',
        Namespace='ML/API',
        Period=60,
        Statistic='Average',
        Threshold=1000.0,  # 1 second
        ActionsEnabled=True,
        AlarmActions=['arn:aws:sns:us-east-1:xxx:ml-alerts']
    )
```

**CloudWatch Logs Insights for ML:**

```python
# Query logs
# Example: Find slow predictions
query = """
fields @timestamp, @message, @logStream
| filter @message like /prediction/
| parse @message "Prediction took * ms" as latency
| stats avg(latency) as avg_latency, max(latency) as max_latency by bin(5m)
| sort @timestamp desc
"""

# Use CloudWatch Logs Insights API
logs = boto3.client('logs')
response = logs.start_query(
    logGroupName='/ecs/ml-api',
    startTime=int((datetime.now() - timedelta(hours=1)).timestamp()),
    endTime=int(datetime.now().timestamp()),
    queryString=query
)
```

**Structured Logging for ML:**

```python
import json
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logger
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log with context
def log_prediction(request_id, features, prediction, latency):
    logger.info("Prediction completed", extra={
        "request_id": request_id,
        "features_count": len(features),
        "prediction": prediction,
        "latency_ms": latency,
        "model_version": "v1.2.3"
    })
```

---

### AWS Security Best Practices for ML

**1. IAM Roles and Policies:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::my-ml-models/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

**2. VPC Configuration:**

```yaml
# Private subnets for ECS tasks
PrivateSubnet:
  Type: AWS::EC2::Subnet
  Properties:
    VpcId: !Ref MLVPC
    CidrBlock: 10.0.1.0/24
    AvailabilityZone: us-east-1a

# Security Group - Only allow ALB traffic
ECSSecurityGroup:
  Type: AWS::EC2::SecurityGroup
  Properties:
    GroupDescription: Security group for ECS tasks
    VpcId: !Ref MLVPC
    SecurityGroupIngress:
      - IpProtocol: tcp
        FromPort: 8000
        ToPort: 8000
        SourceSecurityGroupId: !Ref ALBSecurityGroup
```

**3. Secrets Management:**

```python
import boto3
import json
import base64

# Use AWS Secrets Manager
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')

def get_secret(secret_name):
    """Retrieve secret from AWS Secrets Manager"""
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        raise

# Store API keys, database credentials, etc.
# secrets_client.create_secret(
#     Name='ml-api-keys',
#     SecretString=json.dumps({
#         'api_key': 'your-api-key',
#         'db_password': 'your-password'
#     })
# )
```

**4. Encryption at Rest and in Transit:**

```yaml
# S3 bucket with encryption
ModelBucket:
  Type: AWS::S3::Bucket
  Properties:
    BucketEncryption:
      ServerSideEncryptionConfiguration:
        - ServerSideEncryptionByDefault:
            SSEAlgorithm: AES256
    PublicAccessBlockConfiguration:
      BlockPublicAcls: true
      BlockPublicPolicy: true
```

**5. API Authentication with AWS Cognito:**

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import boto3
import jwt

cognito = boto3.client('cognito-idp')
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token from Cognito"""
    try:
        token = credentials.credentials
        # Verify token with Cognito
        response = cognito.get_user(AccessToken=token)
        return response
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(data: PredictionRequest, user=Depends(verify_token)):
    # Authenticated prediction
    return {"prediction": model.predict(data.features)}
```

---

### AWS Cost Management for ML Workloads

**1. Budgets and Billing Alarms:**

```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name ML-Monthly-Billing-Alarm \
  --alarm-description "Alert when monthly costs exceed $500" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 500 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=Currency,Value=USD

# Create budget
aws budgets create-budget \
  --account-id <account-id> \
  --budget file://budget.json
```

**budget.json:**
```json
{
  "BudgetName": "ML-Monthly-Budget",
  "BudgetLimit": {
    "Amount": "500",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostFilters": {
    "Service": ["Amazon ECS", "Amazon S3", "AWS Lambda"]
  }
}
```

**2. Cost Optimization Strategies:**

- **Use Spot Instances** for training (with checkpointing)
- **Right-size ECS tasks** (match CPU/memory to actual usage)
- **S3 Lifecycle Policies** (move old models to Glacier)
- **Reserved Capacity** for predictable workloads
- **Auto-scaling** to scale down during low traffic

**3. Cost Monitoring:**

```python
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')  # Cost Explorer

def get_ml_costs(start_date, end_date):
    """Get ML-related costs"""
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'Or': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon ECS']}},
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon S3']}},
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['AWS Lambda']}}
            ]
        },
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'},
            {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
        ]
    )
    return response
```

---

## Orchestration and Scaling with Kubernetes

### Kubernetes (k8s) Basics

**What is Kubernetes?**
Container orchestration platform for managing containerized applications.

**Key Concepts:**

**1. Pods:**
Smallest deployable unit - contains one or more containers.

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-api-pod
spec:
  containers:
  - name: ml-api
    image: ml-api:latest
    ports:
    - containerPort: 8000
```

**2. Deployments:**
Manages Pods and provides updates, rollbacks, scaling.

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-deployment
spec:
  replicas: 3  # Number of pods
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

**3. Services:**
Exposes Pods to network traffic (load balancing).

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer  # Or ClusterIP, NodePort
```

**Deploying to Kubernetes:**
```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods
kubectl get deployments
kubectl get services

# Scale deployment
kubectl scale deployment ml-api-deployment --replicas=5

# View logs
kubectl logs -l app=ml-api

# Delete
kubectl delete -f deployment.yaml -f service.yaml
```

### Cloud Services for ML Deployment

**AWS SageMaker:**
Managed ML platform.

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# Create estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
    instance_type='ml.m5.large',
    framework_version='0.24-1',
    py_version='py3'
)

# Train
sklearn_estimator.fit({'training': 's3://bucket/data'})

# Deploy
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Predict
result = predictor.predict(data)
```

**Azure ML:**
Microsoft's ML platform.

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

# Load workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path='model.pkl',
    model_name='MyModel'
)

# Deploy
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name='ml-api',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
```

**GCP Vertex AI:**
Google's ML platform.

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project='my-project', location='us-central1')

# Deploy model
endpoint = aiplatform.Endpoint.create(display_name='ml-api')

# Upload model
model = aiplatform.Model.upload(
    display_name='MyModel',
    artifact_uri='gs://bucket/model',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest'
)

# Deploy to endpoint
endpoint.deploy(
    model=model,
    deployed_model_display_name='ml-api',
    machine_type='n1-standard-2',
    min_replica_count=1,
    max_replica_count=3
)

# Predict
predictions = endpoint.predict(instances=data)
```

---

## Continuous Deployment (CD) Pipeline

### Automating Deployment

**GitHub Actions CD Pipeline:**
```yaml
# .github/workflows/deploy.yml
name: Deploy ML Model

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/
      
      - name: Build Docker image
        run: |
          docker build -t ml-api:${{ github.sha }} .
          docker tag ml-api:${{ github.sha }} ml-api:latest
      
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push ml-api:${{ github.sha }}
          docker push ml-api:latest
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-api-deployment ml-api=ml-api:${{ github.sha }}
          kubectl rollout status deployment/ml-api-deployment
      
      - name: Health check
        run: |
          sleep 10
          curl -f http://ml-api-service/health || exit 1
```

### Health Check and Rollback Strategy

**Health Check Implementation:**
```python
# health_check.py
import requests
import time
import sys

def health_check(url, max_retries=3, timeout=5):
    """Check if service is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=timeout)
            if response.status_code == 200:
                return True
        except Exception as e:
            print(f"Health check failed (attempt {i+1}): {e}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    if health_check(url):
        print("Service is healthy")
        sys.exit(0)
    else:
        print("Service is unhealthy")
        sys.exit(1)
```

**Rollback Strategy:**
```yaml
# Rollback on failure
- name: Deploy
  run: kubectl set image deployment/ml-api-deployment ml-api=ml-api:new
  continue-on-error: true

- name: Health check
  run: python health_check.py http://ml-api-service
  continue-on-error: true

- name: Rollback on failure
  if: failure()
  run: |
    kubectl rollout undo deployment/ml-api-deployment
    kubectl rollout status deployment/ml-api-deployment
```

**Triggering Deployment:**
- On successful CI
- On Model Registry approval
- On manual trigger
- On schedule

---

## Production Server Setup

### NGINX Reverse Proxy Configuration

NGINX acts as a reverse proxy, load balancer, and SSL terminator for ML APIs.

**Basic NGINX Configuration:**

```nginx
# /etc/nginx/sites-available/ml-api
upstream ml_backend {
    # Load balancing with least connections
    least_conn;
    server 127.0.0.1:8000 weight=3;
    server 127.0.0.1:8001 weight=2;
    server 127.0.0.1:8002 weight=1 backup;
    
    # Keep connections alive
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=ml_api_limit:10m rate=10r/s;
    limit_req zone=ml_api_limit burst=20 nodelay;
    
    # Request size limit (for large feature vectors)
    client_max_body_size 10M;
    
    # Timeouts (important for ML inference)
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    send_timeout 300s;
    
    # Prediction endpoint
    location /predict {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if using streaming predictions)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint (bypass rate limiting)
    location /health {
        access_log off;
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
    }
    
    # Metrics endpoint (internal only)
    location /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://ml_backend;
    }
    
    # Static files (if serving model documentation)
    location /static/ {
        alias /var/www/ml-api/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

**Enable and Test Configuration:**

```bash
# Test NGINX configuration
sudo nginx -t

# Enable site
sudo ln -s /etc/nginx/sites-available/ml-api /etc/nginx/sites-enabled/

# Reload NGINX
sudo systemctl reload nginx

# Check status
sudo systemctl status nginx
```

### SSL Certificate Setup with Let's Encrypt

**Install Certbot:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# CentOS/RHEL
sudo yum install certbot python3-certbot-nginx
```

**Obtain SSL Certificate:**

```bash
# Automatic configuration with NGINX
sudo certbot --nginx -d api.yourdomain.com

# Manual certificate only
sudo certbot certonly --nginx -d api.yourdomain.com

# Auto-renewal setup (already configured by certbot)
sudo certbot renew --dry-run
```

**Certificate Auto-Renewal:**

```bash
# Add to crontab (certbot usually does this automatically)
sudo crontab -e
# Add: 0 0,12 * * * certbot renew --quiet
```

### Domain Configuration

**DNS Setup:**

1. **A Record**: Point domain to server IP
   ```
   api.yourdomain.com  A  123.45.67.89
   ```

2. **CNAME Record** (if using subdomain):
   ```
   api  CNAME  yourdomain.com
   ```

**Verify DNS:**

```bash
# Check DNS propagation
dig api.yourdomain.com
nslookup api.yourdomain.com

# Test from different locations
curl -I https://api.yourdomain.com/health
```

### Enhanced Security for ML APIs

**1. Rate Limiting in FastAPI:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def predict(request: Request, data: PredictionRequest):
    # Prediction logic
    pass
```

**2. API Key Authentication:**

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import os

API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(
    data: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    # Prediction logic
    pass
```

**3. Input Validation and Sanitization:**

```python
from pydantic import BaseModel, validator
import numpy as np

class PredictionRequest(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features(cls, v):
        # Check length
        if len(v) < 1 or len(v) > 1000:
            raise ValueError('Features must be between 1 and 1000')
        
        # Check for NaN or Inf
        features_array = np.array(v)
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            raise ValueError('Features cannot contain NaN or Inf')
        
        # Check value ranges
        if np.any(np.abs(features_array) > 1e10):
            raise ValueError('Feature values out of acceptable range')
        
        return v
```

**4. Request Logging and Monitoring:**

```python
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        logger.info(f"Client: {request.client.host}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} | "
            f"Time: {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(LoggingMiddleware)
```

### Production Error Handling

**Structured Error Responses:**

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class MLAPIError(Exception):
    def __init__(self, message: str, error_code: str, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code

@app.exception_handler(MLAPIError)
async def ml_api_error_handler(request, exc: MLAPIError):
    logger.error(f"ML API Error: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "request_id": request.headers.get("X-Request-ID", "unknown")
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "request_id": request.headers.get("X-Request-ID", "unknown")
            }
        }
    )
```

### Production Logging Setup

**Structured Logging Configuration:**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('ml-api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(JSONFormatter())
```

### AWS EC2 Setup for ML Deployment

**1. Launch EC2 Instance:**

```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install python3.9 python3-pip python3-venv -y
sudo apt-get install nginx -y
sudo apt-get install certbot python3-certbot-nginx -y
```

**2. Setup Application:**

```bash
# Create application directory
mkdir -p /var/www/ml-api
cd /var/www/ml-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn[standard] scikit-learn

# Copy your application files
# (use git, scp, or deployment pipeline)
```

**3. Setup Systemd Service:**

```ini
# /etc/systemd/system/ml-api.service
[Unit]
Description=ML API Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/ml-api
Environment="PATH=/var/www/ml-api/venv/bin"
ExecStart=/var/www/ml-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and Start Service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-api
sudo systemctl start ml-api
sudo systemctl status ml-api
```

**4. Configure Firewall:**

```bash
# Allow HTTP, HTTPS, and SSH
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Production Checklist

- [ ] NGINX configured as reverse proxy
- [ ] SSL certificate installed and auto-renewal configured
- [ ] Domain DNS properly configured
- [ ] Rate limiting implemented
- [ ] API authentication/authorization set up
- [ ] Input validation and sanitization
- [ ] Structured logging configured
- [ ] Error handling with proper error codes
- [ ] Health check endpoints implemented
- [ ] Monitoring and alerting set up
- [ ] Firewall rules configured
- [ ] Systemd service for auto-restart
- [ ] Backup strategy for models and data
- [ ] Documentation for API endpoints

---

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

## Additional Resources

### Production Server Setup

**NGINX:**
- [NGINX Official Documentation](https://nginx.org/en/docs/)
- [NGINX Beginner's Guide](https://nginx.org/en/docs/beginners_guide.html)
- [NGINX Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [NGINX Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)

**SSL/TLS & Security:**
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Certbot User Guide](https://eff-certbot.readthedocs.io/)
- [SSL/TLS Best Practices (Mozilla)](https://wiki.mozilla.org/Security/Server_Side_TLS)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)

**Domain & DNS:**
- [DNS Basics (Cloudflare)](https://www.cloudflare.com/learning/dns/what-is-dns/)
- [DNS Configuration Guide](https://www.cloudflare.com/learning/dns/dns-records/)

**AWS EC2:**
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [EC2 User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/)
- [Setting up NGINX on EC2 (AWS)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nginx.html)

**Tutorials:**
- [Setting up NGINX as Reverse Proxy (DigitalOcean)](https://www.digitalocean.com/community/tutorials/how-to-set-up-a-node-js-application-for-production-on-ubuntu-20-04)
- [SSL Certificate Setup with Let's Encrypt (DigitalOcean)](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-20-04)
- [Production FastAPI Deployment (TestDriven.io)](https://testdriven.io/blog/fastapi-deployment/)

---

## Next Steps

- Practice deploying models to different platforms
- Experiment with containerization
- Set up production servers with NGINX and SSL
- Configure domain and DNS for your ML APIs
- Set up monitoring and logging
- Learn about model versioning
- Explore edge deployment
- Move to [14-mlops-basics](../14-mlops-basics/README.md)

**Remember**: Deployment is as important as training! A great model is useless if it can't serve predictions.

