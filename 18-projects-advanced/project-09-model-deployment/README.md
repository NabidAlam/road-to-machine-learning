# Project 9: Model Deployment & Serving Project

Deploy a machine learning model as a production-ready API service with Docker, monitoring, and best practices.

## Difficulty
Advanced

## Time Estimate
1-2 weeks

## Skills You'll Practice
- Model Deployment
- FastAPI
- Docker
- REST APIs
- Model Versioning
- Cloud Deployment
- Model Monitoring
- CI/CD for ML

## Learning Objectives

By completing this project, you will learn to:
- Build production-ready REST APIs with FastAPI
- Containerize ML models with Docker
- Deploy models to cloud platforms
- Implement model versioning
- Set up monitoring and logging
- Handle errors and edge cases
- Optimize for production
- Implement A/B testing (optional)

## Prerequisites

Before starting, you should have completed:
- Phase 2-6: Core ML concepts
- Phase 13: Model Deployment (all topics)
- Understanding of REST APIs
- Basic knowledge of Docker
- Familiarity with cloud platforms (optional)

## Dataset & Model

**Choose any trained model:**
- Classification model (e.g., from previous projects)
- Regression model
- Pre-trained model from your portfolio

**Recommended:**
- Use a model from a previous project
- Or train a simple model for this project
- Focus on deployment, not model complexity

## Project Steps

### Step 1: Model Preparation
- Load or train your model
- Serialize model (pickle, joblib, or format-specific)
- Create model metadata
- Test model locally
- Prepare preprocessing pipeline

### Step 2: Build FastAPI Application
- Install FastAPI: `pip install fastapi uvicorn`
- Create basic API structure
- Implement prediction endpoint
- Add health check endpoint
- Add model info endpoint
- Implement request/response models with Pydantic
- Add error handling

### Step 3: Advanced FastAPI Features
- Add dependency injection for model loading
- Implement background tasks (logging)
- Add request validation
- Create batch prediction endpoint
- Add API documentation (automatic with FastAPI)
- Implement CORS middleware

### Step 4: Docker Containerization
- Create Dockerfile
- Optimize Docker image size
- Add health checks
- Test container locally
- Build and run Docker container

### Step 5: Model Versioning
- Implement model versioning system
- Store multiple model versions
- Add version selection in API
- Create model registry (simple file-based or database)

### Step 6: Monitoring and Logging
- Add logging for predictions
- Implement prediction tracking
- Add performance metrics
- Create monitoring dashboard (optional)
- Set up alerts (optional)

### Step 7: Testing
- Write unit tests for API
- Test prediction endpoints
- Test error handling
- Load testing (optional)

### Step 8: Cloud Deployment
- Choose cloud platform (AWS/GCP/Azure/Heroku)
- Deploy application
- Configure environment variables
- Set up domain/URL
- Test deployed service

### Step 9: CI/CD Setup (Optional)
- Create GitHub Actions workflow
- Automate testing
- Automate deployment
- Set up staging environment

### Step 10: Documentation
- API documentation (auto-generated)
- Deployment guide
- Usage examples
- Troubleshooting guide

## Code Structure

```
project-09-model-deployment/
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── dependencies.py      # Dependency injection
│   └── utils.py            # Utility functions
├── models/
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   └── metadata.json
├── tests/
│   ├── test_api.py
│   └── test_predictions.py
├── docker/
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD (optional)
├── requirements.txt
├── docker-compose.yml       # For local development
└── .env.example
```

## Implementation Examples

### 1. FastAPI Application Structure
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = joblib.load('models/model_v1.pkl')

# Create app
app = FastAPI(
    title="ML Model API",
    description="Production ML model serving API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    confidence: float

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        confidence = float(max(probabilities))
        
        # Log prediction
        background_tasks.add_task(
            log_prediction,
            request.features,
            int(prediction),
            confidence
        )
        
        return PredictionResponse(
            prediction=int(prediction),
            probabilities=probabilities,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def log_prediction(features, prediction, confidence):
    logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")

# Run with: uvicorn app.main:app --reload
```

### 2. Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Docker Compose (for local development)
```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model_v1.pkl
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

### 4. Testing
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0, 4.0]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
```

## Evaluation Criteria

Your deployment should:
- Have working REST API
- Be containerized with Docker
- Handle errors gracefully
- Include proper logging
- Have API documentation
- Include health checks
- Be tested
- Be deployable to cloud

## Key Features to Implement

1. **API Endpoints**
   - `/health` - Health check
   - `/predict` - Single prediction
   - `/predict/batch` - Batch predictions
   - `/model/info` - Model information
   - `/docs` - Auto-generated API docs

2. **Error Handling**
   - Invalid input validation
   - Model loading errors
   - Prediction errors
   - Proper HTTP status codes

3. **Performance**
   - Fast response times
   - Efficient model loading
   - Caching (optional)
   - Async operations (optional)

4. **Security**
   - Input validation
   - Rate limiting (optional)
   - Authentication (optional)
   - HTTPS (in production)

## Extensions

1. **A/B Testing**
   - Serve multiple model versions
   - Route traffic between versions
   - Compare performance

2. **Model Monitoring**
   - Track prediction distributions
   - Monitor model drift
   - Alert on anomalies

3. **Feature Store Integration**
   - Connect to feature store
   - Real-time feature computation

4. **Kubernetes Deployment**
   - Deploy to Kubernetes
   - Auto-scaling
   - Load balancing

5. **GraphQL API**
   - Implement GraphQL endpoint
   - Flexible querying

## Deployment Options

### Option 1: Heroku (Easiest)
```bash
# Install Heroku CLI
# Create Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT

# Deploy
git push heroku main
```

### Option 2: AWS (EC2/ECS/Lambda)
- EC2: Deploy Docker container
- ECS: Container orchestration
- Lambda: Serverless (for smaller models)

### Option 3: Google Cloud (Cloud Run)
```bash
gcloud run deploy ml-api --source .
```

### Option 4: Azure (Container Instances/App Service)
- Deploy Docker container
- Use Azure Container Registry

## Resources

- [Model Deployment Guide](../../13-model-deployment/deployment.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Tutorial](../../resources/docker_tutorial.md)

## Tips for Success

1. **Start Local**: Test everything locally first
2. **Use Docker**: Containerize early
3. **Test Thoroughly**: Write tests before deployment
4. **Monitor**: Add logging from the start
5. **Document**: Keep deployment notes
6. **Iterate**: Deploy simple version first, then enhance
7. **Security**: Consider security from the beginning

## Common Pitfalls to Avoid

- Not handling errors properly
- Missing input validation
- Not testing before deployment
- Ignoring logging
- Large Docker images
- Not versioning models
- Hardcoding configuration

## Next Steps

After completing this project:
- Deploy to production
- Add monitoring
- Implement A/B testing
- Scale your deployment
- Learn Kubernetes

---

**Ready to deploy?** Start by building your FastAPI application and containerizing it with Docker!

