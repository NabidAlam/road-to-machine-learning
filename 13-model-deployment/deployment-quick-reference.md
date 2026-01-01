# Model Deployment Quick Reference Guide

Quick reference for deploying ML models.

## Table of Contents

- [Model Serialization](#model-serialization)
- [API Frameworks](#api-frameworks)
- [Docker Commands](#docker-commands)
- [Cloud Platforms](#cloud-platforms)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Model Serialization

### Save Models

```python
# Scikit-learn
joblib.dump(model, 'model.joblib')

# Keras
model.save('model.h5')

# PyTorch
torch.save(model.state_dict(), 'model.pth')
```

---

## API Frameworks

### FastAPI

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(features: list[float]):
    return {"prediction": model.predict([features])[0]}
```

### Flask

```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'prediction': model.predict([features])[0]})
```

---

## Docker Commands

```bash
# Build
docker build -t ml-api .

# Run
docker run -p 8000:8000 ml-api

# Compose
docker-compose up
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
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Best Practices

- **Clear hypothesis**: Define what you're testing
- **Primary metric**: One clear success metric
- **Sample size**: Ensure statistical power
- **Don't peek early**: Avoid stopping prematurely
- **Monitor**: Track deployed model performance

## Cloud Platforms

| Platform | Use Case | Command |
|----------|----------|---------|
| **Heroku** | Simple | `git push heroku main` |
| **AWS Lambda** | Serverless | Deploy via console/CLI |
| **GCP Cloud Run** | Containers | `gcloud run deploy` |
| **Azure** | Enterprise | `az container create` |
| **AWS SageMaker** | ML Platform | `sagemaker.deploy()` |

---

## Common Issues & Solutions

### Issue 1: Model Too Large

**Solution**: Quantize, prune, use smaller models

### Issue 2: Slow Response

**Solution**: Optimize model, use caching, batch processing

---

## Best Practices Checklist

- [ ] Validate all inputs
- [ ] Handle errors gracefully
- [ ] Add health checks
- [ ] Use HTTPS in production
- [ ] Monitor performance
- [ ] Version your models
- [ ] Document API

---

**Remember**: Deployment is as important as training!

