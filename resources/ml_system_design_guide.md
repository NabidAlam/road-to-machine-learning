# ML System Design Comprehensive Guide

Complete guide to designing scalable, production-ready machine learning systems.

## Table of Contents

- [Introduction to ML System Design](#introduction-to-ml-system-design)
- [Core Concepts](#core-concepts)
- [Scalability Patterns](#scalability-patterns)
- [Data Infrastructure](#data-infrastructure)
- [Model Serving Architecture](#model-serving-architecture)
- [High Availability & Reliability](#high-availability-reliability)
- [Monitoring & Observability](#monitoring-observability)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to ML System Design

### What is ML System Design?

**ML System Design** is the practice of designing machine learning systems that are:
- **Scalable**: Handle increasing load efficiently
- **Reliable**: High availability and fault tolerance
- **Performant**: Low latency, high throughput
- **Maintainable**: Easy to update and monitor
- **Cost-Effective**: Optimize resource usage

### Why System Design Matters for ML

**Unique ML Challenges:**
- Model inference can be compute-intensive
- Data pipelines need to handle large volumes
- Models need to be updated without downtime
- Real-time vs batch processing trade-offs
- Model versioning and A/B testing

**Key Differences from Traditional Systems:**
- **Stateful Models**: Models need to be loaded in memory
- **Data Dependencies**: Models depend on training data
- **Non-Deterministic**: Model performance can degrade
- **Resource Intensive**: GPU/CPU requirements vary

---

## Core Concepts

### Requests & Responses

**Request Flow in ML Systems:**
```
Client Request
    ↓
API Gateway / Load Balancer
    ↓
Preprocessing Service
    ↓
Model Serving Layer
    ↓
Post-processing
    ↓
Response to Client
```

**Request Types:**
- **Synchronous**: Real-time predictions (low latency required)
- **Asynchronous**: Batch predictions (higher latency acceptable)
- **Streaming**: Continuous data processing

**Response Design:**
```python
# Good response structure
{
    "prediction": 0.85,
    "confidence": 0.92,
    "model_version": "v2.1",
    "processing_time_ms": 45,
    "timestamp": "2024-01-15T10:30:00Z"
}

# Include metadata for debugging
{
    "prediction": 0.85,
    "probabilities": [0.15, 0.85],
    "model_info": {
        "version": "v2.1",
        "training_date": "2024-01-01"
    },
    "request_id": "req_12345"
}
```

### Latency

**Latency Components:**
1. **Network Latency**: Request/response transmission
2. **Preprocessing Time**: Data transformation
3. **Model Inference Time**: Prediction computation
4. **Post-processing Time**: Result formatting

**Latency Targets:**
- **Real-time ML**: < 100ms (recommendation systems)
- **Interactive ML**: < 500ms (chatbots, search)
- **Batch ML**: < 5 minutes (reports, analytics)

**Optimizing Latency:**
```python
# 1. Model Optimization
# - Use smaller models where possible
# - Quantization (INT8 instead of FP32)
# - Model pruning

# 2. Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(features_hash):
    return model.predict(features)

# 3. Batch Processing
# Process multiple requests together
def batch_predict(requests):
    features = [r['features'] for r in requests]
    predictions = model.predict_batch(features)
    return predictions

# 4. Async Processing
import asyncio

async def async_predict(features):
    # Non-blocking prediction
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, features)
    return result
```

### Throughput

**Throughput Metrics:**
- **Requests Per Second (RPS)**: Number of requests handled
- **Predictions Per Second (PPS)**: Actual predictions made
- **Queries Per Second (QPS)**: Database/vector DB queries

**Improving Throughput:**
```python
# 1. Horizontal Scaling
# Multiple model instances behind load balancer

# 2. Batch Processing
def batch_predict(features_list, batch_size=32):
    """Process in batches for better GPU utilization"""
    predictions = []
    for i in range(0, len(features_list), batch_size):
        batch = features_list[i:i+batch_size]
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds)
    return predictions

# 3. Model Parallelism
# Split model across multiple GPUs
import torch
from torch.nn.parallel import DataParallel

model = DataParallel(model, device_ids=[0, 1, 2, 3])

# 4. Pipeline Parallelism
# Different stages on different machines
```

**Throughput vs Latency Trade-off:**
- Higher batch size → Higher throughput, Higher latency
- Lower batch size → Lower throughput, Lower latency
- Need to balance based on use case

---

## Scalability Patterns

### Load Balancing

**Why Load Balancing for ML?**
- Distribute requests across multiple model instances
- Handle traffic spikes
- Improve availability (if one instance fails)

**Load Balancing Strategies:**

**1. Round Robin:**
```python
# Simple round-robin
servers = ['server1', 'server2', 'server3']
current = 0

def get_server():
    global current
    server = servers[current]
    current = (current + 1) % len(servers)
    return server
```

**2. Least Connections:**
```python
# Route to server with fewest active connections
def get_server_least_connections(servers):
    return min(servers, key=lambda s: s.active_connections)
```

**3. Weighted Round Robin:**
```python
# Weight servers by capacity
servers = [
    {'name': 'server1', 'weight': 3, 'capacity': 'high'},
    {'name': 'server2', 'weight': 2, 'capacity': 'medium'},
    {'name': 'server3', 'weight': 1, 'capacity': 'low'}
]
```

**4. Model-Aware Routing:**
```python
# Route to appropriate model based on request
def route_request(request):
    if request['complexity'] == 'simple':
        return simple_model_server
    elif request['complexity'] == 'complex':
        return complex_model_server
    else:
        return default_model_server
```

**Load Balancer Configuration (Nginx):**
```nginx
upstream ml_backend {
    least_conn;  # Use least connections
    server ml-api-1:8000 weight=3;
    server ml-api-2:8000 weight=2;
    server ml-api-3:8000 weight=1;
    
    # Health checks
    keepalive 32;
}

server {
    listen 80;
    location /predict {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

### Caching

**Caching Strategies for ML:**

**1. Prediction Caching:**
```python
import redis
import hashlib
import json

class PredictionCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def _cache_key(self, features):
        """Generate cache key from features"""
        features_str = json.dumps(features, sort_keys=True)
        return f"pred:{hashlib.md5(features_str.encode()).hexdigest()}"
    
    def get(self, features):
        """Get cached prediction"""
        key = self._cache_key(features)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, features, prediction):
        """Cache prediction"""
        key = self._cache_key(features)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(prediction)
        )

# Usage
cache = PredictionCache(redis_client)

def predict_with_cache(features):
    # Check cache first
    cached = cache.get(features)
    if cached:
        return cached
    
    # Compute prediction
    prediction = model.predict(features)
    
    # Cache result
    cache.set(features, prediction)
    
    return prediction
```

**2. Model Output Caching:**
```python
# Cache intermediate model outputs
@lru_cache(maxsize=10000)
def get_embeddings(text):
    return embedding_model.encode(text)
```

**3. Feature Caching:**
```python
# Cache preprocessed features
def get_features(raw_data):
    cache_key = f"features:{hash(raw_data)}"
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    features = preprocess(raw_data)
    redis.setex(cache_key, 3600, json.dumps(features))
    return features
```

**Cache Invalidation:**
```python
# Invalidate cache when model updates
def invalidate_model_cache(model_version):
    pattern = f"pred:*"
    keys = redis.keys(pattern)
    if keys:
        redis.delete(*keys)
```

### Vertical Scaling

**Vertical Scaling (Scale Up):**
- Increase resources on single machine
- More CPU, RAM, GPU
- Simpler but limited by hardware

**When to Use:**
- Single model instance sufficient
- Model too large to split
- Low traffic volume
- Cost-effective for small scale

**Example:**
```python
# Upgrade instance type
# Small: 2 CPU, 4GB RAM
# Medium: 4 CPU, 8GB RAM
# Large: 8 CPU, 16GB RAM
# XLarge: 16 CPU, 32GB RAM
```

### Horizontal Scaling

**Horizontal Scaling (Scale Out):**
- Add more instances
- Distribute load across machines
- Better for high traffic

**When to Use:**
- High traffic volume
- Need high availability
- Stateless services
- Cost-effective at scale

**Implementation:**
```python
# Kubernetes horizontal scaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3  # Start with 3 instances
  template:
    spec:
      containers:
      - name: ml-api
        image: ml-api:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Auto-scaling Triggers:**
- CPU utilization > 70%
- Memory usage > 80%
- Request queue length
- Custom metrics (prediction latency)

---

## Data Infrastructure

### Databases

**Database Types for ML:**

**1. Relational Databases (SQL):**
- Structured data
- ACID transactions
- Examples: PostgreSQL, MySQL

**2. NoSQL Databases:**
- Unstructured/semi-structured data
- High scalability
- Examples: MongoDB, Cassandra

**3. Vector Databases:**
- Store embeddings
- Similarity search
- Examples: Pinecone, Weaviate, FAISS

**4. Time-Series Databases:**
- Metrics and monitoring data
- Examples: InfluxDB, TimescaleDB

### Replication

**Database Replication:**
- **Master-Slave**: One write, multiple reads
- **Master-Master**: Multiple write nodes
- **Read Replicas**: Scale read operations

**Why Replication for ML:**
- Separate read/write workloads
- Scale feature retrieval
- Improve availability

**Example Configuration:**
```python
# Read from replica, write to master
class DatabaseRouter:
    def __init__(self):
        self.master = connect_master()
        self.replicas = [connect_replica(i) for i in range(3)]
        self.replica_index = 0
    
    def get_read_connection(self):
        """Round-robin read replicas"""
        conn = self.replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replicas)
        return conn
    
    def get_write_connection(self):
        """Always use master for writes"""
        return self.master

# Usage
router = DatabaseRouter()

# Read from replica
features = router.get_read_connection().query("SELECT * FROM features")

# Write to master
router.get_write_connection().execute("INSERT INTO predictions ...")
```

### Sharding

**Sharding Strategies:**

**1. Feature-Based Sharding:**
```python
# Shard by feature category
shards = {
    'user_features': shard_1,
    'product_features': shard_2,
    'interaction_features': shard_3
}

def get_features(user_id):
    user_features = shards['user_features'].get(user_id)
    product_features = shards['product_features'].get(user_id)
    return combine_features(user_features, product_features)
```

**2. Hash-Based Sharding:**
```python
def get_shard(user_id, num_shards=4):
    """Route to shard based on hash"""
    hash_value = hash(user_id)
    return hash_value % num_shards

# Route requests to appropriate shard
shard_id = get_shard(user_id)
features = shards[shard_id].get(user_id)
```

**3. Range-Based Sharding:**
```python
# Shard by user ID ranges
shards = {
    'shard_1': (0, 1000000),
    'shard_2': (1000001, 2000000),
    'shard_3': (2000001, 3000000)
}

def get_shard(user_id):
    for shard_name, (min_id, max_id) in shards.items():
        if min_id <= user_id <= max_id:
            return shard_name
```

### Message Queues

**Why Message Queues for ML:**
- Decouple components
- Handle traffic spikes
- Async processing
- Retry failed operations

**Message Queue Patterns:**

**1. Request-Response Pattern:**
```python
import redis
import json
import uuid

class MLMessageQueue:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.request_queue = "ml:requests"
        self.response_queue = "ml:responses"
    
    def send_request(self, features):
        """Send prediction request"""
        request_id = str(uuid.uuid4())
        message = {
            'request_id': request_id,
            'features': features,
            'timestamp': time.time()
        }
        self.redis.lpush(self.request_queue, json.dumps(message))
        return request_id
    
    def get_response(self, request_id, timeout=30):
        """Get prediction response"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.redis.get(f"ml:response:{request_id}")
            if response:
                return json.loads(response)
            time.sleep(0.1)
        return None
```

**2. Pub/Sub Pattern:**
```python
import redis

class MLPubSub:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.pubsub = self.redis.pubsub()
    
    def publish_prediction(self, channel, prediction):
        """Publish prediction result"""
        self.redis.publish(channel, json.dumps(prediction))
    
    def subscribe(self, channel, callback):
        """Subscribe to predictions"""
        self.pubsub.subscribe(channel)
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                prediction = json.loads(message['data'])
                callback(prediction)
```

**3. Task Queue (Celery):**
```python
from celery import Celery

app = Celery('ml_tasks', broker='redis://localhost:6379')

@app.task
def predict_task(features):
    """Async prediction task"""
    return model.predict(features)

# Send task
result = predict_task.delay(features)

# Get result
prediction = result.get(timeout=30)
```

---

## Model Serving Architecture

### Stateless Architecture

**Stateless Design:**
- No server-side session state
- Each request is independent
- Easy to scale horizontally

**Benefits:**
- Horizontal scaling
- Load balancing
- Fault tolerance
- Simple deployment

**Implementation:**
```python
# Stateless API service
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model at startup (shared across requests)
model = joblib.load('model.pkl')

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Stateless prediction endpoint"""
    # No state stored between requests
    features = request.features
    prediction = model.predict([features])[0]
    return {"prediction": prediction}
```

### Stateful Architecture

**Stateful Design:**
- Maintain state between requests
- Session management
- Context preservation

**When to Use:**
- Conversational AI (chatbots)
- Multi-step workflows
- User-specific model fine-tuning

**Implementation:**
```python
# Stateful service with session management
from fastapi import FastAPI
from typing import Dict

app = FastAPI()
sessions: Dict[str, dict] = {}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Stateful prediction with context"""
    session_id = request.session_id
    
    # Retrieve or create session
    if session_id not in sessions:
        sessions[session_id] = {
            'context': [],
            'user_preferences': {}
        }
    
    session = sessions[session_id]
    
    # Use context in prediction
    features = request.features
    context = session['context']
    
    # Enhanced prediction with context
    prediction = model.predict_with_context(features, context)
    
    # Update session state
    session['context'].append({
        'features': features,
        'prediction': prediction
    })
    
    return {"prediction": prediction, "session_id": session_id}
```

**Stateful with Redis:**
```python
import redis

class SessionManager:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get_session(self, session_id):
        """Get session from Redis"""
        data = self.redis.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return {'context': [], 'preferences': {}}
    
    def save_session(self, session_id, session_data, ttl=3600):
        """Save session to Redis"""
        self.redis.setex(
            f"session:{session_id}",
            ttl,
            json.dumps(session_data)
        )
```

---

## High Availability & Reliability

### High Availability

**HA Strategies:**

**1. Redundancy:**
- Multiple instances
- Multiple data centers
- Failover mechanisms

**2. Health Checks:**
```python
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    checks = {
        'model_loaded': model is not None,
        'database': check_database(),
        'cache': check_cache(),
        'disk_space': check_disk_space()
    }
    
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        return {"status": "unhealthy", "checks": checks}, 503
```

**3. Circuit Breaker:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half_open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e

# Usage
breaker = CircuitBreaker()

def predict_with_breaker(features):
    return breaker.call(model.predict, features)
```

**4. Graceful Degradation:**
```python
def predict_with_fallback(features):
    """Predict with fallback to simpler model"""
    try:
        # Try complex model
        return complex_model.predict(features)
    except Exception as e:
        logger.warning(f"Complex model failed: {e}")
        # Fallback to simple model
        return simple_model.predict(features)
```

### Monitoring & Observability

**Key Metrics to Monitor:**

**1. System Metrics:**
- CPU usage
- Memory usage
- GPU utilization
- Network I/O
- Disk I/O

**2. Application Metrics:**
- Request rate (RPS)
- Latency (p50, p95, p99)
- Error rate
- Prediction accuracy
- Cache hit rate

**3. Business Metrics:**
- User engagement
- Conversion rate
- Revenue impact

**Monitoring Implementation:**
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter('ml_requests_total', 'Total requests')
request_latency = Histogram('ml_request_latency_seconds', 'Request latency')
prediction_accuracy = Gauge('ml_prediction_accuracy', 'Prediction accuracy')
active_connections = Gauge('ml_active_connections', 'Active connections')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    request_count.inc()
    active_connections.inc()
    
    try:
        features = request.features
        prediction = model.predict([features])[0]
        
        # Record latency
        request_latency.observe(time.time() - start_time)
        
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise
    finally:
        active_connections.dec()
```

**Logging:**
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_prediction(request_id, features, prediction, latency):
    """Structured logging"""
    log_entry = {
        'request_id': request_id,
        'timestamp': time.time(),
        'features_hash': hash(str(features)),
        'prediction': prediction,
        'latency_ms': latency * 1000,
        'model_version': model.version
    }
    logger.info(json.dumps(log_entry))
```

**Distributed Tracing:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model_version", model.version)
        
        with tracer.start_as_current_span("preprocess"):
            features = preprocess(request.features)
        
        with tracer.start_as_current_span("inference"):
            prediction = model.predict([features])[0]
        
        span.set_attribute("prediction", prediction)
        return {"prediction": prediction}
```

---

## Best Practices

### Design Principles

1. **Start Simple**: Begin with basic architecture, optimize later
2. **Design for Scale**: Plan for growth from the start
3. **Fail Gracefully**: Handle errors and degrade gracefully
4. **Monitor Everything**: Track metrics, logs, traces
5. **Automate Operations**: CI/CD, auto-scaling, auto-recovery

### Performance Optimization

1. **Model Optimization**: Quantization, pruning, distillation
2. **Caching**: Cache predictions, features, embeddings
3. **Batch Processing**: Process multiple requests together
4. **Async Processing**: Use async for I/O operations
5. **Resource Management**: Right-size instances, use spot instances

### Cost Optimization

1. **Right-Sizing**: Match resources to workload
2. **Reserved Instances**: For predictable workloads
3. **Spot Instances**: For fault-tolerant workloads
4. **Auto-Scaling**: Scale down during low traffic
5. **Caching**: Reduce compute costs

---

## Resources

### Further Reading

- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)
- [System Design Primer](https://github.com/donnemartin/system-design-primer)

### Tools

- **Load Balancing**: Nginx, HAProxy, AWS ELB
- **Caching**: Redis, Memcached
- **Message Queues**: RabbitMQ, Kafka, AWS SQS
- **Monitoring**: Prometheus, Grafana, Datadog
- **Tracing**: Jaeger, Zipkin, OpenTelemetry

---

**Remember**: System design is iterative. Start with a simple architecture, measure performance, identify bottlenecks, and optimize. Focus on the metrics that matter for your use case!

