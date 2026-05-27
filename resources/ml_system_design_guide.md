# ML System Design Comprehensive Guide

This guide is for anyone who wants machine learning to actually work in the real world: reliable, observable, and shippable. You do not have to read it in order. Pick a path below, wander, and come back when something breaks in your own stack and you want a map.

> **New to backend system design?** Read [System Design for Beginners](../system-design/README.md) first. It's a 21-lesson primer on the general concepts this guide assumes you know: HTTP, TCP, caching, load balancing, CAP, sharding, message queues. This guide then applies those ideas to ML specifically.

## Table of Contents

- [How to use this guide](#how-to-use-this-guide)
- [Key terms in simple words](#key-terms-in-simple-words)
- [Case study: a debit card risk score](#case-study-a-debit-card-risk-score)
- [Introduction to ML System Design](#introduction-to-ml-system-design)
- [Real world systems you are designing for](#real-world-systems-you-are-designing-for)
- [End to end ML lifecycle](#end-to-end-ml-lifecycle)
- [Requirements, SLAs, and Capacity](#requirements-slas-and-capacity)
- [Core Concepts](#core-concepts)
- [Why training and serving should match](#why-training-and-serving-should-match)
- [Scalability Patterns](#scalability-patterns)
- [Data Infrastructure](#data-infrastructure)
- [Model Serving Architecture](#model-serving-architecture)
- [RAG and LLM oriented serving](#rag-and-llm-oriented-serving)
- [High Availability & Reliability](#high-availability-reliability)
- [Monitoring and observability](#monitoring-observability)
- [Data Drift & Data Quality](#data-drift-labels-and-data-quality)
- [Security, Privacy & Compliance](#security-privacy-compliance)
- [Testing & Safe Rollouts](#testing-safe-rollouts)
- [Interview and design review checklist](#interview-and-design-review-checklist)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## How to use this guide

Same guide, three speeds: someone curious, someone wiring their first service, someone cramming for a design interview. Nothing here is meant to sound like a certification exam.

| You are… | Suggested path |
|-----------|----------------|
| **New to production ML** | Start with [key terms](#key-terms-in-simple-words), then [introduction](#introduction-to-ml-system-design), [real world shapes](#real-world-systems-you-are-designing-for), and the [lifecycle](#end-to-end-ml-lifecycle). Read the [case study](#case-study-a-debit-card-risk-score) once like a short story. On the first pass you can skip code blocks entirely. |
| **Comfortable with ML code, lighter on ops** | [Lifecycle](#end-to-end-ml-lifecycle) → [case study](#case-study-a-debit-card-risk-score) → [training versus serving](#why-training-and-serving-should-match) → [serving](#model-serving-architecture) → [monitoring](#monitoring-observability) → [testing and rollouts](#testing-safe-rollouts). |
| **Interview mode** | Skim the glossary, read the case study, then walk the [interview checklist](#interview-and-design-review-checklist) out loud. Drill only the sections where your answers feel thin. |

**How callouts work**

- Lines that start with **Put simply** give a gut level version of what follows.
- Code is here to **illustrate** shapes and responsibilities. Rename things, tighten timeouts, and match your org’s standards before you paste anything into prod.

---

## Key terms in simple words

| Term | Simple meaning | Why teams say it |
|------|----------------|------------------|
| **Latency** | How long **one** user waits from “ask” to “answer.” | Users leave when the spinner never stops. |
| **Throughput** | How many requests the system **finishes per second** (capacity). | Black Friday vs a quiet Tuesday differ here. |
| **p95 / p99 latency** | “95% (or 99%) of requests were **faster than** this time.” | One slow user is noise; **p99** catches “bad tail” experiences. |
| **Synchronous** | User waits on the line for the answer (web score, chat). | Needs predictable latency. |
| **Asynchronous / batch** | Work is queued; answers come later (nightly reports). | Cheaper; higher latency OK. |
| **Load balancer** | Distributes traffic across **copies** of the same service. | One machine should not be a single point of failure. |
| **Stateless service** | Any replica can answer; no hidden memory on one box. | Easiest to scale horizontally. |
| **Stateful service** | Remembers session or context (often in Redis or a DB). | Chat, multi step flows; harder to scale. |
| **Canary** | Send **a few %** of traffic to a new version first. | Limits blast radius of a bad deploy. |
| **Shadow traffic** | New version gets a **copy** of requests; users still see old output. | Compare stability before trusting predictions. |
| **RAG** | **Retrieve** snippets, then **generate** an answer grounded in them. | Reduces “made up” answers if retrieval is good. |
| **Drift** | Live data / labels no longer match training reality. | Model accuracy decays even if code never changed. |
| **Train and serve skew** | Training code path and serving code path **disagree** (subtle bugs). | Classic “great in Jupyter, bad in prod.” |
| **NFR** | Nonfunctional requirements: speed, uptime, cost, privacy, not the math of the model. | What system design questions usually test. |
| **Cold start** | New user or item has **little or no history**, so features are thin. | Recsys and ads need explicit fallbacks, not only a big model. |
| **Idempotency** | Repeating the same request with the same key does **not** double charge or double act. | Payments and writes that retry on timeouts stay safe. |
| **Error budget** | Allowed **unreliability** per month tied to an SLO (for example 99.9% uptime). | Lets you choose when to ship fast versus freeze for stability. |
| **Lineage** | **Traceable path** from raw data through transforms to a trained artifact and deploy. | Audits, rollbacks, and “which data trained this?” questions. |

---

## Introduction to ML System Design

> **Put simply,** ML system design is mostly about the **software and data plumbing** around a model. The algorithm is one chapter; uptime, latency, and refresh are the rest of the book.

### What is ML System Design?

People use the phrase to mean “design the whole thing that makes ML useful,” not only the loss curve. In practice you are aiming for systems that can grow with traffic, survive bad luck, stay fast enough that humans do not rage quit, stay understandable enough that someone can fix them at 2 a.m., and do not quietly bankrupt the company on cloud bills.

You will hear five qualities repeated in rooms and docs:

- **Scalable:** more load does not mean linear pain; you can add capacity in a controlled way.
- **Reliable:** redundancy, retries, and honest health checks so one bad disk is not a product emergency.
- **Performant:** latency and throughput match what the product promised.
- **Maintainable:** new teammates can change something without ritual sacrifice.
- **Cost effective:** money spent maps to business value, not mystery GPU hours.

### Why System Design Matters for ML

ML adds a few headaches on top of normal backend work. Inference can be expensive. Data volume is rarely cute. Models need refreshes without a two hour maintenance page that trends on Twitter. You constantly choose between answering now versus answering later in a batch. Everyone suddenly cares about **versioning** because “the model last Tuesday” is a real forensic object.

Compared with a plain CRUD service, ML stacks also wrestle with big artifacts in memory, **tight coupling to historical data**, behavior that drifts even when code is frozen, and hardware appetites that change when someone swaps an architecture.

**Interviews versus production.** In a timed discussion you will compress weeks of politics into minutes: still spell out **who consumes the output**, **what breaks user trust**, and **what you would measure first** if traffic doubled tonight. In production the same habits matter, except the whiteboard is Grafana and the clock is an incident channel.

---

## Real world systems you are designing for

> **Put simply,** different products stress **different bottlenecks**: latency, cost, accuracy, or compliance. The table below is a map of “what usually breaks.”

These patterns show up in interviews and on the job. Each stresses different parts of the stack.

| Scenario | What “good” looks like | Typical bottlenecks |
|----------|------------------------|---------------------|
| **Card not present fraud score** | Under 100 ms synchronous score; explainable reason codes; audit trail | Feature lookup from many sources; model size; cold start |
| **Homepage / feed ranking** | High throughput; graceful fallback if the ranker is slow | Embedding service; cache; fan out to many candidates |
| **Document Q&A (RAG)** | Grounded answers; citations; controllable cost | Vector search latency; chunking; LLM token budget |
| **Batch credit risk** | Nightly run completes in window; reproducible artifacts | Data volume; orchestration; backfills |
| **Vision on the edge** | Low power; offline; stable latency | Model size; quantization; OTA updates |

If you take one thing from the table: a notebook winner can still embarrass you in production when nobody wrote down how fast the answer must arrive, where the data actually comes from, or what happens when a dependency stalls.

---

## End to end ML lifecycle

> **Put simply,** a model is one box inside a **long assembly line** of data, training, release, and monitoring. If any stage is weak, users see failures even when the neural net is “state of the art.”

Think in **pipelines**, not files: data moves through stages; each stage has inputs, outputs, owners, and failure behavior.

```
Business goal → Data sources → Ingestion & storage
      → Feature engineering (batch / stream)
      → Training & evaluation → Model registry
      → Deployment (canary / shadow) → Online serving
      → Monitoring (latency, errors, drift) → Retrain / rollback
```

**Example (overnight risk refresh):** A retail bank retrains a delinquency model weekly. Raw payments land in a **data lake**; **Spark** builds training rows; **MLflow** stores metrics and the `model.pkl` artifact; **Airflow** triggers training; on success, a new version is registered and **10% canary** traffic is switched before full promotion. If **population drift** spikes, an alert opens a ticket and traffic rolls back to `v3.2`.

**Example (live product search):** Queries hit an **API gateway**, then **spell check** and **language detect**, then **retrieval** (BM25 and vector), then a **reranker model** on the top 50 candidates, then a JSON response **under ~200 ms** for most requests (p95). Caches store hot queries; a **static fallback** returns popularity sorted results if the ranker times out.

### Model registry, lineage, and promotion gates

A **registry** is not glamour; it is where teams agree what “the model” means: artifact URI, framework version, feature schema hash, evaluation snapshot, and approver. **Lineage** ties that bundle back to datasets and code commits so a rollback is not guesswork.

Promotion should read like a release policy, not a vibe check: **who can move canary to 100%**, which **automated checks** must be green (latency SLO on shadow, calibration drift bounds, fairness slice tests if you promised them), and how long an **embargo** sits before marketing rewrites the FAQ. If you cannot answer “which artifact is in prod right now?” in one query, you are still in notebook territory.

---

## Case study: a debit card risk score

Maya runs backend for a regional bank. Product wants a **pre authorization risk score**: when someone taps their card at a coffee shop, the checkout app calls an internal API and gets back a number between zero and one plus a few reason codes. Nothing fancy on the surface. Under the hood it is a full system design exercise.

**Week zero, in a room with sticky notes.** Compliance insists on an audit trail. Fraud wants answers under a second at the busy lunch hour. Finance caps GPU spend. Maya writes those down as real targets, not vibes. That is already system design: you are negotiating **latency**, **throughput**, **cost**, and **explainability** before anyone opens PyTorch.

**Data path.** Authorization events land on a stream. Batch jobs fold yesterday’s stream into a warehouse table for training. Online serving reads **recent aggregates** (rolling spend, merchant category mix, device fingerprint freshness) from a cache backed by a relational replica. Maya’s team resists the temptation to train on “whatever CSV is handy” and instead builds training rows from the **same logical definitions** the online service will use, with timestamps that respect when a field was actually known. That discipline is what keeps later surprises smaller.

**The bug that actually happened.** Training notebooks filled missing currency amounts with zero after converting to USD. The Java service sent `null` for the same field on a handful of edge cases. The model saw two different worlds. Approvals looked fine in backtests and weird in the pilot. The fix was boring and good: **one small Python package** shared by training and serving that defines imputation, clipping, and column order, versioned next to the model artifact. No philosophy, just fewer places for silence to hide.

**Shipping without heroics.** They pointed five percent of live traffic at the new model while the old one still decided the outcome (**shadow**), compared score distributions and latency tails, then moved to a **canary** where the new model actually influenced a slice of decisions with automatic rollback on error rate. When marketing asked to “just flip it,” Maya could point at the graph and say we already learned what we needed without betting the whole afternoon on one deploy button.

**What you can steal from Maya’s story without working at a bank.** (1) Write the boring requirements first. (2) Treat training and serving as **one product** with two runtimes, not two teams with two scripts. (3) Practice rollouts the way you practice fire drills: small percentages, measurable gates, a story you can tell auditors and tired engineers alike.

---

## Requirements, SLAs, and Capacity

> **Put simply,** before you draw architecture diagrams, write down **how fast**, **how big**, and **how expensive** the system must be. Those numbers drive every later choice.

Before drawing boxes, write down **non functional requirements (NFRs)**:

- **Latency:** p50 / p95 / p99 targets (users feel p95, finance cares about p99).
- **Throughput:** peak RPS, burst multiplier (e.g. 3× average for 5 minutes).
- **Availability:** “three nines” (99.9%) ≈ 43 minutes downtime per month, so budget error budgets accordingly.
- **Correctness and safety:** human in the loop? Rate limits? Idempotency keys for payments?
- **Cost:** $/1k requests, GPU hours/month, egress fees.

### Back of the envelope capacity (napkin math)

Suppose one CPU worker handles **20 inferences/s** at comfortable utilization, and you need **2 000 RPS** sustained.

```
Workers needed ≈ 2000 / 20 ≈ 100 workers
```

Add **headroom** (+30 to 50%) for bursts and deploy failures. If each worker uses **2 GB RAM**, you need **~200 to 300 GB** total addressable memory across the fleet (before replicas for HA).

This is not a substitute for load tests; it tells you whether your architecture is in the right **order of magnitude**.

### Clarifying questions (what to pin down first)

Before boxes and arrows, make the **problem statement** boringly precise. You are not being difficult; you are buying the design room later.

| Theme | Examples of what to ask |
|-------|-------------------------|
| **Consumer** | Who calls this (mobile app, batch job, another service)? Sync or async? |
| **Scale** | Orders of magnitude for RPS, daily rows, peak versus steady, multi region or not. |
| **Latency budget** | Hard ceiling (card swipe) versus soft (dashboard refresh). Where is caching allowed? |
| **Freshness** | How stale can features or an index be before the product lies to users? |
| **Labels and risk** | How are labels produced, delayed, or noisy? Human review for high stakes? |
| **Compliance** | PII, retention, export or delete, model explainability expectations. |
| **Success** | Business metric first, model metric second, so you do not optimize the wrong score. |

If the room cannot answer three of these, your first proposal should stay **small and observable**: one serving path, one way to replay traffic, one dashboard that proves the story.

---

## Core Concepts

> **Put simply,** this section is the **vocabulary** of system design, requests, latency, throughput. If words feel new, read slowly and match each idea to an app you use daily (maps, banking, streaming).

### Requests & Responses

**Request Flow in ML Systems:**
```
Client Request
    ↓
API Gateway / Load Balancer
    ↓
Preprocessing service
    ↓
Model serving layer
    ↓
Post processing
    ↓
Response to Client
```

**Request Types:**
- **Synchronous**: Real time predictions (low latency required)
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
1. **Network Latency**: Request and response transmission
2. **Preprocessing Time**: Data transformation
3. **Model Inference Time**: Prediction computation
4. **Post processing time**: Result formatting

**Latency Targets:**
- **Real time ML**: < 100ms (recommendation systems)
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
    # Non blocking prediction
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, features)
    return result
```

### Throughput

**Throughput Metrics:**
- **Requests Per Second (RPS)**: Number of requests handled
- **Predictions Per Second (PPS)**: Actual predictions made
- **Queries Per Second (QPS)**: Database and vector DB queries

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

**Throughput versus latency tradeoff:**
- Higher batch size → Higher throughput, Higher latency
- Lower batch size → Lower throughput, Lower latency
- Need to balance based on use case

### GPU sharing and online versus batch inference

**Online inference** usually chases **tail latency**: small batches or batch size one, bursty traffic, and GPUs that look “underutilized” in averages while still mattering at p99. **Batch scoring** (nightly features, backfills, re-embedding corpora) chases **throughput** and happily fills a GPU for minutes.

When both land on the same fleet, you get classic contention: a long batch job can steal memory bandwidth and jitter the latency of live requests. Teams reduce pain by **splitting pools** (nodes or queues labeled online versus batch), **hard quotas** per namespace, or **time windows** that forbid heavy batch during peak hours. Smaller orgs sometimes accept shared GPUs but should still **measure queue depth** and p99 side by side, not only average utilization.

If you merge training and serving on shared accelerators, be explicit about **preemption and cost**: training is elastic; user facing inference is not. Treat that boundary as a product decision, not only a scheduler detail.

---

## Why training and serving should match

> **Put simply,** **training** is “study for the exam for weeks”; **inference** is “answer one exam question in seconds.” **Skew** means you accidentally studied different material than what appears on the real test.

**Training** optimizes parameters on historical data (often batch, high throughput, minutes to hours). **Inference** applies fixed parameters to live requests (often low latency, milliseconds unless batch scoring).

**Train and serve skew** means “the model sees a different distribution or encoding at serve time than during training.” Classic causes:

| Cause | Real life symptom | Mitigation |
|-------|-------------------|------------|
| Different preprocessing code paths | Sudden accuracy drop after deploy | Shared transformation library (same code in train & serve) |
| Missing default for new categorical level | `NaN` features at runtime | Explicit “unknown” bucket seen in training |
| Point in time leakage in training labels | Great offline metrics, bad online | As of joins; hold out by time |
| Different scaling (mean and standard deviation) | Drift in feature magnitudes | Store scaler params with the model artifact; version together |

**Minimal pattern: bundle transforms with the model**

```python
# training_pipeline.py (simplified): one object knows how to featurize and predict
import json
from pathlib import Path

import joblib


class ChurnModelPackage:
    """Ship preprocessing + model as one unit to reduce skew."""

    def __init__(self, model, feature_names, impute_values):
        self.model = model
        self.feature_names = feature_names
        self.impute_values = impute_values

    def featurize(self, raw: dict) -> list[float]:
        row = []
        for name in self.feature_names:
            v = raw.get(name)
            if v is None or v == "":
                v = self.impute_values.get(name, 0.0)
            row.append(float(v))
        return row

    def predict_proba(self, raw: dict) -> float:
        x = [self.featurize(raw)]
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        meta = {"feature_names": self.feature_names, "impute_values": self.impute_values}
        (path / "metadata.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "ChurnModelPackage":
        model = joblib.load(path / "model.joblib")
        meta = json.loads((path / "metadata.json").read_text())
        return cls(model, meta["feature_names"], meta["impute_values"])


# Example: same class imported in training notebook and in FastAPI service
```

If the imports look heavy, ignore them on first read. The idea is smaller than the code: one object owns both **how a row gets cleaned** and **how the prediction runs**, so training and serving are not allowed to drift apart quietly.

**Feature store (idea):** offline jobs write **training snapshots**; online path reads **low latency feature vectors** keyed by `user_id` or `session_id`. Stores like Feast, Tecton, or in house Redis and SQL combinations exist to keep **one definition** of “7 day spend” for train and serve.

---

## Scalability Patterns

> **Put simply,** when one computer is not enough, you either **make it bigger** (vertical) or **add more computers** (horizontal) and **spread work** (load balancing, queues, caches).

### Load Balancing

**Why Load Balancing for ML?**
- Distribute requests across multiple model instances
- Handle traffic spikes
- Improve availability (if one instance fails)

**Load Balancing Strategies:**

**1. Round Robin:**
```python
# Simple round robin
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

**4. Model aware routing:**
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
- Cost effective for small scale

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
- Cost effective at scale

**Implementation (Kubernetes YAML, not Python)**

```yaml
# deployment.yaml: stateless inference replicas
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
          image: ghcr.io/your-org/ml-api:1.4.2
          ports:
            - containerPort: 8000
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

**Autoscaling triggers:**
- CPU utilization > 70%
- Memory usage > 80%
- Request queue length
- Custom metrics (prediction latency)

---

## Data Infrastructure

> **Put simply,** models sit on top of **storage** (tables, blobs, vectors, time series). Pick storage the way you pick a backpack, by what you must carry and how often you open it.

### Databases

**Database Types for ML:**

**1. Relational Databases (SQL):**
- Structured data
- ACID transactions
- Examples: PostgreSQL, MySQL

**2. NoSQL Databases:**
- Unstructured and semi structured data
- High scalability
- Examples: MongoDB, Cassandra

**3. Vector Databases:**
- Store embeddings
- Similarity search
- Examples: Pinecone, Weaviate, FAISS

**4. Time series databases:**
- Metrics and monitoring data
- Examples: InfluxDB, TimescaleDB

### Replication

**Database Replication:**
- **Primary with replicas**: One writer, many readers
- **Multi leader (active active)**: Multiple write nodes
- **Read replicas**: Scale read operations

**Why Replication for ML:**
- Separate read and write workloads
- Scale feature retrieval
- Improve availability

**Example Configuration:**
```python
# Read from replica, write to primary
class DatabaseRouter:
    def __init__(self):
        self.primary = connect_primary()
        self.replicas = [connect_replica(i) for i in range(3)]
        self.replica_index = 0
    
    def get_read_connection(self):
        """Round robin read replicas"""
        conn = self.replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replicas)
        return conn
    
    def get_write_connection(self):
        """Always use primary for writes"""
        return self.primary

# Usage
router = DatabaseRouter()

# Read from replica
features = router.get_read_connection().query("SELECT * FROM features")

# Write to primary
router.get_write_connection().execute("INSERT INTO predictions ...")
```

### Sharding

**Sharding Strategies:**

**1. Feature based sharding:**
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

**2. Hash based sharding:**
```python
def get_shard(user_id, num_shards=4):
    """Route to shard based on hash"""
    hash_value = hash(user_id)
    return hash_value % num_shards

# Route requests to appropriate shard
shard_id = get_shard(user_id)
features = shards[shard_id].get(user_id)
```

**3. Range based sharding:**
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

**1. Request and response pattern:**
```python
import redis
import json
import uuid

class MLMessageQueue:
    def __init__(self, redis_client):
        import time

        self._time = time
        self.redis = redis_client
        self.request_queue = "ml:requests"
        self.response_queue = "ml:responses"
    
    def send_request(self, features):
        """Send prediction request"""
        request_id = str(uuid.uuid4())
        message = {
            'request_id': request_id,
            'features': features,
            'timestamp': self._time.time()
        }
        self.redis.lpush(self.request_queue, json.dumps(message))
        return request_id
    
    def get_response(self, request_id, timeout=30):
        """Get prediction response"""
        start_time = self._time.time()
        while self._time.time() - start_time < timeout:
            response = self.redis.get(f"ml:response:{request_id}")
            if response:
                return json.loads(response)
            self._time.sleep(0.1)
        return None
```

**2. Publish subscribe pattern:**
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

> **Put simply,** **serving** is how the saved model becomes an **HTTP (or gRPC) service** that other systems call. Stateless is usually easier; stateful is for conversations and multi step flows.

### Stateless Architecture

**Stateless Design:**
- No server side session state
- Each request is independent
- Easy to scale horizontally

**Benefits:**
- Horizontal scaling
- Load balancing
- Fault tolerance
- Simple deployment

**Implementation:**
```python
# Stateless API service (minimal runnable shape; adapt model load to your stack)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")  # loaded once at worker start


class PredictionRequest(BaseModel):
    features: list[float]


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Each request is independent; safe to scale replicas horizontally."""
    prediction = model.predict([request.features])[0]
    return {"prediction": float(prediction), "model_version": "1.4.2"}
```

### Cascading models when latency or budget bites

Not every request deserves the same price tag. A common production pattern is a **cascade**: a cheap gate (rules, tiny classifier, or shallow retrieval) decides whether to call a heavy model at all, or which size of model to use. Examples: skip the billion parameter reranker when the query is obviously navigational; route “suspicious” events only to a deep fraud net; answer from cache when embeddings for that document have not changed.

State the tradeoff honestly: cascades reduce average cost and tail latency, but add **routing logic** you must monitor (false negatives on the gate hurt you silently). Log which stage served each request so you can tune thresholds with data instead of folklore.

### Stateful Architecture

**Stateful Design:**
- Maintain state between requests
- Session management
- Context preservation

**When to Use:**
- Conversational AI (chatbots)
- Multi step workflows
- User specific model fine tuning

**Implementation:**
```python
# Stateful service with session management
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()
sessions: Dict[str, dict] = {}


class StatefulPredictionRequest(BaseModel):
    session_id: str
    features: list[float]


@app.post("/predict")
async def predict(request: StatefulPredictionRequest):
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
import json
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

## RAG and LLM oriented serving

> **Put simply,** instead of asking the LLM from memory alone, you **look up passages** from your docs first, then ask the model to stay faithful to those passages, like an open book exam.

**Retrieval augmented generation (RAG)** answers user questions by **(1)** retrieving relevant chunks from your docs, **(2)** stuffing them into a prompt, **(3)** asking an LLM to answer **using only that context**. In production you are really designing **three systems**: ingestion and indexing, retrieval, and generation.

**Real life shape:** Internal IT bot at a company with about five thousand people. PDFs and Confluence pages change daily; users expect citations. Design choices:

1. **Chunking:** 300 to 800 tokens with overlap; store `doc_id`, `page`, `last_indexed_at` in metadata.
2. **Vector index:** ANN index (FAISS, Qdrant, managed Pinecone, etc.) sized for recall@k and QPS; separate **staging** vs **prod** indices.
3. **Guardrails:** max retrieved chunks, max output tokens, **timeout** then “I could not find an authoritative answer.”
4. **Cost:** cache embeddings for stable documents; embed again only on diff.

**Tiny retrieval stub (no LLM call, just shows control flow):**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    text: str
    score: float


def retrieve(query: str, index: list[Chunk], top_k: int = 5) -> list[Chunk]:
    """Replace with vector search; keyword overlap stands in for 'similarity'."""
    tokens = set(query.lower().split())
    ranked = sorted(
        index,
        key=lambda c: len(tokens & set(c.text.lower().split())),
        reverse=True,
    )
    return ranked[:top_k]


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)
    return f"Answer using ONLY this context.\n\n{context}\n\nQuestion: {query}"


# Example: two policy snippets and a user question
INDEX = [
    Chunk("policy_pto", "Employees accrue 15 PTO days per calendar year.", 0.0),
    Chunk("policy_remote", "Remote work requires manager approval once per quarter.", 0.0),
]
user_q = "How many PTO days do I get?"
prompt = build_prompt(user_q, retrieve(user_q, INDEX))
# Next step: call your LLM API with `prompt`, temperature, and max_tokens caps.
```

**Latency budget:** If your product SLA is 3 s from end to end, allocate explicit ceilings (for example retrieval 200 ms, rerank 100 ms, LLM 2.4 s) so one slow stage does not starve the others.

---

## High Availability & Reliability

> **Put simply,** **availability** means “still works when something breaks.” You add spare capacity, health checks, and circuit breakers so one bad dependency does not take down the whole product.

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
import time


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

### Multi region inference and disaster recovery for models

**Multi region** usually means choosing between **regional replicas** that serve the same promoted build (one pipeline; each region pulls the same versioned image and weights) and harder **active active** setups where traffic and data truly split. For ML, add **embedding and index freshness**: a vector store in region B that lags region A can look “up” while answers drift.

**Disaster recovery** for models is mostly **recovering the right artifact fast**: registry pin, object storage replication, and a runbook that says how to repoint DNS or the load balancer to a known good version. Agree on **RTO and RPO** in human terms: how long can scoring be down, and how old a model or index is acceptable during a cutover? Separate “**inference unavailable**” (fail over replicas, serve a simpler model) from “**wrong model in every region**” (freeze promotion, roll back via registry, replay shadow traffic before widening again).

Data residency and **cross region egress** can dominate cost; design regional boundaries before you promise global low latency on giant checkpoints.

### Monitoring & Observability

> **Put simply,** **metrics** are numbers over time (how busy, how slow). **Logs** are lines of text when something happens. **Traces** follow one request across many services, like a parcel tracking number through warehouses.

**Key Metrics to Monitor:**

**1. System Metrics:**
- CPU usage
- Memory usage
- GPU utilization
- Network input and output
- Disk input and output

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
import logging
import time

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

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

### Data drift, labels, and data quality

> **Put simply,** the world changes; your frozen model does not. **Drift** is the gap between “what the model expects” and “what users actually send today.”

**Concept drift:** the world changed; the old decision boundary is wrong. **Covariate drift:** inputs shift but the conditional label distribution **P(y|x)** is still learnable with a new model. **Prior shift:** class balance changes.

**Real life:** After a holiday, online commerce traffic spikes on gift categories. A conversion model trained on October data sees new category mix, **latency stays green** while **business metrics drop**. Fix: refresh training data, segment models by region, or add a short horizon **auto retrain** with guardrails.

**What to log for later debugging (not raw PII):** hashed user id, model version, feature schema version, bucketized feature summaries, outcome when known.

**Offline vs online checks:**

| Check | What it tells you |
|-------|-------------------|
| Hold out by **time** | Will the model survive next week? |
| **Backtest** on replayed traffic | Rough impact of ranking changes |
| **Shadow** deploy (challenger gets traffic copy, no user impact) | Live latency & distribution match lab? |

**Offline metrics are a compass, not the destination.** Validation AUC can rise while revenue falls if labels lag, if the policy you evaluate differs from production routing, or if uplift is drowned out by inventory limits. Tie offline suites to **decision aware slices** (region, channel, cold users) and always leave budget to compare against **live business counters**, even if only on a small experiment slice.

---

## Security, Privacy & Compliance

> **Put simply,** a model endpoint is still a **public facing program**. Bad inputs, stolen keys, or leaked logs hurt users and the business the same way they would for any other API.

ML services are normal APIs: **validate inputs**, **authenticate callers**, **apply rate limits**, and **audit** sensitive actions.

- **PII in features:** minimize; tokenize; encrypt at rest; restrict log fields (GDPR and CCPA “right to erasure” applies to stored rows, not always to aggregated metrics).
- **Model theft:** treat large proprietary checkpoints like secrets: private artifact registry, signed images, no public bucket URLs.
- **Prompt injection (LLM and RAG):** treat user text as untrusted instructions; separate system prompts; tool calls behind allowlists.
- **Fairness & abuse:** block automated account creation scoring; review disparate impact when the model affects protected classes.

**Input validation example (FastAPI):**

```python
from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1_000_000)
    country: str = Field(min_length=2, max_length=2)

    @field_validator("country")
    @classmethod
    def upper_iso(cls, v: str) -> str:
        c = v.upper()
        if not c.isalpha():
            raise ValueError("country must be ISO style letters")
        return c
```

---

## Testing & Safe Rollouts

> **Put simply,** never flip 100% of customers to a new model in one click unless you must. **Gradual** exposure (canary and shadow) is how teams learn safely on real traffic.

**Champion and challenger:** production keeps **champion**; **challenger** receives a slice (e.g. 5%) for split comparison on business KPIs, not only accuracy.

**Canary deploy:** route 1% → 10% → 100% while watching error rate and p99 latency; automatic rollback if SLO breached.

**Shadow traffic:** duplicate live requests to a new version; discard responses; compare latencies and distribution of scores, **users never see failures** from the candidate.

**Blue and green:** two full environments; flip load balancer when green is verified; fast rollback is flip back.

```python
# Feature flag style routing (concept only)
import hashlib


def choose_model_version(user_id: str) -> str:
    # sticky canary: same user always sees same version during an experiment
    bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
    return "v2" if bucket < 5 else "v1"
```

---

## Interview and design review checklist

> **Put simply,** interviews reward **structured thinking**, not buzzwords. Walk top to bottom: users → data → model → serving → reliability → observability → rollout → cost and legal.

When someone says “design a recommendation system” or “fraud detection at scale,” walk through:

1. **Users & goals:** Who calls the API? What decision happens with the score?
2. **Scale:** RPS, payload size, sync vs async, regional latency.
3. **Data:** sources, freshness, PII, training window, label definition.
4. **Features:** online and offline parity, store, cold users and items.
5. **Model:** family, size, GPU vs CPU, batching, multi model routing.
6. **Serving:** REST and gRPC, autoscaling, warm pools, timeouts, idempotency.
7. **Reliability:** retries, circuit breakers, degradation, multiple availability zones.
8. **Observability:** golden signals + ML metrics (calibration, drift proxies).
9. **Release:** canary and shadow, rollback, registry pins.
10. **Cost & compliance:** $/1k inferences, retention, legal constraints.

If you cover these ten buckets with **one concrete example each**, you usually demonstrate senior level system thinking.

### At the whiteboard: order and stopping criteria

A workable rhythm is: **two minutes** of clarifying questions, **five to eight minutes** for a happy path diagram (data, train, serve, observe), **five minutes** on failure and cost, **two minutes** on how you would phase an MVP. Say explicitly what you are **deferring** (for example multi region active active on day one) and what would trigger revisiting it.

Stop adding boxes when you can explain **how you would prove** the system healthy in the first week of traffic: three metrics, two alerts, one rollback lever. Interviewers remember coherence more than zoo architectures.

### Failure modes worth naming out loud

You do not need jargon; you need **stories**. Pick a few that fit the prompt: train and serve skew after a library upgrade; cache poisoning after a bad deploy; **thundering herd** when cold replicas spin up; embedding index stale while the UI promises “live”; GPU queue backlog when batch and online share a pool; silent label delay so training thinks yesterday’s outcomes are known today.

---

## Best Practices

> **Put simply,** start boring and reliable; add cleverness only where metrics prove you need it.

### Design Principles

1. **Start Simple**: Begin with basic architecture, optimize later
2. **Design for Scale**: Plan for growth from the start
3. **Fail Gracefully**: Handle errors and degrade gracefully
4. **Monitor Everything**: Track metrics, logs, traces
5. **Automate Operations**: CI and CD pipelines, autoscaling, auto recovery

### Performance Optimization

1. **Model Optimization**: Quantization, pruning, distillation
2. **Caching**: Cache predictions, features, embeddings
3. **Batch Processing**: Process multiple requests together
4. **Async Processing**: Use async for input output operations
5. **Resource Management**: Right size instances, use spot instances

### Cost Optimization

1. **Right sizing**: Match resources to workload
2. **Reserved Instances**: For predictable workloads
3. **Spot Instances**: For fault tolerant workloads
4. **Autoscaling**: Scale down during low traffic
5. **Caching**: Reduce compute costs

---

## Resources

### Further Reading

Structured ML system design (whether for interviews or architecture reviews) rewards the same muscle as classic distributed systems work: **tradeoffs, boundaries, and explicit failure handling**. Combine ML specific material with general backend and data engineering depth so latency, consistency, and cost arguments feel natural, not bolted on.

- [System Design for Beginners](../system-design/README.md) (in this repo) — backend foundations this guide builds on
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

**Remember**: System design is iterative. Start with a simple architecture, measure performance, identify bottlenecks, and optimize. Focus on the metrics that matter for your use case: the goal is a system people can **operate**, not a diagram that wins a single meeting.

