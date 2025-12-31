# Docker Complete Tutorial for Machine Learning

Comprehensive guide to Docker for data scientists and ML engineers. Learn to containerize your ML applications, ensure reproducibility, and deploy with confidence.

## Table of Contents

- [Introduction to Docker](#introduction-to-docker)
- [Why Docker for ML?](#why-docker-for-ml)
- [Installation and Setup](#installation-and-setup)
- [Docker Basics](#docker-basics)
- [Dockerfile Deep Dive](#dockerfile-deep-dive)
- [Docker Compose](#docker-compose)
- [Docker for Machine Learning](#docker-for-machine-learning)
- [Best Practices](#best-practices)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Topics](#advanced-topics)
- [Practice Exercises](#practice-exercises)
- [Additional Resources](#additional-resources)

---

## Introduction to Docker

### What is Docker?

**Docker** is a platform that uses containerization to package applications and their dependencies into lightweight, portable containers. Think of it as a shipping container for software - it ensures your application runs the same way everywhere.

### Key Concepts

**Container vs Virtual Machine:**
- **VM**: Full operating system, heavy, slow to start
- **Container**: Shares host OS, lightweight, fast to start
- **Container**: Isolated environment, portable, consistent

**Docker Components:**
- **Image**: Blueprint for containers (like a class in OOP)
- **Container**: Running instance of an image (like an object)
- **Dockerfile**: Instructions to build an image
- **Docker Hub**: Registry for sharing images
- **Docker Compose**: Tool for multi-container applications

### Why Containers Matter

**Benefits:**
1. **Reproducibility**: Same environment everywhere
2. **Isolation**: No conflicts between projects
3. **Portability**: Run on any machine with Docker
4. **Scalability**: Easy to scale applications
5. **Consistency**: Dev, staging, production all identical

---

## Why Docker for ML?

### The ML Problem

**Without Docker:**
- "Works on my machine" syndrome
- Dependency conflicts between projects
- Different Python versions
- Missing system libraries
- Inconsistent environments

**With Docker:**
- Same environment for everyone
- Isolated dependencies
- Reproducible experiments
- Easy deployment
- Consistent production environment

### ML-Specific Use Cases

1. **Model Deployment**: Package model + API + dependencies
2. **Experiment Reproducibility**: Same environment for all experiments
3. **Development Environment**: Consistent setup for team
4. **CI/CD Pipelines**: Test in containers
5. **Model Serving**: Containerize prediction services
6. **Data Processing**: Isolated data pipelines

---

## Installation and Setup

### Install Docker

**macOS:**
```bash
# Download Docker Desktop for Mac
# https://www.docker.com/products/docker-desktop

# Or using Homebrew
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
```

**Windows:**
```bash
# Download Docker Desktop for Windows
# https://www.docker.com/products/docker-desktop
```

### Verify Installation

```bash
# Check Docker version
docker --version

# Test Docker
docker run hello-world

# Check Docker info
docker info
```

### Docker Desktop (Recommended)

Docker Desktop provides a GUI and includes:
- Docker Engine
- Docker CLI
- Docker Compose
- Kubernetes (optional)

---

## Docker Basics

### Essential Commands

**1. Working with Images:**
```bash
# List images
docker images

# Pull image from Docker Hub
docker pull python:3.9

# Remove image
docker rmi python:3.9

# Remove all unused images
docker image prune -a
```

**2. Working with Containers:**
```bash
# Run container
docker run python:3.9 python --version

# Run interactively
docker run -it python:3.9 bash

# Run in background (detached)
docker run -d python:3.9 sleep 3600

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop <container_id>

# Start stopped container
docker start <container_id>

# Remove container
docker rm <container_id>

# Remove all stopped containers
docker container prune
```

**3. Container Management:**
```bash
# Execute command in running container
docker exec -it <container_id> bash

# View container logs
docker logs <container_id>

# Follow logs
docker logs -f <container_id>

# Inspect container
docker inspect <container_id>

# View container stats
docker stats
```

**4. Port Mapping:**
```bash
# Map host port to container port
docker run -p 8000:8000 python:3.9

# Map multiple ports
docker run -p 8000:8000 -p 5000:5000 python:3.9
```

**5. Volume Mounting:**
```bash
# Mount directory
docker run -v /host/path:/container/path python:3.9

# Mount current directory
docker run -v $(pwd):/app python:3.9

# Named volume
docker volume create myvolume
docker run -v myvolume:/data python:3.9
```

**6. Environment Variables:**
```bash
# Set environment variable
docker run -e VAR_NAME=value python:3.9

# Use .env file
docker run --env-file .env python:3.9
```

---

## Dockerfile Deep Dive

### What is a Dockerfile?

A **Dockerfile** is a text file with instructions to build a Docker image. Each instruction creates a layer in the image.

### Basic Dockerfile Structure

```dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["python", "app.py"]
```

### Dockerfile Instructions

**1. FROM:**
```dockerfile
# Base image
FROM python:3.9-slim

# Use specific tag
FROM python:3.9.7-slim

# Use Alpine (smaller)
FROM python:3.9-alpine
```

**2. WORKDIR:**
```dockerfile
# Set working directory
WORKDIR /app

# All subsequent commands run in /app
```

**3. COPY vs ADD:**
```dockerfile
# COPY (recommended) - simple file copy
COPY requirements.txt .
COPY src/ ./src/

# ADD - can extract archives and download URLs
ADD archive.tar.gz /app/
ADD https://example.com/file.txt /app/
```

**4. RUN:**
```dockerfile
# Execute commands
RUN apt-get update && apt-get install -y git

# Chain commands (reduces layers)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**5. ENV:**
```dockerfile
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

# Multiple variables
ENV VAR1=value1 \
    VAR2=value2
```

**6. ARG:**
```dockerfile
# Build-time arguments
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim

# Pass during build: docker build --build-arg PYTHON_VERSION=3.10 .
```

**7. EXPOSE:**
```dockerfile
# Document port (doesn't actually publish)
EXPOSE 8000
```

**8. CMD vs ENTRYPOINT:**
```dockerfile
# CMD - default command (can be overridden)
CMD ["python", "app.py"]

# ENTRYPOINT - always executed
ENTRYPOINT ["python", "app.py"]

# Combine (ENTRYPOINT + CMD)
ENTRYPOINT ["python"]
CMD ["app.py"]
# Final command: python app.py
```

**9. USER:**
```dockerfile
# Run as non-root user (security)
RUN useradd -m appuser
USER appuser
```

**10. LABEL:**
```dockerfile
# Add metadata
LABEL maintainer="your.email@example.com"
LABEL version="1.0"
```

### Complete ML Dockerfile Example

```dockerfile
# Use Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy requirements first (for layer caching)
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Builds

Optimize image size by using multiple stages:

```dockerfile
# Stage 1: Build
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Run application
CMD ["python", "app.py"]
```

### Building Images

```bash
# Build image
docker build -t myapp:latest .

# Build with tag
docker build -t myapp:v1.0.0 .

# Build with build args
docker build --build-arg PYTHON_VERSION=3.10 -t myapp:latest .

# Build without cache
docker build --no-cache -t myapp:latest .

# Build from specific Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .
```

---

## Docker Compose

### What is Docker Compose?

**Docker Compose** is a tool for defining and running multi-container Docker applications. Use a YAML file to configure services.

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  # ML API service
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.joblib
    volumes:
      - ./models:/models
    restart: unless-stopped

  # Database service
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=mluser
      - POSTGRES_PASSWORD=mlpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis for caching
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Docker Compose Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs

# Execute command in service
docker-compose exec ml-api bash

# Scale service
docker-compose up --scale ml-api=3
```

### ML Application with Compose

```yaml
version: '3.8'

services:
  # FastAPI ML service
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.joblib
      - DATABASE_URL=postgresql://mluser:mlpass@postgres:5432/mldb
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL database
  postgres:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: mldb
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis cache
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Jupyter for development
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    ports:
      - "8888:8888"
    volumes:
      - .:/workspace
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  postgres_data:
  redis_data:
```

---

## Docker for Machine Learning

### ML-Specific Dockerfile Patterns

**1. Jupyter Notebook Environment:**
```dockerfile
FROM jupyter/scipy-notebook:latest

WORKDIR /workspace

# Install additional packages
RUN pip install --no-cache-dir \
    xgboost \
    lightgbm \
    catboost \
    optuna \
    mlflow

# Copy notebooks
COPY notebooks/ /workspace/notebooks/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
```

**2. Model Training Container:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && apt-get clean

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Run training
CMD ["python", "train.py"]
```

**3. Model Serving Container:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and API
COPY model.joblib .
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Complete ML Project Structure

```
ml-project/
├── Dockerfile              # Production API
├── Dockerfile.dev          # Development environment
├── Dockerfile.train        # Training container
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Python dependencies
├── .dockerignore          # Files to exclude
├── app.py                 # FastAPI application
├── train.py               # Training script
├── models/                # Trained models
├── data/                  # Data files
└── notebooks/             # Jupyter notebooks
```

### .dockerignore File

```dockerignore
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints

# Data (if large)
data/raw/
*.csv
*.parquet

# Models (if large)
models/*.joblib
models/*.pkl

# Logs
logs/
*.log

# Environment
.env
.env.local
```

---

## Best Practices

### 1. Layer Caching

Order Dockerfile instructions from least to most frequently changing:

```dockerfile
# Good: Dependencies change less frequently
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # Code changes frequently, so copy last
```

### 2. Use .dockerignore

Always use `.dockerignore` to exclude unnecessary files:

```dockerignore
__pycache__
*.pyc
.git
.env
data/
```

### 3. Multi-Stage Builds

Use multi-stage builds to reduce image size:

```dockerfile
FROM python:3.9-slim as builder
# ... build stage

FROM python:3.9-slim
COPY --from=builder /app /app
```

### 4. Non-Root User

Run containers as non-root user:

```dockerfile
RUN useradd -m appuser
USER appuser
```

### 5. Health Checks

Add health checks for production:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
```

### 6. Specific Tags

Use specific image tags, not `latest`:

```dockerfile
# Good
FROM python:3.9.7-slim

# Avoid
FROM python:latest
```

### 7. Minimize Layers

Combine RUN commands:

```dockerfile
# Good
RUN apt-get update && \
    apt-get install -y package1 package2 && \
    apt-get clean

# Avoid
RUN apt-get update
RUN apt-get install -y package1
RUN apt-get install -y package2
```

### 8. Use Official Images

Prefer official base images:

```dockerfile
# Good
FROM python:3.9-slim

# Avoid
FROM some-random-user/python:3.9
```

---

## Common Issues and Solutions

### Issue 1: Container Exits Immediately

**Problem:** Container starts and immediately stops.

**Solution:**
```dockerfile
# Use CMD instead of RUN
CMD ["python", "app.py"]

# Or keep container alive
CMD ["tail", "-f", "/dev/null"]
```

### Issue 2: Permission Denied

**Problem:** Can't write to mounted volumes.

**Solution:**
```dockerfile
# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser
```

### Issue 3: Large Image Size

**Problem:** Image is too large.

**Solution:**
- Use multi-stage builds
- Use Alpine images
- Remove unnecessary files
- Don't install dev dependencies in production

### Issue 4: Slow Builds

**Problem:** Docker builds are slow.

**Solution:**
- Leverage layer caching
- Use .dockerignore
- Use build cache
- Consider BuildKit

### Issue 5: Port Already in Use

**Problem:** Port conflict when running container.

**Solution:**
```bash
# Use different host port
docker run -p 8001:8000 myapp

# Or stop conflicting container
docker stop $(docker ps -q --filter "publish=8000")
```

---

## Advanced Topics

### Docker Networking

```bash
# Create network
docker network create ml-network

# Run container on network
docker run --network ml-network myapp

# Inspect network
docker network inspect ml-network
```

### Docker Volumes

```bash
# Create named volume
docker volume create model-storage

# Use volume
docker run -v model-storage:/models myapp

# List volumes
docker volume ls

# Inspect volume
docker volume inspect model-storage
```

### Docker Secrets

```bash
# Create secret
echo "mysecret" | docker secret create db_password -

# Use in service
docker service create --secret db_password myapp
```

### Docker BuildKit

Enable BuildKit for faster builds:

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Or use
docker buildx build .
```

---

## Practice Exercises

### Exercise 1: Basic Container

Create a Dockerfile for a simple Python script.

**Solution:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY script.py .
CMD ["python", "script.py"]
```

### Exercise 2: ML API Container

Containerize a FastAPI ML application.

**Solution:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Exercise 3: Multi-Container Setup

Create docker-compose.yml for ML API + Database.

**Solution:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: mldb
```

---

## Additional Resources

### Official Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Tutorials
- [Docker Get Started](https://docs.docker.com/get-started/)
- [Docker for Python Developers](https://docs.docker.com/language/python/)

### Books
- "Docker Deep Dive" by Nigel Poulton
- "The Docker Book" by James Turnbull

### Video Courses
- [Docker Tutorial (TechWorld with Nana)](https://www.youtube.com/watch?v=3c-iBn73dDE)
- [Docker Course (freeCodeCamp)](https://www.youtube.com/watch?v=fqMOX6JJhGo)

---

## Key Takeaways

1. **Docker ensures reproducibility**: Same environment everywhere
2. **Use multi-stage builds**: Reduce image size
3. **Leverage layer caching**: Order instructions wisely
4. **Use .dockerignore**: Exclude unnecessary files
5. **Run as non-root**: Security best practice
6. **Use Docker Compose**: For multi-container apps
7. **Add health checks**: For production deployments
8. **Keep images small**: Faster builds and deployments

---

**Remember**: Docker is essential for ML production. Start simple, practice regularly, and gradually adopt best practices. Containerization makes your ML applications portable, reproducible, and production-ready!

