# Phase 13: Model Deployment

Learn to deploy machine learning models to production.

##  What You'll Learn

- Model Serialization
- REST APIs with Flask/FastAPI
- Docker for ML
- Cloud Deployment
- Model Monitoring
- Best Practices for Production

##  Topics Covered

### 1. Model Serialization
- **Pickle**: Python's native serialization
- **Joblib**: Better for NumPy arrays
- **H5/HDF5**: For Keras models
- **ONNX**: Cross-platform format
- **Saving**: Architecture + weights

### 2. REST APIs
- **Flask**: Simple web framework
  - Creating endpoints
  - Request/response handling
  - Error handling
- **FastAPI**: Modern, fast framework
  - Automatic documentation
  - Type hints
  - Async support
- **API Design**: Best practices

### 3. Docker for ML
- **Containerization**: Package model + dependencies
- **Dockerfile**: Define container
- **Docker Images**: Build and run
- **Docker Compose**: Multi-container apps
- **Benefits**: Reproducibility, portability

### 4. Cloud Deployment
- **AWS**: SageMaker, EC2, Lambda
- **Google Cloud**: Vertex AI, Cloud Run
- **Azure**: Azure ML, Container Instances
- **Heroku**: Simple deployment
- **Choosing Platform**: Based on needs

### 5. Production Server Setup
- **NGINX Configuration**: Reverse proxy, load balancing, SSL termination
- **SSL/TLS Setup**: Let's Encrypt certificates, auto-renewal
- **Domain Configuration**: DNS setup, subdomain routing
- **Security**: Rate limiting, API authentication, input validation
- **Error Handling**: Structured error responses, logging
- **AWS EC2 Setup**: Instance configuration, systemd services, firewall

### 6. Model Serving
- **Batch Inference**: Process in batches
- **Real-time Inference**: Low latency
- **A/B Testing**: Compare model versions (statistical significance, multi-armed bandits, sequential testing)
- **Canary Deployments**: Gradual rollout

### 7. Model Monitoring
- **Performance Metrics**: Track accuracy over time
- **Data Drift**: Detect distribution changes
- **Model Drift**: Performance degradation
- **Logging**: Track predictions and errors
- **Alerts**: Notify on issues

##  Learning Objectives

By the end of this module, you should be able to:
- Serialize and load models
- Create REST APIs for models
- Containerize ML applications
- Deploy to cloud platforms
- Configure production servers (NGINX, SSL, domain)
- Implement security best practices (rate limiting, authentication)
- Monitor deployed models

##  Projects

1. **Flask API**: Deploy a model with Flask
2. **FastAPI Service**: Build FastAPI service
3. **Docker Container**: Containerize ML app
4. **Cloud Deployment**: Deploy to AWS/GCP/Azure
5. **Monitoring Dashboard**: Track model performance

##  Key Concepts

- **API Endpoints**: Expose model as service
- **Containerization**: Package everything together
- **Scalability**: Handle multiple requests
- **Monitoring**: Track model health
- **Versioning**: Manage model versions

## Documentation & Learning Resources

**FastAPI:**
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

**Docker:**
- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Tutorial](https://docs.docker.com/get-started/)
- [Docker for Python Developers](https://docs.docker.com/language/python/)

**Flask:**
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Tutorial](https://flask.palletsprojects.com/tutorial/)

**MLflow:**
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html#deployment)

**Free Courses:**
- [FastAPI Course (YouTube)](https://www.youtube.com/watch?v=0sOvCWFmrtA) - Free tutorial
- [Docker Course (YouTube)](https://www.youtube.com/watch?v=fqMOX6JJhGo) - Free comprehensive course
- [ML Deployment (Coursera)](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) - Free audit available

**Tutorials:**
- [Deploying ML Models (Real Python)](https://realpython.com/flask-connexion-rest-api/)
- [Docker for Data Scientists](https://towardsdatascience.com/docker-for-data-scientists-9c0ce73e826e)
- [ML Model Deployment Guide](https://www.mlflow.org/docs/latest/models.html#deployment)
- [Setting up NGINX as Reverse Proxy (DigitalOcean)](https://www.digitalocean.com/community/tutorials/how-to-set-up-a-node-js-application-for-production-on-ubuntu-20-04)
- [SSL Certificate Setup with Let's Encrypt (DigitalOcean)](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-20-04)
- [Production FastAPI Deployment (TestDriven.io)](https://testdriven.io/blog/fastapi-deployment/)

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
- [Domain Name System (Wikipedia)](https://en.wikipedia.org/wiki/Domain_Name_System)

**Cloud Platforms:**
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform/docs)
- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Heroku Deployment Guide](https://devcenter.heroku.com/articles/getting-started-with-python)

**Video Tutorials:**
- [FastAPI Tutorial (Corey Schafer)](https://www.youtube.com/watch?v=0sOvCWFmrtA)
- [Docker Tutorial (TechWorld with Nana)](https://www.youtube.com/watch?v=3c-iBn73dDE)
- [NGINX Tutorial (LearnLinuxTV)](https://www.youtube.com/watch?v=9JqVp7j2zdc)
- [SSL/TLS Explained (PowerCert)](https://www.youtube.com/watch?v=jQVwXa5CQ2Q)
- [ML Deployment (Sentdex)](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

**[Complete Detailed Guide →](deployment.md)**

**Additional Resources:**
- [Advanced Topics →](deployment-advanced-topics.md) - Advanced deployment patterns, Kubernetes, edge deployment
- [Project Tutorial →](deployment-project-tutorial.md) - Step-by-step model deployment project
- [Quick Reference →](deployment-quick-reference.md) - Quick lookup guide for model deployment

---

**Previous Phase:** [12-natural-language-processing](../12-natural-language-processing/README.md)  
**Next Phase:** [14-mlops-basics](../14-mlops-basics/README.md)

