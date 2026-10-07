# Module 13: Model Deployment

Learn to serve machine learning models through a local or containerized API. Cloud and reverse-proxy topics are optional extensions, not a claim that you ran a full production site.

##  What You'll Learn

- Model Serialization
- REST APIs with Flask/FastAPI
- Docker for ML
- Optional cloud or PaaS deploy paths
- Model Monitoring basics
- Serve-path practices (validation, logging, versioning)

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
- **API Design**: Clear request/response contracts

### 3. Docker for ML
- **Containerization**: Package model + dependencies
- **Dockerfile**: Define container
- **Docker Images**: Build and run
- **Docker Compose**: Multi-container apps
- **Benefits**: Reproducibility, portability

### 4. Optional cloud or PaaS paths
- **AWS**: SageMaker, EC2, Lambda (overview + optional lab)
- **Google Cloud**: Vertex AI, Cloud Run
- **Azure**: Azure ML, Container Instances
- **Heroku / similar**: Simple staging deploy
- **Choosing Platform**: Based on needs and budget

### 5. Optional server setup (beyond local serve)
- **NGINX Configuration**: Reverse proxy, load balancing, SSL termination (survey or optional lab)
- **SSL/TLS Setup**: Let's Encrypt certificates, auto-renewal
- **Domain Configuration**: DNS setup, subdomain routing
- **Security**: Rate limiting, API authentication, input validation
- **Error Handling**: Structured error responses, logging
- **AWS EC2 Setup**: Instance configuration, systemd services, firewall (optional)

### 6. Model Serving
- **Batch Inference**: Process in batches
- **Online Inference**: Low-latency local or container API
- **A/B Testing**: Compare model versions (concepts and small demos)
- **Canary ideas**: Gradual rollout patterns (concepts)

### 7. Model Monitoring
- **Performance Metrics**: Track accuracy over time when labels arrive
- **Data Drift**: Detect distribution changes
- **Model Drift**: Performance degradation
- **Logging**: Track predictions and errors
- **Alerts**: Plan what you would notify on

##  Learning Objectives

By the end of this module, you should be able to:
- Serialize and load models
- Create a local REST API that scores inputs
- Containerize that API with Docker
- Optionally push the same image to a cloud or PaaS staging path
- Sketch reverse-proxy / TLS / auth patterns without claiming you hardened a live site
- Add basic validation, logging, and model version fields
- Outline monitoring signals for a served model

##  Projects

1. **Flask API**: Serve a model locally with Flask
2. **FastAPI Service**: Build a FastAPI scoring service
3. **Docker Container**: Containerize the ML API
4. **Optional staging deploy**: Push the container to AWS/GCP/Azure or a simple PaaS
5. **Monitoring notes**: Log scores/latency and sketch a small dashboard

##  Key Concepts

- **API Endpoints**: Expose model as a local or container service
- **Containerization**: Package everything together
- **Serve path**: Handle requests with clear contracts
- **Monitoring**: Track model health on the path you actually run
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
- [FastAPI Course (YouTube)](https://www.youtube.com/watch?v=0sOvCWFmrtA): Free tutorial
- [Docker Course (YouTube)](https://www.youtube.com/watch?v=fqMOX6JJhGo): Free comprehensive course
- [ML Deployment (Coursera)](https://www.coursera.org/learn/introduction-to-machine-learning-in-production): Free audit available

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

**[Complete Detailed Guide](deployment.md)**

**Additional Resources:**
- [Advanced Topics](deployment-advanced-topics.md): Advanced deployment patterns, Kubernetes, edge deployment
- [Project Tutorial](deployment-project-tutorial.md): Step-by-step model deployment project
- [Quick Reference](deployment-quick-reference.md): Quick lookup guide for model deployment

---

**Previous Module:** [12-natural-language-processing](../12-natural-language-processing/README.md)  
**Next Module:** [14-mlops-basics](../14-mlops-basics/README.md)

