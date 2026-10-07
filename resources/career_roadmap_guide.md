# Career Roadmap Guide: Role-Specific Learning Paths

This guide provides clear, role-specific learning paths for different careers in data science and machine learning. Each path includes recommended modules, projects, and resources tailored to the specific role.

**Honesty note:** Month ranges are realistic study estimates for a human pace (about **8–12 hours/week** with a job and life), not job guarantees. The low end assumes some prior coding comfort. The high end assumes more review, projects, and interrupted weeks. Titles like “Solution Architect” or “Research Scientist” also need depth, portfolio evidence, and (for research) math/experimentation practice beyond checklist completion.

**Canonical stage order and the module 15 (time series) branch** match the root [README](../README.md#learning-path-overview). Stage 6 is vision and language (modules **11–12**). Module **15** is an optional parallel track, see [TIME_SERIES_LEARNING_PATH.md](../TIME_SERIES_LEARNING_PATH.md). **SQL is Stage 1.5 (module 19)**, not Stage 7.5.

## Table of Contents

- [Data Analyst](#data-analyst)
- [Data Scientist](#data-scientist)
- [Machine Learning Engineer](#machine-learning-engineer)
- [LLM Engineer (Large Language Models)](#llm-engineer-large-language-models)
- [GenAI Solution Architect](#genai-solution-architect)
- [Computer Vision Engineer](#computer-vision-engineer)
- [AI Engineer (Generalist)](#ai-engineer-generalist)
- [Data Engineer](#data-engineer)
- [MLOps Engineer](#mlops-engineer)
- [Research Scientist](#research-scientist)
- [Business Intelligence Analyst](#business-intelligence-analyst)
- [Full-Stack AI Engineer](#full-stack-ai-engineer)
- [Backend Engineer](#backend-engineer)

---

## Data Analyst

**Role Focus**: Analyze data to provide insights, create reports, and support business decisions. Focus on data manipulation, visualization, and statistical analysis.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python Basics, Statistics | 2-3 months |
| **Data Fundamentals** | 01 | NumPy, Pandas, Visualization, EDA | 3-4 months |
| **SQL & Databases** | 19 | SQL, Database Fundamentals, NoSQL basics | 2-3 months |
| **ML literacy (optional)** | 02 | What models are for, train/test honesty *(optional)* | 0.5-1 month |
| **Essential Skills** | 20, 21 | Imbalanced Data, Model Explainability | 1-2 months |
| **Project add-ons** | 08 and/or 15 | Clustering / forecasting *(only if your projects need them)* | 1-2 months |
| **Total (core)** | **5 modules** | **Core path without 02 / 08 / 15** | **12-18 months** |

**Optional Module 02:** Useful if you want honest ML literacy (what a model can and cannot do). Not required for every analyst role.

**Modules 08 and 15:** Required only when you take segmentation (Module 08) or forecasting (Module 15) projects. Skip them for a dashboard-and-SQL focused path.

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python Basics, Statistics, Math Fundamentals | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit, EDA. BI tools (Tableau / Power BI) are optional companions, not a Module 01 pass/fail bar | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL Queries, Joins, Window Functions, OLAP/OLTP | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow, Best Practices | ⭐ Optional |
| **20-handling-imbalanced-data** | Data Quality, Resampling Techniques | ⭐⭐ Important |
| **21-model-explainability** | SHAP, LIME, Feature Importance | ⭐⭐ Important |
| **08-unsupervised-learning** | Clustering for segmentation projects | ⭐ Project-only |
| **15-time-series-analysis** | Forecasting projects *(pick one path in [TIME_SERIES_LEARNING_PATH](../TIME_SERIES_LEARNING_PATH.md))* | ⭐ Project-only |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | Customer Data Dashboard with Streamlit, House Price Prediction | Data Visualization, EDA, Dashboard Creation |
| **Intermediate** | Customer Segmentation *(needs Module 08)*, Time Series Forecasting *(needs Module 15)* | Clustering, Time Series Analysis |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Required** | [Ethics in ML](ethics_in_ml.md), [Stakeholder Communication](stakeholder_communication.md), [Experiment Design and A/B Testing](experiment_design_ab_testing.md) |
| **Tools** | [Excel Data Analysis Guide](excel_data_analysis_guide.md), [Power BI Guide](powerbi_guide.md), [Web Scraping Guide](web_scraping_guide.md) |
| **Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [Math Formulas](math_formulas.md), [Git Guide](git_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Python programming (Pandas, NumPy)
- Data visualization in Python (Matplotlib, Seaborn, Plotly). Tableau / Power BI when your workplace uses them
- SQL and database querying
- Statistical analysis and hypothesis testing
- Experiment design awareness (simple A/B and metric honesty)
- Excel for data analysis
- Data cleaning and preprocessing
- Dashboard creation (Streamlit, and/or a BI tool you choose)
- Report writing and data storytelling
- Ethics and stakeholder communication basics

---

## Data Scientist

**Role Focus**: Build predictive models, perform advanced statistical analysis, and extract insights from complex datasets. Bridge between business and technical teams.

**Before Module 04:** Read [Ethics in ML](ethics_in_ml.md). Classification and ranking work can harm people when labels, proxies, or deployment context are careless. Ethics is a required resource on this path, not a late add-on.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Math, Statistics | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, Visualization, EDA | 2-3 months |
| **SQL & Databases** | 19 | Queries, joins, window functions *(Stage 1.5, parallel with 01–02)* | 1-2 months |
| **Ethics + experiments (resources)** | resources | [Ethics in ML](ethics_in_ml.md), [Experiment Design](experiment_design_ab_testing.md), [Causal Inference](causal_inference_guide.md) *(before / beside Module 04)* | 0.5-1 month |
| **ML Basics** | 02-05 | ML Concepts, Regression, Classification, Evaluation | 3-4 months |
| **Advanced ML** | 06-07 | Ensemble Methods, Feature Engineering | 2-3 months |
| **Unsupervised Learning** | 08 | Clustering, Dimensionality Reduction | 1-2 months |
| **Time Series** | 15 | Time Series Analysis, Forecasting *(optional, [pick one path](../TIME_SERIES_LEARNING_PATH.md))* | 1-2 months |
| **GenAI lite (optional)** | 25 Days 1–3 | Prompting, retrieval basics, eval habits. Stop before agents unless you need them | 0.5-1 month |
| **Essential Skills** | 20-21 | Imbalanced Data, Explainability | 1-2 months |
| **Total** | **12 modules** | **Complete Path** *(GenAI lite optional)* | **18-28 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Linear Algebra, Statistics, Calculus | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Pandas, NumPy, Visualization, EDA | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow, Best Practices | ⭐⭐⭐ Critical |
| **03-supervised-learning-regression** | Linear/Polynomial Regression, Regularization | ⭐⭐⭐ Critical |
| **04-supervised-learning-classification** | Logistic Regression, Trees, SVM, KNN, Naive Bayes *(after ethics resource)* | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Cross-Validation, Hyperparameter Tuning, Calibration | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Bagging, Boosting, Stacking | ⭐⭐⭐ Critical |
| **07-feature-engineering** | Feature Selection, Transformation, Encoding | ⭐⭐⭐ Critical |
| **08-unsupervised-learning** | Clustering, PCA, Anomaly Detection | ⭐⭐ Important |
| **15-time-series-analysis** | ARIMA, LSTM, Time Series Forecasting | ⭐⭐ Important |
| **19-sql-database-fundamentals** | SQL, Database Design | ⭐⭐ Important |
| **20-handling-imbalanced-data** | SMOTE, Class Weights, Evaluation Metrics | ⭐⭐ Important |
| **21-model-explainability** | SHAP, LIME, Feature Importance | ⭐⭐⭐ Critical |
| **25-generative-ai-llms** | Days 1–3 only for GenAI lite *(optional)* | ⭐ Optional |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | House Price Prediction, Titanic Survival, Wine Quality | Regression, Classification, EDA |
| **Intermediate** | Customer Churn Prediction, Credit Card Fraud Detection, Time Series Forecasting, Customer Segmentation | Imbalanced Data, Time Series, Clustering |
| **Advanced** | Model Explainability & Interpretability, Causal / A/B write-up on a real metric | SHAP, LIME, Experiment Design |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Required** | [Ethics in ML](ethics_in_ml.md) *(before Module 04)*, [Causal Inference Guide](causal_inference_guide.md), [Experiment Design and A/B Testing](experiment_design_ab_testing.md) |
| **Core Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [Math Formulas](math_formulas.md), [ML Glossary](ml_glossary.md) |
| **Advanced Topics** | [Model Interpretability Guide](model_interpretability.md), [Recommender Systems](recommender_systems.md), [MLFlow Guide](mlflow_comprehensive_guide.md) |
| **Tools** | [Docker Tutorial](docker_tutorial.md), [Git Guide](git_guide.md), [Web Scraping Guide](web_scraping_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md), [Kaggle Competitions](kaggle_competitions.md) |

### Skills Checklist

- Statistical analysis and hypothesis testing
- Causal thinking and experiment design (not only predictive accuracy)
- Machine learning algorithms (supervised and unsupervised)
- Feature engineering and selection
- Model evaluation and optimization
- Time series analysis
- Data visualization and storytelling
- SQL and database querying
- Model explainability (SHAP, LIME)
- Experiment tracking (MLflow)
- Ethics awareness before shipping classifiers
- Business acumen and communication

---

## Machine Learning Engineer

**Role Focus**: Design, build, and deploy ML models to production. Focus on software engineering, MLOps, and scalable ML systems.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Math, Algorithms | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, APIs, Web Development | 2-3 months |
| **SQL & Databases** | 19 | SQL, databases *(Stage 1.5)* | 1-2 months |
| **ML Basics** | 02-05 | ML Concepts, Models, Evaluation | 3-4 months |
| **Advanced ML** | 06-07 | Ensembles, Feature Engineering | 2-3 months |
| **Unsupervised Learning** | 08 | Clustering, Dimensionality Reduction | 1 month |
| **Deep Learning** | 09-10 | Neural Networks, TensorFlow, PyTorch | 2-3 months |
| **System design (core)** | [System Design for Beginners](../system-design/README.md) + [ML System Design Guide](ml_system_design_guide.md) | Latency, storage, APIs, failure modes for ML systems. Include [Chapter 30 · Feature store](../system-design/30-design-realtime-feature-store.md) with Module 14 | 1-2 months |
| **Production** | 13-14 | Model Deployment, MLOps, CI/CD. Feature-store concepts pair with SD Ch. 30 | 3-4 months |
| **Essential Skills** | 20-21 | Imbalanced Data, Explainability | 1-2 months |
| **Total** | **16 modules** + system-design spine | **Complete Path** | **24-36 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, OOP, Algorithms, Math | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas, Flask, FastAPI, Streamlit | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Workflow, Best Practices | ⭐⭐⭐ Critical |
| **03-supervised-learning-regression** | Regression Models, Evaluation | ⭐⭐⭐ Critical |
| **04-supervised-learning-classification** | Classification Models | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Cross-Validation, Hyperparameter Tuning | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Ensemble Techniques | ⭐⭐ Important |
| **07-feature-engineering** | Feature Engineering, Pipelines | ⭐⭐ Important |
| **08-unsupervised-learning** | Clustering, PCA | ⭐ Optional |
| **09-neural-networks-basics** | Neural Networks, Backpropagation | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | TensorFlow, PyTorch | ⭐⭐⭐ Critical |
| **13-model-deployment** | REST APIs, Docker, local or one cloud serve path. Managed platforms (e.g. SageMaker) and online A/B are **Important** when you have a lab. Not a Critical Module 13 gate without hands-on practice | ⭐⭐⭐ Critical *(API + Docker core)* |
| **14-mlops-basics** | DVC, MLflow, CI/CD. Streaming / Spark names appear as a survey. Pair feature-store ideas with [SD Ch. 30](../system-design/30-design-realtime-feature-store.md) | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL, Database Design | ⭐⭐ Important |
| **20-handling-imbalanced-data** | Data Quality, Resampling | ⭐ Optional |
| **21-model-explainability** | SHAP, LIME | ⭐⭐ Important |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | House Price Prediction, Titanic Survival | Model Building, Evaluation |
| **Intermediate** | Customer Churn Prediction, Feature Engineering Mastery | Feature Engineering, Pipelines |
| **Advanced** | End-to-End ML Pipeline, Model Deployment & Serving | Full Pipeline, Deployment, APIs, one cloud demo or local serve |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [DSA for ML Guide](dsa_for_ml_guide.md), [Git Guide](git_guide.md) |
| **System Design (also on Core Path)** | [System Design for Beginners](../system-design/README.md), [ML System Design Guide](ml_system_design_guide.md), [Design a Real-Time Feature Store](../system-design/30-design-realtime-feature-store.md) |
| **MLOps** | [MLFlow Guide](mlflow_comprehensive_guide.md), [Docker Tutorial](docker_tutorial.md), [ML Model Testing](ml_model_testing.md) |
| **Tools** | [Data Validation Guide](data_validation.md), [AutoML Basics](automl_basics.md), [Experiment Design and A/B Testing](experiment_design_ab_testing.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Software engineering (Python, Git, Testing)
- Machine learning algorithms and frameworks
- Model deployment (REST APIs, Docker, local serve or one cloud demo)
- MLOps (MLflow, DVC, CI/CD)
- Data pipelines and ETL basics
- Feature-store concepts (training-serving consistency)
- Cloud platforms (pick one to practice. Do not claim three-cloud mastery from this repo alone)
- Containerization (Docker. Kubernetes as optional survey)
- Monitoring and logging
- A/B testing when you have traffic or a careful offline experiment design
- System design for ML systems

---

## LLM Engineer (Large Language Models)

**Role Focus**: Build, fine-tune, and deploy large language models. Work with transformers, RAG systems, and generative AI applications.

**Honest order:** Eval and RAG before agents. PEFT / fine-tuning after you can measure retrieval quality. MCP and tool use need clear permissions. Do not treat multi-agent demos as the default first project.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Math, Algorithms | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, APIs | 2 months |
| **SQL & Databases** | 19 | SQL *(Stage 1.5)* | 1-2 months |
| **ML Basics** | 02-05 | ML Concepts, Models, Evaluation | 3-4 months |
| **Advanced ML** | 06-07 | Ensembles, Feature Engineering | 1-2 months |
| **Deep Learning** | 09-10 | Neural Networks, PyTorch, TensorFlow | 2-3 months |
| **NLP** | 12 | NLP, Transformers, RAG first, then PEFT / fine-tuning | 4-5 months |
| **Generative AI** | 25 | Prompting, RAG, **eval before agents**, then agents / MCP with least privilege | 1-2 months |
| **Production** | 13-14 | Local serve or **one** cloud demo. Not three-hyperscaler mastery | 2-3 months |
| **Essential Skills** | 21 | Explainability | 0.5-1 month |
| **Total** | **12 modules** | **Complete Path** | **24-36 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Linear Algebra, Statistics | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas, APIs, Flask/FastAPI | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow | ⭐⭐ Important |
| **03-supervised-learning-regression** | Regression Basics | ⭐ Optional |
| **04-supervised-learning-classification** | Classification Basics | ⭐⭐ Important |
| **05-model-evaluation-optimization** | Evaluation Metrics, Hyperparameter Tuning | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Ensemble Basics | ⭐ Optional |
| **07-feature-engineering** | Feature Engineering for Text | ⭐⭐ Important |
| **09-neural-networks-basics** | Neural Networks, Backpropagation | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | PyTorch, TensorFlow | ⭐⭐⭐ Critical |
| **12-natural-language-processing** | Text Preprocessing, Embeddings, Transformers, RAG, then PEFT / fine-tuning | ⭐⭐⭐ Critical |
| **25-generative-ai-llms** | Prompt Engineering, Vector Stores, RAG eval, guardrails, then agents / MCP permissions | ⭐⭐⭐ Critical |
| **13-model-deployment** | REST APIs, Docker, local serve or one cloud demo | ⭐⭐⭐ Critical |
| **14-mlops-basics** | MLflow, Version Control, CI/CD | ⭐⭐ Important |
| **19-sql-database-fundamentals** | SQL, Vector Databases | ⭐⭐ Important |
| **21-model-explainability** | Model Interpretation | ⭐ Optional |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Intermediate** | Sentiment Analysis on Reviews, RAG with retrieval metrics | NLP, Eval, Text Classification |
| **Advanced** | LLM app with RAG + eval harness, optional PEFT, optional agent with scoped tools | RAG, Eval, PEFT, Permissions, Deployment |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [Transformer Fine-Tuning Guide](transformer_fine_tuning_guide.md), [Langchain Guide](langchain_guide.md), [LlamaIndex Guide](llamaindex_guide.md) |
| **Advanced Topics** | [AI Agents Guide](ai_agents_guide.md), [GenAI Production Deployment](genai_production_deployment.md), [MLFlow Guide](mlflow_comprehensive_guide.md) |
| **Tools** | [Docker Tutorial](docker_tutorial.md), [Git Guide](git_guide.md), [Data Validation Guide](data_validation.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Deep learning fundamentals (Neural Networks, Backpropagation)
- NLP techniques (Tokenization, Embeddings, Attention)
- Transformer architectures (BERT, GPT, T5)
- RAG with measurable retrieval / answer quality **before** agents
- PEFT / fine-tuning after a working RAG + eval loop
- Langchain and LlamaIndex when they help. Not required for every project
- Vector databases and embeddings (start with one local option such as FAISS or Chroma)
- Model deployment (local serve or one cloud demo)
- Prompt engineering (zero-shot, few-shot, chain-of-thought)
- Generative configuration (temperature, top-p, top-k)
- LLM evaluation and benchmarking
- MCP / tool use with explicit permissions and least privilege
- Agents only after eval and guardrails. Multi-agent is optional depth

---

## GenAI Solution Architect

**Role Focus**: Design and implement Generative AI solutions, multi-agent systems, and RAG architectures for real products. Lead technical teams and establish GenAI best practices at scale.

**Honest order:** Same as LLM Engineer. Eval and RAG before agents. PEFT after retrieval quality is measurable. MCP tools need permission boundaries. Cloud claims mean one hyperscaler demo or a solid local serve story. Not “all three clouds mastered” from reading alone.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00-01 | Python, Data Fundamentals | 2-3 months |
| **SQL & Databases** | 19 | SQL, databases *(Stage 1.5)* | 1-2 months |
| **ML Basics** | 02-05 | ML Concepts, Evaluation, Optimization | 3-4 months |
| **Deep Learning** | 09-10 | Neural Networks, Frameworks | 2-3 months |
| **NLP & GenAI** | 12, 25 | RAG + eval first, PEFT next, agents / MCP / A2A after permissions and guardrails | 5-6 months |
| **Production** | 13-14 | Deploy + [GenAI Production Deployment](genai_production_deployment.md). One cloud demo or local serve | 3-4 months |
| **Total** | **10 modules** | **Complete Path** | **22-32 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Math, Algorithms | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas, APIs, Flask/FastAPI | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Evaluation, Hyperparameter Tuning | ⭐⭐⭐ Critical |
| **09-neural-networks-basics** | Neural Networks, Backpropagation | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | PyTorch, TensorFlow | ⭐⭐⭐ Critical |
| **12-natural-language-processing** | NLP, Transformers, RAG, then PEFT | ⭐⭐⭐ Critical |
| **25-generative-ai-llms** | Prompting, RAG eval, guardrails, agents, MCP permissions, optional A2A | ⭐⭐⭐ Critical |
| **13-model-deployment** | Deployment, APIs, local serve or one cloud demo | ⭐⭐⭐ Critical |
| **14-mlops-basics** | MLOps, CI/CD, Experiment Tracking | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL, Databases, Vector Stores | ⭐⭐ Important |

### Specialized Skills

**Required expertise (honest bar):**
- RAG architectures with evaluation (retrieval quality, faithfulness, cost)
- Agent and multi-agent designs only after eval and guardrails
- MCP with explicit tool permissions and least privilege
- A2A patterns as optional depth when multi-service agents appear
- Production GenAI on **one** cloud demo or a reproducible local serve path
- Scaling and batching ideas you can explain. Not every hyperscaler SKU
- Cost optimization for LLM inference
- Model serving and orchestration basics

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Intermediate** | Chatbot with RAG + eval report | NLP, Transformers, RAG, Eval |
| **Advanced** | GenAI service with permissions, monitoring, and one deploy path | RAG, Guardrails, MCP, Ops, Cost notes |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [Langchain Guide](langchain_guide.md), [LlamaIndex Guide](llamaindex_guide.md), [AI Agents Guide](ai_agents_guide.md), [GenAI Production Deployment Guide](genai_production_deployment.md) |
| **Specialized** | [Transformer Fine-Tuning Guide](transformer_fine_tuning_guide.md), [MLFlow Comprehensive Guide](mlflow_comprehensive_guide.md), [Model Deployment Cheatsheet](model_deployment_cheatsheet.md) |
| **Tools** | [Docker Tutorial](docker_tutorial.md), [Web Scraping Guide](web_scraping_guide.md), [Git Guide](git_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md), [Stakeholder Communication](stakeholder_communication.md) |

### Skills Checklist

- Generative AI architectures (RAG first. Agents second)
- RAG system design with eval harnesses
- Multi-agent orchestration when the problem needs it
- MCP with permission boundaries
- A2A patterns as optional depth
- Production GenAI deploy (local or one cloud). Not three-cloud mastery by default
- Cost and latency awareness for LLM inference
- Model serving and API development (FastAPI, Flask)
- Containerization (Docker. Kubernetes as optional survey)
- MLOps for GenAI (experiment tracking, model registry, monitoring)
- Vector databases and embeddings
- Prompt engineering and optimization
- PEFT / fine-tuning after RAG quality is measured
- Stakeholder communication and technical leadership practice

---

## Computer Vision Engineer

**Role Focus**: Build and deploy computer vision models for image classification, object detection, segmentation, and image generation.

**Transfer learning first:** Start from pretrained backbones and fine-tune for your dataset. Building YOLO (or similar detectors) from scratch is optional research depth. It is not a Critical pass/fail bar for this path.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Math, Linear Algebra | 2-3 months |
| **Data Fundamentals** | 01 | NumPy, Data Manipulation | 1-2 months |
| **SQL & Databases** | 19 | SQL *(Stage 1.5, optional)* | 0.5-1 month |
| **ML Basics** | 02-05 | ML Concepts, Models, Evaluation | 3-4 months |
| **Advanced ML** | 06-07 | Ensembles, Feature Engineering | 1-2 months |
| **Deep Learning** | 09-10 | Neural Networks, PyTorch, TensorFlow | 2-3 months |
| **Computer Vision** | 11 | CNNs, transfer learning, detection APIs, segmentation, generative survey | 4-6 months |
| **Production** | 13-14 | Model Deployment, MLOps | 2-3 months |
| **Essential Skills** | 21 | Explainability | 0.5-1 month |
| **Total** | **11 modules** | **Complete Path** | **22-34 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Linear Algebra, Statistics | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas, Visualization | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow | ⭐⭐ Important |
| **03-supervised-learning-regression** | Regression Basics | ⭐ Optional |
| **04-supervised-learning-classification** | Classification, Evaluation Metrics | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Cross-Validation, Hyperparameter Tuning | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Ensemble Basics | ⭐ Optional |
| **07-feature-engineering** | Feature Engineering | ⭐ Optional |
| **09-neural-networks-basics** | Neural Networks, Backpropagation | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | PyTorch, TensorFlow, Keras | ⭐⭐⭐ Critical |
| **11-computer-vision** | Image fundamentals, CNNs, pretrained models, transfer learning, detection/segmentation APIs. YOLO-from-scratch and full GAN/diffusion training are optional depth | ⭐⭐⭐ Critical |
| **13-model-deployment** | REST APIs, Docker, Cloud Deployment | ⭐⭐⭐ Critical |
| **14-mlops-basics** | MLflow, Version Control, CI/CD | ⭐⭐ Important |
| **19-sql-database-fundamentals** | SQL Basics | ⭐ Optional |
| **21-model-explainability** | SHAP for Images, Model Interpretation | ⭐⭐ Important |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Intermediate** | Handwritten Digit Recognition (MNIST), Transfer-learned classifier on a small custom set | CNNs, Transfer Learning |
| **Advanced** | Fine-tuned detection or segmentation with pretrained weights, optional generative survey, Model Deployment & Serving | Transfer Learning, Detection APIs, Deployment |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [Math Formulas](math_formulas.md) |
| **System Design** | [System Design for Beginners](../system-design/README.md), [ML System Design Guide](ml_system_design_guide.md) |
| **Tools** | [Docker Tutorial](docker_tutorial.md), [Git Guide](git_guide.md), [MLFlow Guide](mlflow_comprehensive_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Deep learning fundamentals (Neural Networks, CNNs)
- Computer vision architectures via pretrained models (ResNet, EfficientNet, and similar)
- Object detection with pretrained detectors and fine-tuning. From-scratch YOLO is optional
- Image segmentation (transfer-learned or API-based workflows)
- Transfer learning and fine-tuning (**default path**)
- Data augmentation techniques
- Generative vision (GANs / diffusion) as a survey unless your role needs it
- Model deployment and optimization
- OpenCV and image processing
- Model explainability for vision models

---

## AI Engineer (Generalist)

**Role Focus**: Broad expertise across multiple AI domains including ML, NLP, Computer Vision, and Generative AI. Work on end-to-end AI solutions.

### Minimum viable generalist vs electives

| Tier | Modules | What it means |
|------|---------|---------------|
| **Minimum viable generalist** | **00–14** (includes Module **12** NLP) **+ Module 25** | Enough breadth for an honest AI Engineer study path. ML core, CV (11), NLP (12), deploy/MLOps (13–14), GenAI (25) |
| **Electives** | **15**, **22–24** | Time series, RL, GNNs, audio. Useful depth. **Not required** for the AI Engineer title on this roadmap |

Do not treat Modules 22–24 as a gate for calling yourself an AI Engineer here. Add them when your projects or interviews need that domain.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Math, Algorithms | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, APIs, Web Development | 2-3 months |
| **SQL & Databases** | 19 | SQL, databases *(Stage 1.5)* | 1-2 months |
| **ML Basics** | 02-05 | ML Concepts, Models, Evaluation | 3-4 months |
| **Advanced ML** | 06-07 | Ensembles, Feature Engineering | 2-3 months |
| **Unsupervised Learning** | 08 | Clustering, Dimensionality Reduction | 1-2 months |
| **Deep Learning** | 09-10 | Neural Networks, PyTorch, TensorFlow | 2-3 months |
| **Specialized DL** | 11-12 | Computer Vision, NLP | 5-7 months |
| **Generative AI** | 25 | Modern LLMs, RAG, eval before agents | 1-2 months |
| **Production** | 13-14 | Model Deployment, MLOps | 3-4 months |
| **Essential Skills** | 20-21 | Imbalanced Data, Explainability | 1-2 months |
| **Electives** | 15, 22-24 | Time series, RL, GNN, Audio *(optional)* | 2-4 months |
| **Total (minimum viable)** | **~17 modules** *(00–14 + 19 + 20–21 + 25, adjust if you skip optional 20)* | **Generalist core** | **28-42 months** |
| **Total (with electives)** | **up to ~23 modules** | **Full elective map** | **36-54 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Math, Algorithms | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas, APIs, Flask/FastAPI | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts, Workflow | ⭐⭐⭐ Critical |
| **03-supervised-learning-regression** | Regression Models | ⭐⭐⭐ Critical |
| **04-supervised-learning-classification** | Classification Models | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Evaluation, Hyperparameter Tuning | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Ensemble Techniques | ⭐⭐⭐ Critical |
| **07-feature-engineering** | Feature Engineering | ⭐⭐⭐ Critical |
| **08-unsupervised-learning** | Clustering, PCA | ⭐⭐ Important |
| **09-neural-networks-basics** | Neural Networks | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | PyTorch, TensorFlow | ⭐⭐⭐ Critical |
| **11-computer-vision** | CNNs, transfer learning, detection survey | ⭐⭐⭐ Critical |
| **12-natural-language-processing** | NLP, Transformers, Fine-tuning, RAG | ⭐⭐⭐ Critical |
| **25-generative-ai-llms** | Modern LLMs, Prompt Engineering, RAG Systems, eval, then agents | ⭐⭐⭐ Critical |
| **13-model-deployment** | Deployment, APIs, Cloud | ⭐⭐⭐ Critical |
| **14-mlops-basics** | MLOps, CI/CD, MLflow | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL, Databases | ⭐⭐ Important |
| **20-handling-imbalanced-data** | Data Quality | ⭐ Optional |
| **21-model-explainability** | SHAP, LIME | ⭐⭐ Important |
| **15-time-series-analysis** | Time Series, Forecasting | ⭐ Elective |
| **22-reinforcement-learning** | RL, Q-Learning, DQN, Policy Gradients | ⭐ Elective |
| **23-graph-neural-networks** | GNNs, GCNs, GATs | ⭐ Elective |
| **24-audio-speech-processing** | ASR, TTS, Audio Classification | ⭐ Elective |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | House Price Prediction, Titanic Survival | Regression, Classification |
| **Intermediate** | Handwritten Digit Recognition, Sentiment Analysis | CNNs, NLP |
| **Advanced** | Image Classification, LLM app with RAG + eval, End-to-End ML Pipeline, Model Deployment | Multi-domain practice, GenAI, Ops |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [DSA for ML Guide](dsa_for_ml_guide.md), [Math Formulas](math_formulas.md) |
| **System Design** | [System Design for Beginners](../system-design/README.md), [ML System Design Guide](ml_system_design_guide.md) |
| **Specialized** | [Transformer Fine-Tuning Guide](transformer_fine_tuning_guide.md), [Langchain Guide](langchain_guide.md), [LlamaIndex Guide](llamaindex_guide.md), [AI Agents Guide](ai_agents_guide.md) |
| **MLOps** | [MLFlow Guide](mlflow_comprehensive_guide.md), [Docker Tutorial](docker_tutorial.md), [ML Model Testing](ml_model_testing.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Machine learning (supervised, unsupervised). RL only if you took the elective
- Deep learning (CNNs, Transformers)
- Computer vision (transfer learning and detection survey)
- Natural language processing (NLP, Transformers, RAG)
- Generative AI (prompting, RAG, eval. Agents when needed)
- Elective depth as chosen (time series, RL, GNN, audio)
- Model deployment and MLOps
- Software engineering and system design basics
- Frameworks you actually practiced (PyTorch and/or TensorFlow, Hugging Face)
- One cloud demo or local serve. Not automatic multi-cloud mastery
- Experimentation and honest evaluation

---

## Data Engineer

**Role Focus**: Design, build, and maintain data pipelines, data warehouses, and data infrastructure. Focus on data quality, ETL processes, and scalable data systems.

**Dedicated chapter:** [Data Engineering Path](data_engineering_path.md) (ETL, quality, warehouse modeling with Module 19, streaming survey with honest limits).

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Algorithms | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, ETL, APIs | 3-4 months |
| **ML Basics** | 02 | ML Concepts (for understanding). Modules 03–05 optional | 0.5-1 month |
| **Databases & warehouse** | 19 | SQL, NoSQL survey, warehouse / dimensional modeling notes | 3-4 months |
| **Quality** | 20 + [data_validation.md](data_validation.md) | Schemas, checks, drift awareness | 1-2 months |
| **Production survey** | 13-14 | Docker / APIs. Module 14 streaming names (Kafka, Spark) are a **survey**, not mastery | 2-3 months |
| **System design** | [System Design for Beginners](../system-design/README.md) | Pipelines, queues, storage tradeoffs | 1-2 months |
| **Total** | **~6–8 modules** + DE path chapter | **Complete Path** | **18-28 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, OOP, Algorithms | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Pandas, NumPy, ETL, APIs, Web Scraping | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts (understanding) | ⭐ Optional |
| **13-model-deployment** | Docker, Cloud Deployment, APIs | ⭐⭐⭐ Critical |
| **14-mlops-basics** | DVC, MLflow, CI/CD. Kafka / Spark appear as a **tool survey**. Do not claim production Spark/Kafka mastery from Module 14 alone | ⭐⭐ Important *(survey)* |
| **19-sql-database-fundamentals** | SQL, NoSQL survey, database design. Use this module for warehouse modeling habits (facts, dimensions, grain) | ⭐⭐⭐ Critical |
| **20-handling-imbalanced-data** | Data Quality, Validation | ⭐⭐ Important |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | Batch ETL into a warehouse-style schema + quality checks | ETL, SQL, Validation |
| **Advanced** | Scheduled pipeline with tests, docs, and one streaming **spike** (not a full Kafka platform) | Pipelines, Quality, Ops honesty |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Required path** | [Data Engineering Path](data_engineering_path.md), [Data Validation Guide](data_validation.md), [System Design for Beginners](../system-design/README.md) |
| **Core Skills** | [Data Science Cheatsheet](data_science_cheatsheet.md), [Git Guide](git_guide.md), [DSA for ML Guide](dsa_for_ml_guide.md) |
| **System Design** | [ML System Design Guide](ml_system_design_guide.md) *(when pipelines serve ML)* |
| **Tools** | [Docker Tutorial](docker_tutorial.md), [Web Scraping Guide](web_scraping_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Python programming and scripting
- SQL and database design
- NoSQL databases as a survey (pick what your project needs)
- ETL/ELT pipelines you can test and schedule
- Data warehousing concepts with Module 19
- Spark / Kafka **awareness** from Module 14. Mastery needs separate practice jobs
- One cloud practice path. Not automatic multi-cloud mastery
- Containerization (Docker. Kubernetes optional survey)
- Data quality and validation
- Data modeling and schema design

---

## MLOps Engineer

**Role Focus**: Specialize in deploying, monitoring, and maintaining ML models in production. Focus on CI/CD, model versioning, and ML infrastructure.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Algorithms | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, APIs | 2 months |
| **SQL & Databases** | 19 | SQL *(Stage 1.5)* | 1-2 months |
| **ML Basics** | 02-05 | ML Concepts, Models | 3-4 months |
| **Advanced ML** | 06-07 | Ensembles, Feature Engineering | 1-2 months |
| **Deep Learning** | 09-10 | Neural Networks, Frameworks | 2-3 months |
| **NLP & GenAI** | 12, 25 | NLP, Transformers, RAG, eval. Agents optional | 2-3 months |
| **Production** | 13-14 | Deployment, MLOps, CI/CD. Streaming tools as survey | 4-5 months |
| **LLMOps pointer** | [genai_production_deployment.md](genai_production_deployment.md) | Serving, cost, eval, and ops habits for LLM apps | 0.5-1 month |
| **Essential Skills** | 20-21 | Imbalanced Data, Explainability | 1-2 months |
| **Total** | **14 modules** | **Complete Path** | **24-36 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, OOP, Algorithms | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Python, APIs, Flask/FastAPI | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts | ⭐⭐ Important |
| **03-05** | ML Models, Evaluation | ⭐⭐ Important |
| **09-10** | Deep Learning Basics | ⭐⭐ Important |
| **12-natural-language-processing** | NLP, Transformers | ⭐⭐ Important |
| **25-generative-ai-llms** | Modern LLMs, RAG, eval, GenAI deploy habits | ⭐⭐⭐ Critical |
| **13-model-deployment** | REST APIs, Docker, local or one cloud path. Managed platforms and online A/B are Important with a lab | ⭐⭐⭐ Critical *(API + Docker core)* |
| **14-mlops-basics** | DVC, MLflow, CI/CD, feature-store ideas with [SD Ch. 30](../system-design/30-design-realtime-feature-store.md). Kafka / Spark as survey | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL, Databases | ⭐⭐ Important |
| **20-handling-imbalanced-data** | Data Validation | ⭐ Optional |
| **21-model-explainability** | Model Monitoring | ⭐ Optional |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Advanced** | End-to-End ML Pipeline, Model Deployment & Serving, optional LLM serve with cost/eval notes | CI/CD, Monitoring, LLMOps pointer |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Core Skills** | [MLFlow Guide](mlflow_comprehensive_guide.md), [Docker Tutorial](docker_tutorial.md), [ML Model Testing](ml_model_testing.md) |
| **LLMOps** | [GenAI Production Deployment Guide](genai_production_deployment.md) |
| **System Design** | [System Design for Beginners](../system-design/README.md), [ML System Design Guide](ml_system_design_guide.md) |
| **Tools** | [Data Validation Guide](data_validation.md), [Git Guide](git_guide.md), [Experiment Design and A/B Testing](experiment_design_ab_testing.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Software engineering (Python, Git, Testing)
- ML concepts and model understanding
- Containerization (Docker). **Kubernetes / IaC (Terraform, CloudFormation) as optional survey**, not a Critical checklist item from this repo alone
- CI/CD pipelines for ML
- Model versioning (MLflow, DVC)
- Model monitoring and observability
- One cloud practice path
- Distributed systems names (Kafka, Spark) as survey awareness
- Model serving and APIs
- A/B testing when you have a careful design or a lab
- GenAI / LLMOps habits via [genai_production_deployment.md](genai_production_deployment.md) (RAG serve, cost, eval)
- LLM monitoring and cost optimization practice notes

---

## Research Scientist

**Role Focus**: Novel methods, careful experiments, and clear write-ups, plus the depth to read papers, reproduce results, and contribute new ideas. Conduct research, develop algorithms, and push boundaries of ML/AI. Strong theory and coding both matter.

**Practice goal:** Reproduce at least one paper result (or a clear ablation) with a written method note. Checklist completion alone is not a research portfolio.

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python, Advanced Math, Algorithms | 3-4 months |
| **Data Fundamentals** | 01 | Data Manipulation | 1-2 months |
| **SQL & Databases** | 19 | SQL basics *(Stage 1.5, optional)* | 0.5-1 month |
| **Causal + experiment design** | [causal_inference_guide.md](causal_inference_guide.md) + [experiment_design_ab_testing.md](experiment_design_ab_testing.md) | Identification, validity, metrics, power. On the path, not only in the resource table | 1-2 months |
| **ML Basics** | 02-05 | ML Theory, Algorithms, Evaluation | 4-5 months |
| **Advanced ML** | 06-07 | Advanced Algorithms, Theory | 2-3 months |
| **Unsupervised Learning** | 08 | Advanced Clustering, Dimensionality Reduction | 2-3 months |
| **Deep Learning** | 09-10 | Neural Networks, Frameworks | 3-4 months |
| **Specialized DL** | 11-12 | Computer Vision, NLP *(pick depth by research area)* | 5-7 months |
| **Time series (branch)** | 15 | Forecasting, temporal evaluation (optional but common in applied research) | 0.5-1 month |
| **Generative AI** | 25 | Modern LLMs, RAG, eval. Agents / RLHF as needed for your topic | 1-2 months |
| **Advanced Specialized** | 22-24 | RL, GNN, Audio *(match your research area. Not all required)* | 2-3 months |
| **Essential Skills** | 21 | Explainability | 0.5-1 month |
| **Total** | **varies by specialization** | **Complete Path** | **36-54 months** |

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, Advanced Math, Linear Algebra, Statistics, Calculus | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | NumPy, Pandas | ⭐⭐ Important |
| **02-introduction-to-ml** | ML Theory, Concepts | ⭐⭐⭐ Critical |
| **03-supervised-learning-regression** | Regression Theory, Statistical Analysis | ⭐⭐⭐ Critical |
| **04-supervised-learning-classification** | Classification Theory, Algorithms | ⭐⭐⭐ Critical |
| **05-model-evaluation-optimization** | Evaluation Theory, Optimization | ⭐⭐⭐ Critical |
| **06-ensemble-methods** | Ensemble Theory, Advanced Techniques | ⭐⭐⭐ Critical |
| **07-feature-engineering** | Feature Engineering Theory | ⭐⭐ Important |
| **08-unsupervised-learning** | Clustering Theory, Dimensionality Reduction | ⭐⭐⭐ Critical |
| **09-neural-networks-basics** | Neural Network Theory, Backpropagation | ⭐⭐⭐ Critical |
| **10-deep-learning-frameworks** | PyTorch, TensorFlow | ⭐⭐⭐ Critical |
| **11-computer-vision** | CV Theory, Architectures | ⭐⭐ Research-area |
| **12-natural-language-processing** | NLP Theory, Transformers | ⭐⭐ Research-area |
| **25-generative-ai-llms** | Modern LLMs, eval, optional RLHF | ⭐⭐ Research-area |
| **15-time-series-analysis** | Time Series Theory | ⭐⭐ Important when relevant |
| **22-reinforcement-learning** | RL Theory, MDPs, Deep RL | ⭐⭐ Research-area |
| **23-graph-neural-networks** | GNN Theory, Graph Algorithms | ⭐⭐ Research-area |
| **24-audio-speech-processing** | Audio Processing Theory, ASR, TTS | ⭐⭐ Research-area |
| **19-sql-database-fundamentals** | SQL Basics | ⭐ Optional |
| **21-model-explainability** | Explainability Theory | ⭐⭐ Important |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Intermediate** | Feature Engineering Mastery, Ensemble Methods Comparison, Causal / experiment write-up | Advanced Techniques, Validity |
| **Advanced** | **Paper reproduction** (code + metrics + failure notes), optional novel ablation | Research, Experimentation, Writing |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Required on path** | [Causal Inference Guide](causal_inference_guide.md), [Experiment Design and A/B Testing](experiment_design_ab_testing.md) |
| **Core Skills** | [Math Formulas](math_formulas.md), [ML Glossary](ml_glossary.md), [Reinforcement Learning](reinforcement_learning.md) |
| **Advanced Topics** | [Model Interpretability Guide](model_interpretability.md), [Recommender Systems](recommender_systems.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Open Source Contribution](open_source_contribution.md) |

### Skills Checklist

- Strong mathematical foundation (Linear Algebra, Statistics, Calculus, Optimization)
- Deep understanding of ML theory and algorithms
- Causal inference and experimental design on the learning path
- Paper reading and **reproduction** practice
- Deep learning architectures and theory
- Domain modules matched to your research area (not every specialty by default)
- Generative AI depth when your topic needs it
- Clear write-ups of methods and limits
- Programming (Python, PyTorch, TensorFlow)
- Hypothesis testing and honest negative results
- Critical thinking and problem-solving

---

## Business Intelligence Analyst

**Role Focus**: Create dashboards, reports, and visualizations to support business decisions. Focus on data visualization, reporting, and business metrics.

**OLAP / KPI storytelling:** Learn grain, facts, dimensions, and metric definitions in **Module 19**. Pair KPI experiments and honest uplift language with [Experiment Design and A/B Testing](experiment_design_ab_testing.md).

### Core Learning Path

| Phase | Modules | Focus Areas | Time Estimate |
|-------|---------|-------------|---------------|
| **Foundation** | 00 | Python Basics, Statistics | 2-3 months |
| **Data Fundamentals** | 01 | Data Manipulation, Visualization, Dashboards | 3-4 months |
| **SQL & Databases** | 19 | SQL, OLAP/OLTP, KPI / warehouse storytelling | 2-3 months |
| **ML literacy (optional)** | 02 | What models are for *(optional)*. Modules 03–05 optional | 0.5-1 month |
| **Essential Skills** | 20-21 | Data Quality, Explainability | 1-2 months |
| **Project add-ons** | 08 and/or 15 | Segmentation / forecasting *(only if projects need them)* | 1-2 months |
| **Total (core)** | **~5 modules** | **Core path without 02 / 08 / 15** | **12-20 months** |

**Optional Module 02:** Same note as Data Analyst. Useful literacy. Not required for every BI role.

**Modules 08 and 15:** Required only for segmentation or forecasting projects.

### Essential Modules

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python Basics, Statistics | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Pandas, Visualization (Matplotlib, Seaborn, Plotly), Streamlit. Tableau / Power BI are workplace tools. Not a Module 01 Critical gate | ⭐⭐⭐ Critical |
| **02-introduction-to-ml** | ML Concepts (understanding) | ⭐ Optional |
| **03-05** | ML Basics (understanding) | ⭐ Optional |
| **19-sql-database-fundamentals** | SQL, OLAP/OLTP, Database Design, KPI grain | ⭐⭐⭐ Critical |
| **20-handling-imbalanced-data** | Data Quality | ⭐⭐ Important |
| **21-model-explainability** | Explainability for Business | ⭐⭐ Important |
| **08 / 15** | Clustering / forecasting | ⭐ Project-only |

### Recommended Projects

| Level | Projects | Skills Practiced |
|-------|----------|------------------|
| **Beginner** | Customer Data Dashboard with Streamlit or Power BI | Dashboard Creation, Visualization |
| **Intermediate** | KPI scorecard with Module 19 modeling, optional segmentation *(08)* or forecast *(15)* | Business Analysis, Metric Honesty |

### Essential Resources

| Resource Type | Files |
|--------------|-------|
| **Required** | [Experiment Design and A/B Testing](experiment_design_ab_testing.md), [Stakeholder Communication](stakeholder_communication.md) |
| **Core Skills** | [Excel Data Analysis Guide](excel_data_analysis_guide.md), [Power BI Guide](powerbi_guide.md), [Data Science Cheatsheet](data_science_cheatsheet.md) |
| **Tools** | [Web Scraping Guide](web_scraping_guide.md), [Git Guide](git_guide.md) |
| **Career** | [Career & Portfolio Guide](career_portfolio.md), [Interview Prep](interview_prep.md) |

### Skills Checklist

- Data visualization in Python and/or one BI tool you choose
- SQL and OLAP-style querying (Module 19)
- Excel for data analysis
- Dashboard creation and design
- Business metrics and KPIs with clear definitions
- Experiment / A/B awareness for metric claims
- Report writing and storytelling
- Statistical analysis basics
- Data cleaning and preprocessing
- Stakeholder communication
- Business acumen

---

## Backend Engineer

**Role Focus**: Design, build, and operate APIs and services that clients and other systems depend on. You care about request lifecycle, data integrity, security, observability, and graceful failure. This path is **language-agnostic** at the concept layer and pairs with hands-on Node/Postgres lessons in the Full-Stack Track.

**Sync note:** Keep Module 19 and System Design chapters aligned with the Backend roadmap. ML modules 13–14 stay optional unless you serve models.

**Start here:** [Backend Engineer Roadmap](backend_engineering_roadmap.md)

### Core learning path

| Phase | Focus | Primary resources |
|-------|--------|-------------------|
| **0–1** | HTTP, routing, serialization | [Backend roadmap](backend_engineering_roadmap.md#phase-1-the-request-path), [System Design HTTP](../system-design/06-http.md), [Phase B](full_stack_track/phase-b-node-apis.md) |
| **2–3** | Auth, REST, layering, Postgres | [API Design](../system-design/09-api-design.md), [Phase C](full_stack_track/phase-c-sql-databases.md), [Module 19](../19-sql-database-fundamentals/README.md) |
| **4–5** | Cache, queues, search | [Caching](../system-design/10-caching.md), [Message Queues](../system-design/19-message-queues.md), [Phase E](full_stack_track/phase-e-backend-advanced.md) |
| **6–7** | Ops, security, scale, concurrency | [Backend roadmap Phases 5–7](backend_engineering_roadmap.md#phase-5-reliability-and-operations), [System Design scaling](../system-design/12-proxies-and-load-balancing.md) |

### Essential modules (this repository)

| Module | Topics | Priority |
|--------|--------|----------|
| **19-sql-database-fundamentals** | SQL, joins, relational design | ⭐⭐⭐ Critical |
| **13-model-deployment** | FastAPI, serving, API patterns for ML | ⭐⭐ Important (if you serve models) |
| **14-mlops-basics** | Monitoring, deployment pipelines | ⭐⭐ Important (as you ship) |

### Recommended portfolio builds

| Level | Build | Why it matters |
|-------|--------|----------------|
| **Starter** | REST API + Postgres + auth | Proves request path and persistence |
| **Intermediate** | Cache + background jobs + structured logs | Mirrors real team services |
| **Advanced** | Search + graceful shutdown + load test notes | Shows production thinking |

### Essential resources (in this repo)

| Area | Files |
|------|-------|
| **Roadmap** | [Backend Engineer Roadmap](backend_engineering_roadmap.md) |
| **Architecture** | [System Design for Beginners](../system-design/README.md) |
| **Hands-on** | [Full-Stack Track Phases B, C, E](full_stack_track/README.md) |
| **ML APIs** | [Model Deployment](../13-model-deployment/README.md), [ML System Design Guide](ml_system_design_guide.md) |

### Skills checklist

- HTTP methods, status codes, and headers
- REST resource design and OpenAPI contracts
- Postgres schema, indexes, and transactions
- Authn/authz (sessions or JWT + RBAC)
- Caching and message queues
- Structured logging and basic metrics
- Security baseline (OWASP API Top 10 awareness)
- IO vs CPU concurrency model for your stack

---

## Full-Stack AI Engineer

**Role Focus**: Ship AI-enabled products end to end: typed application code, APIs, databases, frontend, containers, and safe LLM features (RAG, streaming, evaluation). You still lean on this repo for **ML literacy** and **GenAI fundamentals**. The rest is a structured companion path with external learning and portfolio builds.

**Sync note:** Use the same GenAI honesty as LLM / GenAI Architect paths. Eval before agents. One deploy story. Completing the blueprint does not guarantee a hire.

**Important naming note**: In the [Full-Stack AI Engineer Blueprint](full_stack_ai_engineer_roadmap.md), labels like **Module 01** are **topic blocks inside that guide**, not the same thing as this repository’s folder **`01-python-for-data-science`**. Repo modules stay **00–25**. The blueprint’s phases **A–H** describe software and product skills around them.

### Core learning path (this repo + companion blueprint)

| Track | What to use | Focus | Time (indicative) |
|-------|-------------|-------|-------------------|
| **ML & data core (in-repo)** | 00, 01, 19, 25 (and 13–14 as you advance) | Python stack, SQL, production/MLOps touchpoints, LLMs and GenAI | Overlap with main roadmap; treat as parallel or prerequisite slices |
| **Product engineering (companion)** | [Full-Stack AI Engineer Blueprint](full_stack_ai_engineer_roadmap.md) Phases **A–H** and [in-repo lesson chapters](full_stack_track/README.md) | TypeScript, Node/Express, Postgres/Prisma, Next.js, Docker/Nginx, AI integration | **18–30 months** at ~8–12 hrs/week on ML plus parallel blueprint hours |

### Essential modules (this repository)

| Module | Topics | Priority |
|--------|--------|----------|
| **00-prerequisites** | Python, math, stats | ⭐⭐⭐ Critical |
| **01-python-for-data-science** | Data stack, EDA, APIs where relevant | ⭐⭐⭐ Critical |
| **19-sql-database-fundamentals** | SQL, relational design | ⭐⭐⭐ Critical |
| **25-generative-ai-llms** | LLMs, prompting, app-facing GenAI | ⭐⭐⭐ Critical |
| **13-14** | Deployment, MLOps, APIs | ⭐⭐ Important (as you ship) |

### Recommended portfolio builds

| Level | Build | Why it matters |
|-------|--------|----------------|
| **Starter** | Typed CLI or small API + Postgres | Proves fundamentals before frameworks |
| **Intermediate** | Auth + RBAC + CRUD + migrations | Mirrors real product backends |
| **Advanced** | Full-stack app + RAG or streaming assistant + observability | Strong portfolio evidence for AI-in-product work |

### Essential resources (in this repo)

| Area | Files |
|------|-------|
| **Shipping & ops** | [Git Guide](git_guide.md), [Docker Tutorial](docker_tutorial.md), [Model Deployment Cheatsheet](model_deployment_cheatsheet.md), [MLOps Cheatsheet](mlops_cheatsheet.md) |
| **GenAI & RAG** | [Generative AI Comprehensive Guide](generative_ai_comprehensive_guide.md), [RAG Comprehensive Guide](rag_comprehensive_guide.md), [Langchain Guide](langchain_guide.md) |
| **Full-stack lessons (in-repo)** | [Full-stack track index](full_stack_track/README.md) |
| **System design** | [ML System Design Guide](ml_system_design_guide.md), [GenAI Production Deployment Guide](genai_production_deployment.md) |

**Full phase-by-phase plan and free external links**: [Full-Stack AI Engineer Blueprint](full_stack_ai_engineer_roadmap.md)

### Skills checklist

- TypeScript for application and API layers
- REST design, validation, auth (sessions/JWT), and error handling
- Relational modeling, SQL, migrations, and ORM basics
- Frontend architecture (e.g. Next.js) and secure client/server boundaries
- Containers, reverse proxy, and environment/config hygiene
- LLM integration: prompts, structured output, streaming, RAG, basic evaluation and cost awareness

---

## Quick Reference: Role Comparison

| Role | Primary Focus | Key Skills | Typical Projects |
|------|--------------|-----------|-----------------|
| **Data Analyst** | Insights & Reports | SQL, Visualization, Excel, ethics/comms | Dashboards, Reports |
| **Data Scientist** | Predictive Models | ML, Statistics, causal/experiments | Classification, Forecasting |
| **ML Engineer** | Production ML | SE, MLOps, system design | Model Deployment, APIs |
| **LLM Engineer** | Language Models | Transformers, RAG, eval-first | RAG apps, measured LLM features |
| **GenAI Solution Architect** | Production GenAI | RAG, eval, MCP permissions, one cloud/local | GenAI services with ops notes |
| **CV Engineer** | Image Processing | Transfer learning, CNNs | Classification, Detection |
| **AI Engineer** | Generalist AI | 00–14 + 12 + 25 core. 15/22–24 electives | Multi-domain Projects |
| **Data Engineer** | Data Infrastructure | ETL, quality, warehouse, streaming survey | Data Pipelines, Warehouses |
| **MLOps Engineer** | ML Operations | CI/CD, monitoring, LLMOps pointer | ML Infrastructure, CI/CD |
| **Research Scientist** | Novel methods, experiments, write-ups | Theory, causal, paper reproduction | Reproductions, Papers |
| **BI Analyst** | Business Intelligence | Module 19 KPIs, visualization, experiments | Dashboards, Business Reports |
| **Full-Stack AI Engineer** | End-to-end AI products | TypeScript, APIs, SQL, Next.js, LLM/RAG | Auth + data app + AI feature in production |
| **Backend Engineer** | APIs and services | HTTP, Postgres, cache, queues, security | REST API + auth + observability in production |

---

## How to Use This Guide

1. **Identify Your Target Role**: Choose the role that aligns with your career goals.
2. **Follow the Learning Path**: Complete modules in the recommended order.
3. **Build Projects**: Work on projects relevant to your role to build a portfolio.
4. **Use Resources**: Refer to the essential resources for each role.
5. **Track Progress**: Use the skills checklist to track your learning progress.
6. **Customize**: Adjust the path based on your background and goals.

---

## Additional Tips

- **Start Broad, Then Specialize**: Begin with foundational modules, then focus on your area of interest.
- **Build a Portfolio**: Complete projects relevant to your target role.
- **Practice Regularly**: Consistent practice is key to mastery.
- **Join Communities**: Engage with others in your field.
- **Stay Updated**: ML/AI is rapidly evolving. Keep learning new techniques.
- **Network**: Connect with professionals in your target role.

---

**Try next:** Pick one role row above. Schedule the next two modules on your calendar this week.
