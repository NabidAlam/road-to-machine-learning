# Road to ML: From Zero to Hero

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/NabidAlam/road-to-machine-learning?style=social)](https://github.com/NabidAlam/road-to-machine-learning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NabidAlam/road-to-machine-learning?style=social)](https://github.com/NabidAlam/road-to-machine-learning/network/members)
[![GitHub issues](https://img.shields.io/github/issues/NabidAlam/road-to-machine-learning)](https://github.com/NabidAlam/road-to-machine-learning/issues)
[![GitHub license](https://img.shields.io/github/license/NabidAlam/road-to-machine-learning)](https://github.com/NabidAlam/road-to-machine-learning/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**A comprehensive, step-by-step guide to learning Machine Learning from absolute basics to advanced topics**

*Perfect for beginners • 21 Learning Modules • 18 Real-World Projects • Production-Ready Skills*

[Get Started](#getting-started) • [Learning Path](#learning-path) • [Projects](#phase-8-projects) • [Contribute](#contributing)

</div>

---

## Why This Repository?

- **Zero to Hero Path**: Complete learning journey from basics to advanced ML
- **Beginner-Friendly**: No prior experience needed - we cover everything
- **Hands-On Learning**: 18 practical projects to build your portfolio
- **Production Focus**: Learn deployment, MLOps, and real-world skills
- **Well-Organized**: Logical progression with clear learning objectives
- **Community-Driven**: Open source, contributions welcome!

**Perfect for**: Students, career switchers, self-learners, and anyone wanting to master ML

## Table of Contents

- [Overview](#overview)
- [Learning Path](#learning-path)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Resources](#resources)
- [Contributing](#contributing)

## Overview

This repository provides a structured learning path for machine learning, organized in a logical progression from fundamentals to advanced topics. Each module includes:

- **Clear explanations** of concepts
- **Hands-on exercises** with solutions
- **Practical projects** to reinforce learning
- **Additional resources** for deeper understanding

## Learning Path

### Phase 0: Foundation (Prerequisites)
**Goal**: Build the mathematical and programming foundation needed for ML

- **00-prerequisites**
  - Python Basics (Variables, Data Types, Control Flow, Functions, OOP)
  - Time Complexity & Algorithm Efficiency (Big O notation)
  - Iterators & Generators (Memory-efficient data processing)
  - Essential Mathematics (Linear Algebra, Statistics, Calculus basics)
  - Environment Setup (Python, Jupyter, Virtual Environments)

### Phase 1: Data Fundamentals
**Goal**: Master data manipulation and visualization

- **01-python-for-data-science**
  - NumPy: Arrays, operations, broadcasting
  - Pandas: DataFrames, data manipulation, cleaning
  - Matplotlib & Seaborn: Data visualization
  - Plotly & Dash: Interactive visualizations and web applications
  - Streamlit: Building interactive dashboards and ML applications
  - EDA, Data Sources, Regular Expressions, Advanced Data Wrangling

### Phase 2: Machine Learning Basics
**Goal**: Understand core ML concepts and algorithms

- **02-introduction-to-ml**
  - What is Machine Learning?
  - Types of ML (Supervised, Unsupervised, Reinforcement)
  - ML Workflow and Best Practices

- **03-supervised-learning-regression**
  - Linear Regression
  - Polynomial Regression
  - Regularization (Ridge, Lasso)
  - Evaluation Metrics (MSE, RMSE, MAE, R²)
  - Statistical Regression Analysis (statsmodels: TSS, RSS, ESS, F-statistic, p-values, confidence intervals)

- **04-supervised-learning-classification**
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Evaluation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

- **05-model-evaluation-optimization**
  - Train/Validation/Test Split
  - Cross-Validation (K-Fold, Stratified)
  - Hyperparameter Tuning (Grid Search, Random Search)
  - Bias-Variance Tradeoff
  - Overfitting and Underfitting

### Phase 3: Advanced Supervised Learning
**Goal**: Explore ensemble methods and advanced techniques

- **06-ensemble-methods**
  - Bagging (Bootstrap Aggregating)
  - Boosting (AdaBoost, Gradient Boosting, XGBoost)
  - Stacking
  - Voting Classifiers

- **07-feature-engineering**
  - Feature Selection
  - Feature Transformation
  - Handling Categorical Variables (One-Hot, Label, Target, Frequency, WOE Encoding)
  - Feature Scaling and Normalization
  - Dimensionality Reduction (PCA)
  - Advanced Discretization (Decision Tree-based binning, custom strategies)

### Phase 4: Unsupervised Learning
**Goal**: Learn to work with unlabeled data

- **08-unsupervised-learning**
  - Clustering (K-Means, Hierarchical, DBSCAN)
  - Dimensionality Reduction (PCA, t-SNE)
  - Anomaly Detection
  - Association Rules

### Phase 5: Deep Learning Fundamentals
**Goal**: Introduction to neural networks and deep learning

- **09-neural-networks-basics**
  - Perceptron
  - Multi-Layer Perceptron (MLP)
  - Activation Functions
  - Backpropagation
  - Gradient Descent Variants

- **10-deep-learning-frameworks**
  - TensorFlow/Keras Basics
  - PyTorch Basics
  - Building and Training Neural Networks
  - Model Saving and Loading

### Phase 6: Specialized Deep Learning
**Goal**: Master domain-specific deep learning applications

- **11-computer-vision**
  - Convolutional Neural Networks (CNNs)
  - Image Classification
  - Transfer Learning
  - Data Augmentation
  - Object Detection Basics

- **12-natural-language-processing**
  - Text Preprocessing
  - Word Embeddings (Word2Vec, GloVe)
  - Recurrent Neural Networks (RNNs)
  - Long Short-Term Memory (LSTM)
  - Transformers Basics
  - Sentiment Analysis

- **15-time-series-analysis**
  - Time Series Fundamentals (Trend, Seasonality, Stationarity)
  - Statistical Methods (ARIMA, SARIMA, Exponential Smoothing)
  - Deep Learning for Time Series (LSTM, GRU, Transformers)
  - Feature Engineering for Time Series
  - Evaluation and Validation

### Phase 6.5: Essential Data Science Skills
**Goal**: Master critical skills for real-world ML applications

- **19-sql-database-fundamentals**
  - Database Fundamentals and SQL Basics
  - SQL DDL and DML Commands
  - SQL Joins, Subqueries, and Window Functions
  - Advanced SQL Topics (CTEs, Window Functions, Stored Procedures)
  - Data Cleaning with SQL
  - SQL with Python Integration

- **20-handling-imbalanced-data**
  - Understanding Imbalanced Data Problems
  - Resampling Techniques (SMOTE, ADASYN, Undersampling, Combined)
  - Algorithm-Level Solutions (Class Weights, Threshold Tuning, Cost-Sensitive Learning)
  - Appropriate Evaluation Metrics (PR-AUC, F1-Score)
  - Complete Workflow Examples
  - Best Practices and Common Pitfalls

- **21-model-explainability**
  - Feature Importance Methods (Tree-based, Permutation)
  - SHAP (SHapley Additive exPlanations) - Tree, Kernel, and Advanced
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Partial Dependence Plots (PDP) and ICE Plots
  - Model Interpretation Best Practices
  - Regulatory Compliance and Ethical AI

### Phase 7: Production & Deployment
**Goal**: Learn to deploy ML models in production

- **13-model-deployment**
  - Model Serialization
  - REST APIs with Flask/FastAPI
  - Docker for ML
  - Cloud Deployment (AWS, GCP, Azure)
  - Model Monitoring

- **14-mlops-basics**
  - Version Control for ML (DVC, MLflow)
  - CI/CD for ML
  - Experiment Tracking
  - Model Registry

### Phase 8: Projects
**Goal**: Apply knowledge through real-world projects

- **16-projects-beginner** (5 projects)
  - House Price Prediction ✓ (Available)
  - Iris Flower Classification ✓ (Available)
  - Titanic Survival Prediction ✓ (Available)
  - Spam Email Detection ✓ (Available)
  - Wine Quality Prediction ✓ (Available)

- **17-projects-intermediate** (6 projects)
  - Handwritten Digit Recognition (MNIST) ✓ (Available - README)
  - Customer Churn Prediction ✓ (Available - README)
  - Movie Recommendation System ✓ (Available - README)
  - Credit Card Fraud Detection ✓ (Available - README)
  - Customer Segmentation ✓ (Available - README)
  - Time Series Forecasting ✓ (Available - README)

- **18-projects-advanced** (7 projects)
  - Image Classification (CIFAR-10) ✓ (Available - README)
  - Sentiment Analysis on Reviews ✓ (Available - README)
  - Time Series Forecasting (Advanced) ✓ (Available - README)
  - Chatbot Development ✓ (Available - README)
  - Object Detection ✓ (Available - README)
  - End-to-End ML Pipeline ✓ (Available - README)
  - Generative Model (GAN/VAE) ✓ (Available - README)

## Prerequisites

Before starting, you should have:
- Basic computer literacy
- Willingness to learn and practice
- A computer with internet connection

**Note**: No prior programming or math experience required! We'll cover everything you need.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/NabidAlam/road-to-machine-learning.git
cd road-to-machine-learning
```

### 2. Set Up Your Environment

#### Option A: Using Anaconda (Recommended for Beginners)

```bash
# Install Anaconda from https://www.anaconda.com/products/individual

# Create a new environment
conda create -n ml-env python=3.10
conda activate ml-env

# Install required packages
pip install -r requirements.txt
```

#### Option B: Using Python venv

```bash
# Create virtual environment
python -m venv ml-env

# Activate (Windows)
ml-env\Scripts\activate

# Activate (Mac/Linux)
source ml-env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Install Jupyter Notebook

```bash
pip install jupyter notebook
jupyter notebook
```

### 4. Start Learning!

**Quick Start Option**: Want to see ML in action immediately? Check out [GETTING_STARTED.md](GETTING_STARTED.md) for a 30-minute first project!

**Full Learning Path**: Follow the modules in order:
1. Start with `00-prerequisites`
2. Progress through each module sequentially
3. Complete exercises and projects
4. Practice, practice, practice!

## Repository Structure

**Note:** All learning modules now include comprehensive detailed guides with code examples, exercises, and solutions. Beginner projects are fully available with READMEs and code. Intermediate and advanced projects have detailed READMEs with instructions.

```
road-to-ml/

 00-prerequisites/
    01-python-basics.md (includes time complexity, iterators/generators)
    02-linear-algebra.md
    03-statistics-probability.md
    04-calculus.md
    05-environment-setup.md
    README.md

 01-python-for-data-science/
    01-numpy.md
    02-pandas.md
    03-visualization.md (includes Plotly & Dash)
    04-exploratory-data-analysis.md
    05-data-sources-and-integration.md
    06-regular-expressions-text-processing.md
    07-advanced-data-wrangling.md
    08-working-with-dates-times.md
    09-streamlit-dashboards.md (NEW)
    README.md

 02-introduction-to-ml/
    introduction-to-ml.md
    ml-terminology.md
    problem-identification-algorithm-selection.md
    first-ml-project-tutorial.md
    common-pitfalls-best-practices.md
    README.md

 03-supervised-learning-regression/
    regression.md
    regression-advanced-topics.md (includes statsmodels)
    regression-project-tutorial.md
    regression-quick-reference.md
    README.md

 04-supervised-learning-classification/
    classification.md
    classification-advanced-topics.md
    classification-project-tutorial.md
    classification-quick-reference.md
    README.md

 05-model-evaluation-optimization/
    evaluation-optimization.md
    evaluation-optimization-advanced-topics.md
    evaluation-optimization-project-tutorial.md
    evaluation-optimization-quick-reference.md
    README.md

 06-ensemble-methods/
    ensemble-methods.md
    ensemble-methods-advanced-topics.md
    ensemble-methods-project-tutorial.md
    ensemble-methods-quick-reference.md
    README.md

 07-feature-engineering/
    feature-engineering.md (includes WOE encoding, advanced discretization)
    feature-engineering-advanced-topics.md
    feature-engineering-project-tutorial.md
    feature-engineering-quick-reference.md
    README.md

 08-unsupervised-learning/
    unsupervised-learning.md
    unsupervised-learning-advanced-topics.md
    unsupervised-learning-project-tutorial.md
    unsupervised-learning-quick-reference.md
    README.md

 09-neural-networks-basics/
    neural-networks.md
    neural-networks-advanced-topics.md
    neural-networks-project-tutorial.md
    neural-networks-quick-reference.md
    README.md

 10-deep-learning-frameworks/
    deep-learning-frameworks.md
    deep-learning-frameworks-advanced-topics.md
    deep-learning-frameworks-project-tutorial.md
    deep-learning-frameworks-quick-reference.md
    README.md

 11-computer-vision/
    computer-vision.md
    computer-vision-advanced-topics.md
    computer-vision-project-tutorial.md
    computer-vision-quick-reference.md
    README.md

 12-natural-language-processing/
    nlp.md
    nlp-advanced-topics.md
    nlp-project-tutorial.md
    nlp-quick-reference.md
    README.md

 13-model-deployment/
    deployment.md
    deployment-advanced-topics.md
    deployment-project-tutorial.md
    deployment-quick-reference.md
    README.md

 14-mlops-basics/
    mlops.md
    mlops-advanced-topics.md
    mlops-project-tutorial.md
    mlops-quick-reference.md
    README.md

 15-time-series-analysis/
    time-series-analysis.md
    time-series-analysis-advanced-topics.md
    time-series-project-tutorial.md
    time-series-quick-reference.md
    README.md

 16-projects-beginner/
    projects-beginner.md
    projects-beginner-advanced-topics.md
    projects-beginner-project-tutorial.md
    projects-beginner-quick-reference.md
    [5 project directories with code]
    README.md

 17-projects-intermediate/
    [6 project directories with READMEs]
    README.md

 18-projects-advanced/
    [7 project directories with READMEs]
    README.md

 19-sql-database-fundamentals/ (NEW)
    sql-database.md
    sql-database-advanced-topics.md
    sql-database-project-tutorial.md
    sql-database-quick-reference.md
    README.md

 20-handling-imbalanced-data/ (NEW)
    imbalanced-data.md
    imbalanced-data-advanced-topics.md
    imbalanced-data-project-tutorial.md
    imbalanced-data-quick-reference.md
    README.md

 21-model-explainability/ (NEW)
    model-explainability.md
    model-explainability-advanced-topics.md
    model-explainability-project-tutorial.md
    model-explainability-quick-reference.md
    README.md

 resources/
    [15 comprehensive guides including cheatsheets, git guide, etc.]

 requirements.txt
 LICENSE
 README.md
 CONTRIBUTING.md
 DISCLAIMER.md
 GETTING_STARTED.md
 LEARNING_ROADMAP.md
 QUICK_START.md
```

## Resources

### Books
- **Hands-On Machine Learning** by Aurélien Géron
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Introduction to Statistical Learning** by James et al.

### Online Courses
- **Machine Learning** by Andrew Ng (Coursera)
- **Deep Learning Specialization** by Andrew Ng (Coursera)
- **Fast.ai Practical Deep Learning**
- **CS229: Machine Learning** (Stanford)

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Kaggle Forums](https://www.kaggle.com/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/machine-learning)

### Cheatsheets & Guides
- [Data Science & ML Cheatsheet](resources/data_science_cheatsheet.md) - Quick reference for daily work (NumPy, Pandas, PyTorch, TensorFlow, OpenCV, FastAPI, and more)
- [Complete Git & GitHub Guide](resources/git_guide.md) - Comprehensive Git tutorial with commands, outputs, practice exercises, and solutions
- [Math Formulas Reference](resources/math_formulas.md) - Essential mathematical formulas for ML (Statistics, Linear Algebra, Calculus, ML metrics)
- [Common Errors & Debugging Guide](resources/common_errors.md) - Troubleshooting guide for common ML errors and debugging strategies
- [Ethics in ML Guide](resources/ethics_in_ml.md) - Comprehensive guide to bias, fairness, responsible AI, and ethical ML practices
- [Model Interpretability Guide](resources/model_interpretability.md) - Understanding and explaining ML model predictions (SHAP, LIME, feature importance)
- [ML Glossary](resources/ml_glossary.md) - Comprehensive glossary of ML terms and concepts for beginners
- [Reinforcement Learning Basics](resources/reinforcement_learning.md) - Introduction to RL, key concepts, algorithms, and applications
- [Recommender Systems Guide](resources/recommender_systems.md) - Building recommendation systems (collaborative filtering, content-based, hybrid approaches)

### Career & Interview Resources
- [Career & Portfolio Guide](resources/career_portfolio.md) - Build your portfolio, prepare for interviews, and launch your ML career
- [Interview Preparation Guide](resources/interview_prep.md) - ML interview questions, coding challenges, system design, and preparation strategies

## Learning Tips

1. **Follow the Order**: Modules are designed to build on each other. Don't skip ahead!
2. **Practice Regularly**: Code along with examples and complete all exercises
3. **Build Projects**: Apply what you learn by building projects
4. **Join Communities**: Engage with others learning ML
5. **Be Patient**: ML is complex - take your time to understand concepts
6. **Experiment**: Don't just copy code - experiment and break things!

## Contributing

We welcome contributions! This repository is for the community, by the community. Here's how you can help:

### Ways to Contribute

- **Add Projects**: Share your ML projects with the community
- **Improve Documentation**: Fix typos, clarify explanations, add examples
- **Create Exercises**: Add practice problems and solutions
- **Report Issues**: Found a bug? Let us know!
- **Suggest Features**: Have an idea? Open an issue!

### Quick Start for Contributors

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/road-to-machine-learning.git`
3. **Create** a branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Your contributions make this resource better for everyone!**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This repository contains links to external resources. Please see [DISCLAIMER.md](DISCLAIMER.md) for important information about external links and resources.

## Show Your Support

If you find this repository helpful, please consider:

- **Star this repo** - It helps others discover this resource
- **Fork it** - Create your own learning path
- **Share it** - Help others on their ML journey
- **Contribute** - Add projects, fix issues, improve content

**Every star and fork helps the community grow!**

## What You'll Achieve

By completing this roadmap, you'll be able to:

- Build and train ML models from scratch
- Deploy models to production
- Work with real-world datasets
- Understand deep learning concepts
- Create computer vision and NLP applications
- Implement MLOps best practices
- Build a portfolio of ML projects

## Statistics

- **21 Learning Modules** covering all ML topics (including SQL, Imbalanced Data, Model Explainability)
- **18 Projects** from beginner to advanced
- **90+ Resources** including books, courses, datasets, and tools
- **5-6 Months** estimated learning time
- **Zero Prerequisites** - start from scratch!

## Disclaimer

**External Links Disclaimer:**

This repository contains links to external websites, courses, documentation, and resources provided for educational purposes only. The maintainers:

- **Do not endorse** any specific external service or content provider
- **Are not responsible** for the availability, accuracy, or content of external links
- **Cannot guarantee** that external links will remain accessible or unchanged
- **Do not assume liability** for any issues arising from the use of external resources

**GDPR Compliance:**

This repository does not collect, store, or process any personal data. It is a static educational resource. Any data processing occurs through GitHub (the platform) or external websites, which have their own privacy policies. See [DISCLAIMER.md](DISCLAIMER.md) for complete disclaimer and GDPR information.

---

<div align="center">

**Made with ❤️ for the ML community**

**Happy Learning!**

*Remember: The journey of a thousand miles begins with a single step. Start with module 00 and keep going!*

[Back to Top](#road-to-ml-from-zero-to-hero)

</div>

