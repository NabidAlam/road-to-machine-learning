# Beginner Projects

Hands-on projects to apply what you've learned in the foundational modules. **All six projects include runnable code or notebooks in the repo.**

> **Timing:** Start the first beginner project after **Stage 2** (modules 02–05) while continuing later modules, you do not need to wait for Stage 9. See [START-HERE.md](../START-HERE.md#projects-run-in-parallel-not-only-at-the-end).

## Prerequisites

Before starting these projects, you should have completed:
- Module 00: Prerequisites
- Module 01: Python for Data Science
- Module 02: Introduction to ML
- Module 03: Supervised Learning. Regression
- Module 04: Supervised Learning. Classification

## ML for beginners curriculum map (projects)

Use these projects to **implement algorithms in context** (regression, classification, cleaning, feature work). Cross-check concepts with the [intro map](../02-introduction-to-ml/introduction-to-ml.md#ml-for-beginners-curriculum-map-this-guide).

| Practice focus | Project |
|----------------|---------|
| Regression, metrics, end-to-end pipeline | [House price prediction](project-01-house-price-prediction/README.md) |
| Classification, multiple algorithms | [Iris classification](project-02-iris-classification/README.md) |
| EDA, cleaning, feature engineering, classification | [Titanic survival](project-03-titanic-survival/README.md) |
| Text features, classification | [Spam detection](project-04-spam-detection/README.md) |
| Regression on tabular physicochemical data | [Wine quality](project-05-wine-quality/README.md) |
| Dashboard / reporting (optional extension) | [Customer dashboard](project-06-customer-dashboard/README.md) |

##  Projects

### Project 1: House Price Prediction
**Difficulty**: Beginner  
**Time**: 2-3 days  
**Skills**: Regression, Feature Engineering, EDA

Predict house prices using features like size, location, number of rooms, etc.

**What you'll learn:**
- Data cleaning and preprocessing
- Feature engineering
- Linear regression
- Model evaluation

**Dataset**: [California Housing Prices](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) or [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

### Project 2: Iris Flower Classification
**Difficulty**: Beginner  
**Time**: 1 day  
**Skills**: Classification, EDA, Multiple Algorithms

Classify iris flowers into three species using petal and sepal measurements.

**What you'll learn:**
- Exploratory data analysis
- Multiple classification algorithms
- Model comparison
- Visualization

**Dataset**: Built into scikit-learn (`load_iris()`)

---

### Project 3: Titanic Survival Prediction
**Difficulty**: Beginner  
**Time**: 2-3 days  
**Skills**: Classification, Feature Engineering, Data Cleaning

Predict which passengers survived the Titanic disaster.

**What you'll learn:**
- Handling missing data
- Categorical encoding
- Feature engineering
- Multiple classification models

**Dataset**: [Kaggle Titanic](https://www.kaggle.com/c/titanic)

---

### Project 4: Spam Email Detection
**Difficulty**: Beginner  
**Time**: 2-3 days  
**Skills**: Text Classification, NLP Basics, Feature Engineering

Classify emails as spam or not spam using text features.

**What you'll learn:**
- Text preprocessing
- Feature extraction from text
- Classification with text data
- Evaluation metrics

**Dataset**: [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

### Project 5: Wine Quality Prediction
**Difficulty**: Beginner  
**Time**: 2-3 days  
**Skills**: Regression/Classification, Feature Selection

Predict wine quality based on chemical properties (can be regression or classification).

**What you'll learn:**
- Feature selection
- Handling imbalanced data
- Model tuning
- Both regression and classification approaches

**Dataset**: [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

### Project 6: Customer Data Dashboard with Streamlit
**Difficulty**: Beginner  
**Time**: 2-3 days  
**Skills**: Streamlit, Data Visualization, Interactive Dashboards

Build an interactive dashboard to visualize and analyze customer data using Streamlit.

**What you'll learn:**
- Build interactive web applications with Streamlit
- Create dynamic visualizations
- Implement user input and filtering
- Design user-friendly dashboards
- Deploy Streamlit applications
- Work with real-world customer data

**Dataset**: E-commerce Customer Data, Telco Customer Churn, or any customer dataset

**Extensions:**
- Add machine learning predictions
- Real-time data updates
- Multi-page dashboard
- Advanced visualizations

---

##  Project Structure

Each project should include:
```
project-name/
 README.md              # Project description and instructions
 data/                  # Dataset (or links to download)
 notebooks/             # Jupyter notebooks
    01-exploratory-data-analysis.ipynb
    02-data-preprocessing.ipynb
    03-model-training.ipynb
    04-model-evaluation.ipynb
 src/                   # Source code (if applicable)
 models/                # Saved models
 results/               # Results and visualizations
 requirements.txt       # Project-specific dependencies
```

##  Tips for Success

1. **Start Simple**: Begin with basic models, then improve
2. **Explore First**: Always do EDA before modeling
3. **Iterate**: Build, evaluate, improve, repeat
4. **Document**: Comment your code and explain decisions
5. **Visualize**: Create plots to understand data and results
6. **Compare**: Try multiple algorithms and compare results

##  Learning Outcomes

After completing these projects, you should be able to:
- Clean and preprocess real-world data
- Apply regression and classification algorithms
- Evaluate models appropriately
- Present results clearly
- Handle common data issues (missing values, outliers, etc.)

##  Additional Resources

- [Kaggle Learn](https://www.kaggle.com/learn): Micro-courses with projects
- [UCI ML Repository](https://archive.ics.uci.edu/): More datasets
- [Papers with Code](https://paperswithcode.com/): See how others solved problems

---

**Complete Guide:**
- [Projects Beginner Guide →](projects-beginner.md): Comprehensive guide to building ML projects
- [Advanced Topics →](projects-beginner-advanced-topics.md): Advanced techniques for improving projects
- [Project Tutorial →](projects-beginner-project-tutorial.md): Step-by-step Titanic project walkthrough
- [Quick Reference →](projects-beginner-quick-reference.md): Quick lookup guide

**Ready to start?** Pick a project and begin! Start with Iris Classification if you're completely new, or jump to House Price Prediction for a more comprehensive project.

**Next Level:** [17-projects-intermediate](../17-projects-intermediate/README.md)

