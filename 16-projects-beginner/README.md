# Beginner Projects

Hands-on projects to apply what you've learned in the foundational modules.

##  Prerequisites

Before starting these projects, you should have completed:
- Phase 0: Prerequisites
- Phase 1: Python for Data Science
- Phase 2: Introduction to ML
- Phase 3: Supervised Learning - Regression
- Phase 4: Supervised Learning - Classification

##  Projects

### Project 1: House Price Prediction
**Difficulty**:   
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
**Difficulty**:   
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
**Difficulty**:   
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
**Difficulty**:   
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
**Difficulty**:   
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

- [Kaggle Learn](https://www.kaggle.com/learn) - Micro-courses with projects
- [UCI ML Repository](https://archive.ics.uci.edu/) - More datasets
- [Papers with Code](https://paperswithcode.com/) - See how others solved problems

---

**Ready to start?** Pick a project and begin! Start with Iris Classification if you're completely new, or jump to House Price Prediction for a more comprehensive project.

**Next Level:** [17-projects-intermediate](../17-projects-intermediate/README.md)

