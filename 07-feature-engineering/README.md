# Phase 7: Feature Engineering

Learn to create and select the best features for your models.

##  What You'll Learn

- Feature Selection Techniques
- Feature Transformation
- Handling Categorical Variables
- Feature Scaling and Normalization
- Dimensionality Reduction (PCA)
- Creating New Features

##  Topics Covered

### 1. Feature Selection
- **Filter Methods**: Statistical tests (Chi-square, correlation)
- **Wrapper Methods**: Forward/backward selection
- **Embedded Methods**: Lasso, tree-based importance
- **Removing Redundant Features**: Correlation analysis

### 2. Feature Transformation
- **Log Transformation**: Handle skewed distributions
- **Power Transformation**: Box-Cox transformation
- **Binning**: Convert continuous to categorical
- **Polynomial Features**: Create interaction terms

### 3. Categorical Variables
- **One-Hot Encoding**: Binary columns for each category
- **Label Encoding**: Numeric labels (for tree models)
- **Target Encoding**: Mean target per category
- **Frequency Encoding**: Count of category occurrences

### 4. Feature Scaling
- **Standardization**: Mean 0, Std 1 (Z-score)
- **Normalization**: Scale to [0, 1] range
- **Robust Scaling**: Using median and IQR
- **When to scale**: Required for distance-based algorithms

### 5. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Linear dimensionality reduction
- **t-SNE**: Non-linear visualization
- **UMAP**: Modern alternative to t-SNE
- **When to use**: High-dimensional data, visualization

### 6. Creating Features
- **Domain Knowledge**: Industry-specific features
- **Temporal Features**: Time-based features
- **Interaction Features**: Combinations of features
- **Aggregation Features**: Group statistics

##  Learning Objectives

By the end of this module, you should be able to:
- Select relevant features
- Transform features appropriately
- Handle categorical variables
- Apply dimensionality reduction
- Create domain-specific features

##  Projects

1. **Feature Engineering Challenge**: Improve model with better features
2. **PCA Visualization**: Visualize high-dimensional data
3. **Categorical Encoding Comparison**: Compare encoding methods

##  Key Concepts

- **Garbage In, Garbage Out**: Good features = good models
- **Domain Knowledge**: Often more valuable than algorithms
- **Feature Importance**: Understand which features matter
- **Curse of Dimensionality**: Too many features can hurt

## Documentation & Learning Resources

**Official Documentation:**
- [Feature Selection - Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)
- [PCA - Scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

**Free Courses:**
- [Feature Engineering (Kaggle Learn)](https://www.kaggle.com/learn/feature-engineering) - Free comprehensive course
- [Feature Engineering (Coursera)](https://www.coursera.org/learn/feature-engineering) - Free audit available

**Tutorials:**
- [Feature Engineering Tutorial](https://www.kaggle.com/learn/feature-engineering)
- [PCA Explained Simply](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
- [Feature Selection Methods](https://machinelearningmastery.com/feature-selection-machine-learning-python/)
- [Categorical Encoding Guide](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

**Video Tutorials:**
- [PCA (StatQuest)](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [Feature Engineering (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk)
- [Dimensionality Reduction](https://www.youtube.com/watch?v=9Z4wKJY5J8I)

**Books:**
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) - Check library access

**[Complete Detailed Guide →](feature-engineering.md)**

**Additional Resources:**
- [Advanced Topics →](feature-engineering-advanced-topics.md) - Automated feature engineering, pipelines, advanced techniques
- [Project Tutorial →](feature-engineering-project-tutorial.md) - Step-by-step feature engineering project
- [Quick Reference →](feature-engineering-quick-reference.md) - Quick lookup guide for feature engineering

---

**Previous Module:** [06-ensemble-methods](../06-ensemble-methods/README.md)  
**Next Module:** [08-unsupervised-learning](../08-unsupervised-learning/README.md)

