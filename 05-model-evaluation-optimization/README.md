# Phase 5: Model Evaluation & Optimization

Learn to properly evaluate models and optimize their performance.

##  What You'll Learn

- Train/Validation/Test Split
- Cross-Validation Techniques
- Hyperparameter Tuning
- Bias-Variance Tradeoff
- Overfitting and Underfitting
- Model Selection Strategies

##  Topics Covered

### 1. Data Splitting
- **Train Set**: Used to train the model
- **Validation Set**: Used to tune hyperparameters
- **Test Set**: Used for final evaluation (only touched once!)
- **Stratified Splitting**: Maintains class distribution

### 2. Cross-Validation
- **K-Fold Cross-Validation**: Divide data into k folds
- **Stratified K-Fold**: Maintains class distribution
- **Leave-One-Out**: Extreme case of k-fold
- **Time Series Cross-Validation**: For time-dependent data

### 3. Hyperparameter Tuning
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameters
- **Bayesian Optimization**: Smart search using previous results
- **Optuna/Hyperopt**: Advanced optimization libraries

### 4. Bias-Variance Tradeoff
- **Bias**: Error from oversimplifying assumptions
- **Variance**: Error from sensitivity to small fluctuations
- **Tradeoff**: Balancing bias and variance
- **Bias-Variance Decomposition**: Understanding error sources

### 5. Overfitting and Underfitting
- **Overfitting**: Model memorizes training data
- **Underfitting**: Model too simple to capture patterns
- **Solutions**: Regularization, more data, simpler models

### 6. Learning Curves
- Plotting training vs validation performance
- Identifying overfitting/underfitting
- Determining if more data will help

##  Learning Objectives

By the end of this module, you should be able to:
- Properly split data for ML workflows
- Implement cross-validation
- Tune hyperparameters effectively
- Diagnose and fix overfitting/underfitting
- Interpret learning curves

##  Projects

1. **Hyperparameter Tuning Project**: Optimize a model's hyperparameters
2. **Cross-Validation Comparison**: Compare different CV strategies
3. **Learning Curve Analysis**: Analyze model learning behavior

##  Key Concepts

- **Never touch test set until final evaluation!**
- **Validation set** is for hyperparameter tuning
- **Cross-validation** gives more reliable performance estimates
- **Learning curves** help diagnose model issues
- **Regularization** helps prevent overfitting

##  Additional Resources

- [Model Evaluation - Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-Validation - Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning - Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html)

---

**Previous Phase:** [04-supervised-learning-classification](../04-supervised-learning-classification/README.md)  
**Next Phase:** [06-ensemble-methods](../06-ensemble-methods/README.md)

