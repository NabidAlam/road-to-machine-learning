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

## Documentation & Learning Resources

**Official Documentation:**
- [Model Evaluation - Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-Validation - Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning - Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html)

**Free Courses:**
- [Model Evaluation (Coursera)](https://www.coursera.org/learn/machine-learning) - Week 6 of Andrew Ng's course
- [Hyperparameter Tuning (Kaggle Learn)](https://www.kaggle.com/learn/intro-to-machine-learning) - Free micro-course

**Tutorials:**
- [Cross-Validation Explained](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning Guide](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
- [Bias-Variance Tradeoff Explained](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
- [Learning Curves Tutorial](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

**Video Tutorials:**
- [Cross-Validation (StatQuest)](https://www.youtube.com/watch?v=fSytzGwwBVw)
- [Bias-Variance Tradeoff (StatQuest)](https://www.youtube.com/watch?v=EuBBz3bI-aA)
- [Overfitting (StatQuest)](https://www.youtube.com/watch?v=Anq4Pgd5el8)

**Tools:**
- [Optuna Documentation](https://optuna.org/) - Advanced hyperparameter optimization
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/) - Bayesian optimization

**[Complete Detailed Guide →](evaluation-optimization.md)**

### Additional Resources

- **[Advanced Evaluation Topics](evaluation-optimization-advanced-topics.md)** - Nested cross-validation, custom scoring functions, model selection strategies, early stopping, ensemble model selection, performance profiling, and common pitfalls
- **[Complete Evaluation Project Tutorial](evaluation-optimization-project-tutorial.md)** - Step-by-step walkthrough of properly evaluating and optimizing a model from data splitting to final evaluation
- **[Evaluation Quick Reference](evaluation-optimization-quick-reference.md)** - Quick reference guide with code snippets, data splitting strategies, cross-validation methods, hyperparameter tuning, and best practices

---

**Previous Phase:** [04-supervised-learning-classification](../04-supervised-learning-classification/README.md)  
**Next Phase:** [06-ensemble-methods](../06-ensemble-methods/README.md)

