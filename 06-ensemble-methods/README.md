# Phase 6: Ensemble Methods

Learn to combine multiple models for better performance.

##  What You'll Learn

- Bagging (Bootstrap Aggregating)
- Boosting (AdaBoost, Gradient Boosting, XGBoost)
- Stacking
- Voting Classifiers
- When to use Ensemble Methods

##  Topics Covered

### 1. Bagging
- **Bootstrap Aggregating**: Train multiple models on different subsets
- **Random Forest**: Bagging with decision trees
- **Pros**: Reduces variance, handles overfitting
- **Cons**: Less interpretable

### 2. Boosting
- **AdaBoost**: Adaptive boosting
- **Gradient Boosting**: Sequential error correction
- **XGBoost**: Optimized gradient boosting
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features well

### 3. Stacking
- **Meta-Learner**: Train a model on predictions of base models
- **Blending**: Similar to stacking with validation set
- **When to use**: When you have diverse base models

### 4. Voting
- **Hard Voting**: Majority class wins
- **Soft Voting**: Average probabilities
- **When to use**: Quick ensemble of different algorithms

##  Learning Objectives

By the end of this module, you should be able to:
- Implement various ensemble methods
- Understand when to use each technique
- Tune ensemble hyperparameters
- Build winning competition models

##  Projects

1. **Kaggle Competition**: Use ensembles to improve performance
2. **Model Comparison**: Compare single models vs ensembles
3. **XGBoost Project**: Build a production-ready model with XGBoost

##  Key Concepts

- **Wisdom of the Crowd**: Multiple models often better than one
- **Diversity**: Ensembles work best with diverse base models
- **Bias-Variance**: Ensembles reduce variance
- **Computational Cost**: Ensembles are more expensive

##  Additional Resources

- [Ensemble Methods - Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

**Previous Phase:** [05-model-evaluation-optimization](../05-model-evaluation-optimization/README.md)  
**Next Phase:** [07-feature-engineering](../07-feature-engineering/README.md)

