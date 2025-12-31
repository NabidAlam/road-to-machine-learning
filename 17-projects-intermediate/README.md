# Intermediate Projects

More challenging projects that combine multiple ML concepts and techniques.

##  Prerequisites

Before starting these projects, you should have completed:
- All Beginner Projects
- Phase 5: Model Evaluation & Optimization
- Phase 6: Ensemble Methods
- Phase 7: Feature Engineering
- Phase 8: Unsupervised Learning (for some projects)

##  Projects

### Project 1: Handwritten Digit Recognition (MNIST)
**Difficulty**:   
**Time**: 3-5 days  
**Skills**: Neural Networks, Image Processing, Deep Learning Basics

Build a neural network to classify handwritten digits (0-9).

**What you'll learn:**
- Building neural networks
- Image preprocessing
- Hyperparameter tuning
- Deep learning workflow

**Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) (built into Keras/TensorFlow)

**Extensions:**
- Try different architectures
- Use data augmentation
- Achieve >99% accuracy

---

### Project 2: Customer Churn Prediction
**Difficulty**:   
**Time**: 4-5 days  
**Skills**: Classification, Feature Engineering, Imbalanced Data, Business Metrics

Predict which customers will leave a service (churn prediction).

**What you'll learn:**
- Handling imbalanced datasets
- Business-focused metrics (cost-sensitive learning)
- Feature engineering from customer data
- Ensemble methods
- Model interpretation

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or similar

**Extensions:**
- Calculate cost of churn
- Recommend retention strategies
- Build a dashboard

---

### Project 3: Movie Recommendation System
**Difficulty**:   
**Time**: 5-7 days  
**Skills**: Collaborative Filtering, Content-Based Filtering, Matrix Factorization

Build a recommendation system to suggest movies to users.

**What you'll learn:**
- Collaborative filtering
- Content-based filtering
- Matrix factorization
- Evaluation metrics for recommendations
- Hybrid approaches

**Dataset**: [MovieLens](https://grouplens.org/datasets/movielens/)

**Extensions:**
- Hybrid recommendation system
- Real-time recommendations
- Cold-start problem solutions

---

### Project 4: Credit Card Fraud Detection
**Difficulty**:   
**Time**: 4-5 days  
**Skills**: Anomaly Detection, Imbalanced Data, Classification, Feature Engineering

Detect fraudulent credit card transactions.

**What you'll learn:**
- Handling highly imbalanced data
- Anomaly detection techniques
- Precision/Recall tradeoffs
- Cost-sensitive learning
- Feature engineering for fraud detection

**Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Extensions:**
- Real-time fraud detection
- Explainable AI for fraud cases
- Cost-benefit analysis

---

### Project 5: Customer Segmentation
**Difficulty**:   
**Time**: 3-4 days  
**Skills**: Unsupervised Learning, Clustering, Dimensionality Reduction

Segment customers into groups based on behavior and demographics.

**What you'll learn:**
- K-means clustering
- Hierarchical clustering
- Choosing number of clusters
- Interpreting clusters
- Business applications

**Dataset**: [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) or e-commerce data

**Extensions:**
- Multiple clustering algorithms
- Cluster visualization
- Actionable insights from segments

---

### Project 6: Time Series Forecasting
**Difficulty**:   
**Time**: 4-5 days  
**Skills**: Time Series Analysis, ARIMA, LSTM, Feature Engineering

Forecast future values in a time series (sales, stock prices, etc.).

**What you'll learn:**
- Time series preprocessing
- ARIMA models
- LSTM for sequences
- Feature engineering for time series
- Evaluation for time series

**Dataset**: [Airline Passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers) or stock data

**Extensions:**
- Multiple forecasting methods
- Confidence intervals
- Real-time forecasting

---

### Project 7: Feature Engineering Mastery
**Difficulty**:   
**Time**: 4-5 days  
**Skills**: Feature Engineering, Feature Selection, sklearn Pipeline, Advanced Encoding

Master comprehensive feature engineering techniques on a complex real-world dataset.

**What you'll learn:**
- Apply comprehensive feature engineering techniques
- Use sklearn Pipeline and ColumnTransformer effectively
- Handle mixed data types (numeric, categorical, text)
- Implement advanced encoding techniques (WOE, Target Encoding)
- Perform feature selection using multiple methods
- Build robust preprocessing pipelines

**Dataset**: House Prices, Credit Card Default, or Employee Attrition datasets

**Extensions:**
- Automated feature engineering
- Feature store implementation
- Feature monitoring

---

### Project 8: Ensemble Methods Comparison
**Difficulty**:   
**Time**: 4-5 days  
**Skills**: Ensemble Methods, Bagging, Boosting, Stacking, Hyperparameter Tuning

Comprehensive comparison of ensemble methods: Bagging, Boosting, Stacking, and Voting.

**What you'll learn:**
- Understand different ensemble methods and when to use them
- Implement and compare Bagging vs Boosting
- Use advanced boosting algorithms (XGBoost, LightGBM, CatBoost)
- Build Stacking and Voting ensembles
- Tune hyperparameters for ensemble methods
- Compare model performance comprehensively

**Dataset**: Titanic, Credit Card Fraud, Customer Churn, or House Prices

**Extensions:**
- Custom ensemble creation
- Ensemble of ensembles
- Feature importance comparison

---

##  Project Structure

Each project should include:
```
project-name/
 README.md              # Detailed project description
 data/                  # Dataset
 notebooks/             # Analysis notebooks
    01-data-exploration.ipynb
    02-feature-engineering.ipynb
    03-model-development.ipynb
    04-model-evaluation.ipynb
    05-results-analysis.ipynb
 src/                   # Source code modules
    data_preprocessing.py
    feature_engineering.py
    models.py
    evaluation.py
 models/                # Saved models
 results/                # Results, plots, reports
 tests/                 # Unit tests
 requirements.txt       # Dependencies
 config.yaml            # Configuration file
```

##  Tips for Success

1. **Plan First**: Outline your approach before coding
2. **Iterate**: Build baseline, then improve incrementally
3. **Document**: Keep detailed notes on decisions and results
4. **Visualize**: Create comprehensive visualizations
5. **Compare**: Try multiple approaches and compare
6. **Present**: Create a clear presentation of results

##  Learning Outcomes

After completing these projects, you should be able to:
- Handle complex, real-world datasets
- Apply advanced feature engineering
- Use ensemble methods effectively
- Handle imbalanced data
- Build end-to-end ML pipelines
- Present results to stakeholders

##  Additional Resources

- [Kaggle Competitions](https://www.kaggle.com/competitions) - Practice with real competitions
- [Papers with Code](https://paperswithcode.com/) - See state-of-the-art approaches
- [Towards Data Science](https://towardsdatascience.com/) - Learn from others' projects

---

**Ready for a challenge?** Start with MNIST if you want to learn deep learning, or Customer Churn for a business-focused project.

**Next Level:** [18-projects-advanced](../18-projects-advanced/README.md)

