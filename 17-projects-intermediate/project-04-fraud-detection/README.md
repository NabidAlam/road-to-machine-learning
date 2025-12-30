# Project 4: Credit Card Fraud Detection

Detect fraudulent credit card transactions.

## Difficulty
Intermediate

## Time Estimate
4-5 days

## Skills You'll Practice
- Anomaly Detection
- Imbalanced Data Handling
- Classification
- Feature Engineering
- Precision/Recall Tradeoffs

## Learning Objectives

By completing this project, you will learn to:
- Handle highly imbalanced data
- Apply anomaly detection techniques
- Balance precision and recall
- Use cost-sensitive learning
- Engineer features for fraud detection

## Dataset

**Credit Card Fraud Detection**
- [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Highly imbalanced (0.17% fraud)
- Anonymized features (PCA transformed)
- Time and amount features

## Project Steps

### Step 1: Load and Explore Data
- Load dataset
- Analyze class distribution (highly imbalanced!)
- Check for missing values
- Explore feature distributions
- Analyze fraud patterns

### Step 2: Handle Imbalance
- Use SMOTE for oversampling
- Use undersampling techniques
- Adjust class weights
- Try different sampling strategies

### Step 3: Feature Engineering
- Create time-based features
- Transform amount feature
- Create interaction features
- Feature scaling

### Step 4: Anomaly Detection
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Compare with classification

### Step 5: Model Training
- Train multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Isolation Forest
- Use cross-validation

### Step 6: Model Evaluation
- Focus on recall (catch fraud)
- Calculate precision (minimize false alarms)
- ROC-AUC curve
- Precision-Recall curve
- Cost-benefit analysis

### Step 7: Threshold Tuning
- Adjust classification threshold
- Balance precision and recall
- Calculate cost of errors
- Choose optimal threshold

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis
2. **Model** with high recall for fraud
3. **Evaluation Report** with metrics and tradeoffs
4. **Cost Analysis** of different thresholds

## Evaluation Metrics

- **Recall**: Most important (catch actual fraud)
- **Precision**: Minimize false positives
- **F1-Score**: Balance
- **ROC-AUC**: Overall performance
- **Cost**: Financial impact

## Key Challenges

- Extreme class imbalance (0.17% fraud)
- Need high recall (catch fraud)
- Need reasonable precision (avoid false alarms)
- Cost-sensitive problem

## Tips

- Focus on recall first (catch fraud)
- Use anomaly detection techniques
- Try ensemble methods
- Carefully tune threshold
- Calculate cost of false negatives vs false positives
- Use stratified sampling for evaluation

## Resources

- [Kaggle Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Imbalanced Learn](https://imbalanced-learn.org/stable/)
- [Anomaly Detection Guide](https://scikit-learn.org/stable/modules/outlier_detection.html)

## Extensions

- Real-time fraud detection system
- Explainable AI for fraud cases
- Cost-benefit analysis dashboard
- Deploy as API service

## Next Steps

After completing this project:
- Try other fraud datasets
- Experiment with deep learning
- Move to [Project 5: Customer Segmentation](../project-05-customer-segmentation/README.md)

