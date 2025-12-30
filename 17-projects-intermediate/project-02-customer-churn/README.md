# Project 2: Customer Churn Prediction

Predict which customers will leave a service (churn prediction).

## Difficulty
Intermediate

## Time Estimate
4-5 days

## Skills You'll Practice
- Classification
- Feature Engineering
- Imbalanced Data Handling
- Business Metrics
- Model Interpretation

## Learning Objectives

By completing this project, you will learn to:
- Handle imbalanced datasets
- Apply business-focused metrics
- Engineer features from customer data
- Use ensemble methods
- Interpret model results for business

## Dataset

**Telco Customer Churn Dataset**
- [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Customer information and churn status
- Features: demographics, services, charges, etc.

## Project Steps

### Step 1: Load and Explore Data
- Load dataset
- Analyze churn rate
- Explore feature distributions
- Identify missing values
- Check for class imbalance

### Step 2: Data Preprocessing
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Create train/test split

### Step 3: Feature Engineering
- Create new features (e.g., tenure groups, charge ratios)
- Handle categorical variables
- Feature selection
- Correlation analysis

### Step 4: Handle Class Imbalance
- Use SMOTE for oversampling
- Adjust class weights
- Try different sampling strategies
- Compare approaches

### Step 5: Model Training
- Train multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Ensemble methods
- Use cross-validation

### Step 6: Model Evaluation
- Calculate accuracy, precision, recall, F1
- Focus on recall (finding churners)
- Calculate business metrics (cost of churn)
- Feature importance analysis

### Step 7: Business Insights
- Identify key churn factors
- Recommend retention strategies
- Calculate ROI of interventions
- Create actionable insights

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis
2. **Model** with good recall for churners
3. **Business Report** with insights and recommendations
4. **Dashboard** (optional) showing key metrics

## Evaluation Metrics

- **Recall**: Most important (find actual churners)
- **Precision**: Minimize false positives
- **F1-Score**: Balance
- **Business Cost**: Calculate cost of churn vs retention

## Key Features to Explore

- Tenure (how long customer has been with company)
- Monthly charges
- Services used
- Contract type
- Payment method

## Tips

- Focus on recall (finding churners is more important)
- Calculate cost of false negatives vs false positives
- Feature engineering is crucial
- Try ensemble methods
- Interpret results for business stakeholders

## Resources

- [Kaggle Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)
- [Business Metrics for ML](https://towardsdatascience.com/business-metrics-for-machine-learning-7d3c3b3c0c0e)

## Extensions

- Build retention campaign recommendations
- Create real-time churn prediction system
- A/B test retention strategies
- Deploy model as API

## Next Steps

After completing this project:
- Try other churn datasets
- Experiment with advanced techniques
- Move to [Project 3: Movie Recommendation](../project-03-movie-recommendation/README.md)

