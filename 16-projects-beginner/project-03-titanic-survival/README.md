# Project 3: Titanic Survival Prediction

Predict which passengers survived the Titanic disaster using passenger information.

## Difficulty
Beginner

## Time Estimate
2-3 days

## Skills You'll Practice
- Classification
- Feature Engineering
- Data Cleaning
- Handling Missing Values
- Categorical Encoding

## Learning Objectives

By completing this project, you will learn to:
- Handle missing data effectively
- Encode categorical variables
- Engineer features from raw data
- Apply multiple classification algorithms
- Evaluate classification models
- Handle imbalanced datasets

## Dataset

**Kaggle Titanic Competition**
- [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Download train.csv and test.csv
- Classic beginner-friendly dataset

## Project Steps

### Step 1: Load and Explore Data
- Load train and test datasets
- Check data shape and basic statistics
- Identify missing values
- Explore feature distributions
- Analyze survival rates by different features

### Step 2: Data Preprocessing
- Handle missing values (Age, Cabin, Embarked)
- Create new features (Family Size, Title, etc.)
- Encode categorical variables (Sex, Embarked)
- Remove irrelevant features

### Step 3: Feature Engineering
- Extract title from Name
- Create family size from SibSp and Parch
- Create age groups
- Create fare groups
- Handle outliers

### Step 4: Model Training
- Split data into train/validation sets
- Train multiple models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors
- Compare model performance

### Step 5: Model Evaluation
- Calculate accuracy, precision, recall, F1-score
- Create confusion matrix
- Plot ROC curve
- Analyze feature importance

### Step 6: Model Improvement
- Tune hyperparameters
- Try ensemble methods
- Feature selection
- Final model selection and submission

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis:
   - EDA with visualizations
   - Data preprocessing steps
   - Feature engineering
   - Model training and evaluation
   - Results and conclusions

2. **Submission File**:
   - CSV file with predictions for test set
   - Format: PassengerId, Survived

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted survivors, how many actually survived
- **Recall**: Of actual survivors, how many were found
- **F1-Score**: Harmonic mean of precision and recall

## Key Features to Explore

- **Pclass**: Passenger class (1, 2, 3)
- **Sex**: Gender (male, female)
- **Age**: Age of passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation
- **Cabin**: Cabin number (many missing)

## Feature Engineering Ideas

- Extract title from Name (Mr, Mrs, Miss, etc.)
- Create family size = SibSp + Parch + 1
- Create is_alone flag
- Create age groups (child, adult, senior)
- Create fare groups
- Extract deck from Cabin (if available)

## Tips

- Start with simple models (Logistic Regression)
- Visualize survival rates by different features
- Handle missing Age carefully (use median or predict)
- Sex is a very important feature
- Try creating new features from existing ones
- Submit to Kaggle to see your score!

## Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
- [Scikit-learn Classification](https://scikit-learn.org/stable/supervised_learning.html#classification)

## Next Steps

After completing this project:
- Submit predictions to Kaggle
- Try advanced feature engineering
- Experiment with ensemble methods
- Move to [Project 4: Spam Detection](../project-04-spam-detection/README.md)

