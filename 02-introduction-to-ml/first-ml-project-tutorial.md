# Your First ML Project: Step-by-Step Tutorial

Complete walkthrough of building your first machine learning project from scratch. Follow along to build a real ML model!

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Problem Definition](#step-1-problem-definition)
- [Step 2: Data Collection](#step-2-data-collection)
- [Step 3: Data Exploration](#step-3-data-exploration)
- [Step 4: Data Preparation](#step-4-data-preparation)
- [Step 5: Model Training](#step-5-model-training)
- [Step 6: Model Evaluation](#step-6-model-evaluation)
- [Step 7: Making Predictions](#step-7-making-predictions)
- [Next Steps](#next-steps)

---

## Project Overview

**Project**: Predict Iris Flower Species

**Goal**: Build a model that can classify iris flowers into three species based on measurements.

**Dataset**: Famous Iris dataset (built into scikit-learn)
- 150 samples
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Setosa, Versicolor, Virginica

**Type**: Classification (Supervised Learning)

**Difficulty**: Beginner-friendly

**Time**: 30-60 minutes

---

## Step 1: Problem Definition

### Understand the Problem

**Question**: Can we predict iris species from flower measurements?

**Business Value**: 
- Botanical classification
- Quality control in flower production
- Educational example

**Success Criteria**: 
- Accuracy > 90%
- Model can classify new flowers correctly

### Define Inputs and Outputs

**Inputs (Features)**:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Output (Target)**:
- Species: Setosa, Versicolor, or Virginica

---

## Step 2: Data Collection

### Load the Dataset

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
```

**Output:**
```
Dataset loaded successfully!
Shape: (150, 6)

First few rows:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  species  species_name
0                5.1               3.5                1.4               0.2        0        setosa
1                4.9               3.0                1.4               0.2        0        setosa
2                4.7               3.2                1.3               0.2        0        setosa
3                4.6               3.1                1.5               0.2        0        setosa
4                5.0               3.6                1.4               0.2        0        setosa
```

---

## Step 3: Data Exploration

### Basic Information

```python
# Dataset information
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nClass Distribution:")
print(df['species_name'].value_counts())
```

### Check for Issues

```python
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check data types
print("\nData types:")
print(df.dtypes)
```

### Visualizations

```python
# Pair plot to see relationships
sns.pairplot(df, hue='species_name', diag_kind='hist')
plt.suptitle('Iris Dataset - Feature Relationships', y=1.02)
plt.show()

# Box plots for each feature by species
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
features = iris.feature_names

for idx, feature in enumerate(features):
    row = idx // 2
    col = idx % 2
    sns.boxplot(data=df, x='species_name', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Species')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

**Insights from Exploration:**
- No missing values ✓
- No duplicates ✓
- 50 samples per class (balanced) ✓
- Features are correlated (petal length/width highly correlated)
- Setosa is easily separable from other two species

---

## Step 4: Data Preparation

### Separate Features and Target

```python
# Features (X) - what we use to predict
X = df[iris.feature_names]

# Target (y) - what we want to predict
y = df['species']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures:\n{X.head()}")
print(f"\nTarget:\n{y.head()}")
```

### Split Data

```python
# Split into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTest class distribution:")
print(pd.Series(y_test).value_counts().sort_index())
```

**Output:**
```
Training set: 120 samples
Test set: 30 samples

Training class distribution:
0    40
1    40
2    40
dtype: int64

Test class distribution:
0    10
1    10
2    10
dtype: int64
```

### Feature Scaling (Optional for this dataset)

```python
# For this dataset, scaling isn't critical (all features in similar range)
# But it's good practice to know how to do it

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit on training data only!
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data (don't fit again!)
X_test_scaled = scaler.transform(X_test)

# Note: For this tutorial, we'll use unscaled data
# Random Forest doesn't require scaling
```

---

## Step 5: Model Training

### Choose Algorithm

For this problem:
- **Classification** problem
- **Small dataset** (150 samples)
- **Need good performance**
- **Interpretability helpful but not critical**

**Choice**: Random Forest Classifier
- Good performance
- Handles non-linear relationships
- Works well with small datasets
- Provides feature importance

### Create and Train Model

```python
# Create model
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    random_state=42,     # For reproducibility
    max_depth=5          # Prevent overfitting
)

# Train the model
print("Training model...")
model.fit(X_train, y_train)
print("Training complete!")

# Check training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")
```

**Output:**
```
Training model...
Training complete!
Training Accuracy: 1.0000
```

### Feature Importance

```python
# See which features are most important
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

**Insight**: Petal measurements are more important than sepal measurements for classification.

---

## Step 6: Model Evaluation

### Make Predictions

```python
# Predict on test set
y_pred = model.predict(X_test)

print("Predictions made!")
print(f"First 10 predictions: {y_pred[:10]}")
print(f"First 10 actual: {y_test.iloc[:10].values}")
```

### Calculate Metrics

```python
# Overall accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))
```

**Output:**
```
Test Accuracy: 1.0000 (100.00%)

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### Confusion Matrix

```python
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\nConfusion Matrix:")
print(cm)
```

### Check for Overfitting

```python
# Compare training and test accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Difference: {abs(train_acc - test_acc):.4f}")

if abs(train_acc - test_acc) < 0.05:
    print("✓ Good generalization! Model is not overfitting.")
elif train_acc > test_acc + 0.1:
    print("⚠ Warning: Possible overfitting (large gap between train and test)")
else:
    print("✓ Model performance is consistent.")
```

---

## Step 7: Making Predictions

### Predict on New Data

```python
# Example: New flower measurements
new_flower = {
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)': 0.2
}

# Convert to array
new_flower_array = np.array([
    new_flower['sepal length (cm)'],
    new_flower['sepal width (cm)'],
    new_flower['petal length (cm)'],
    new_flower['petal width (cm)']
]).reshape(1, -1)

# Make prediction
prediction = model.predict(new_flower_array)
prediction_proba = model.predict_proba(new_flower_array)

species_name = iris.target_names[prediction[0]]
confidence = prediction_proba[0][prediction[0]] * 100

print(f"Predicted Species: {species_name}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nProbabilities for all classes:")
for i, species in enumerate(iris.target_names):
    print(f"  {species}: {prediction_proba[0][i]*100:.2f}%")
```

**Output:**
```
Predicted Species: setosa
Confidence: 100.00%

Probabilities for all classes:
  setosa: 100.00%
  versicolor: 0.00%
  virginica: 0.00%
```

### Batch Predictions

```python
# Predict on multiple flowers at once
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Flower 1
    [6.2, 3.4, 5.4, 2.3],  # Flower 2
    [7.2, 3.0, 5.8, 1.6]   # Flower 3
])

predictions = model.predict(new_flowers)
probabilities = model.predict_proba(new_flowers)

print("Batch Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    species = iris.target_names[pred]
    confidence = prob[pred] * 100
    print(f"Flower {i+1}: {species} (confidence: {confidence:.2f}%)")
```

---

## Complete Code

Here's the complete code in one script:

```python
# Complete First ML Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 2. Explore data
print("Dataset Info:")
print(df.info())
print("\nSummary:")
print(df.describe())

# 3. Prepare data
X = df[iris.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 6. Make predictions
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_flower)
print(f"Prediction: {iris.target_names[prediction[0]]}")

print("\n✓ First ML project complete!")
```

---

## Next Steps

### Try These Variations

1. **Try Different Algorithms**
   ```python
   # Compare multiple models
   from sklearn.linear_model import LogisticRegression
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.svm import SVC
   
   models = {
       'Logistic Regression': LogisticRegression(),
       'Decision Tree': DecisionTreeClassifier(),
       'SVM': SVC(),
       'Random Forest': RandomForestClassifier()
   }
   
   for name, model in models.items():
       model.fit(X_train, y_train)
       score = model.score(X_test, y_test)
       print(f"{name}: {score:.4f}")
   ```

2. **Visualize Decision Boundaries**
   - Plot how model separates classes
   - Use 2D projections for visualization

3. **Feature Engineering**
   - Create new features (ratios, products)
   - See if performance improves

4. **Hyperparameter Tuning**
   - Try different values for `n_estimators`, `max_depth`
   - Use GridSearchCV for automated tuning

---

## Common Issues and Solutions

### Issue 1: Low Accuracy

**Possible Causes:**
- Model too simple (underfitting)
- Not enough data
- Poor features

**Solutions:**
- Try more complex model
- Get more data
- Engineer better features

### Issue 2: Overfitting

**Signs:**
- High training accuracy, low test accuracy

**Solutions:**
- Reduce model complexity (lower max_depth)
- Get more training data
- Add regularization

### Issue 3: Data Leakage

**Signs:**
- Suspiciously high accuracy
- Test accuracy higher than training

**Solutions:**
- Ensure test set is truly unseen
- Don't use test set for feature selection
- Split data before any preprocessing

---

## Resources

- [Iris Dataset Documentation](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/index.html)

---

## Key Takeaways

1. **Follow the Workflow**: Problem → Data → Train → Evaluate → Predict
2. **Start Simple**: Begin with basic models, iterate
3. **Evaluate Properly**: Always use separate test set
4. **Understand Results**: Don't just look at accuracy
5. **Practice**: Try variations, experiment!

---

**Congratulations!** You've built your first ML model! 🎉

**Next**: Move to [03-supervised-learning-regression](../03-supervised-learning-regression/README.md) or [04-supervised-learning-classification](../04-supervised-learning-classification/README.md) to learn more algorithms!

