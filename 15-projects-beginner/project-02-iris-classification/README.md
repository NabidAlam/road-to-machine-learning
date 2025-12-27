# Project 2: Iris Flower Classification

Classify iris flowers into three species using petal and sepal measurements.

## Project Overview

This is a classic machine learning project perfect for beginners. We'll use the famous Iris dataset to build a classification model that can identify iris species based on flower measurements.

## Learning Objectives

- Load and explore a dataset
- Perform exploratory data analysis (EDA)
- Split data into training and testing sets
- Train multiple classification models
- Evaluate and compare model performance
- Visualize results

## Dataset

The Iris dataset is built into scikit-learn and contains 150 samples of iris flowers with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Three species: Setosa, Versicolor, and Virginica

## Step-by-Step Guide

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
```

### Step 2: Load and Explore Data

```python
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())
print("\nSpecies Distribution:")
print(df['species'].value_counts())
```

### Step 3: Exploratory Data Analysis

```python
# Pair plot to see relationships between features
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle('Pair Plot of Iris Features', y=1.02)
plt.show()

# Box plots for each feature by species
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, feature in enumerate(feature_names):
    row = idx // 2
    col = idx % 2
    sns.boxplot(data=df, x='species', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Species')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.iloc[:, :4].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()
```

### Step 4: Prepare Data for Modeling

```python
# Separate features and target
X = df.iloc[:, :4].values
y = df['species'].cat.codes.values

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
```

### Step 5: Train Multiple Models

```python
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}")
```

### Step 6: Compare Models

```python
# Visualize model comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Comparison - Accuracy Scores', fontsize=14, fontweight='bold')
plt.ylim([0.9, 1.0])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Step 7: Confusion Matrix for Best Model

```python
# Get the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model_name]['predictions']

# Create confusion matrix
cm = confusion_matrix(y_test, best_predictions)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()
```

### Step 8: Make Predictions on New Data

```python
# Example: Predict species for new measurements
new_measurements = np.array([[5.1, 3.5, 1.4, 0.2],  # Setosa
                             [6.2, 3.4, 5.4, 2.3],  # Virginica
                             [5.9, 3.0, 4.2, 1.5]]) # Versicolor

best_model = results[best_model_name]['model']
predictions = best_model.predict(new_measurements)

print("Predictions for new measurements:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {target_names[pred]}")
```

## Expected Results

- All three models should achieve high accuracy (>95%)
- Random Forest typically performs best on this dataset
- The dataset is well-balanced, so accuracy is a good metric

## Key Takeaways

1. **Data Exploration is Crucial**: Understanding your data before modeling helps identify patterns and potential issues
2. **Multiple Models**: Always try multiple algorithms - different models work better for different datasets
3. **Visualization Helps**: Plots make it easier to understand data and model performance
4. **Simple Datasets First**: Starting with well-known datasets like Iris helps build confidence

## Next Steps

- Try feature engineering (create new features from existing ones)
- Experiment with different train/test splits
- Try other classification algorithms (SVM, KNN, Naive Bayes)
- Add cross-validation for more robust evaluation
- Deploy the model as a simple web app

## Resources

- [Iris Dataset Documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#classification)

---

**Happy Learning!** Complete this project and move on to more challenging ones!

