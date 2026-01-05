# Complete Beginner Project Tutorial

Step-by-step walkthrough of building a complete ML project from scratch.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Setup and Data Loading](#step-1-setup-and-data-loading)
- [Step 2: Exploratory Data Analysis](#step-2-exploratory-data-analysis)
- [Step 3: Data Preprocessing](#step-3-data-preprocessing)
- [Step 4: Model Training](#step-4-model-training)
- [Step 5: Evaluation and Improvement](#step-5-evaluation-and-improvement)

---

## Project Overview

**Project**: Titanic Survival Prediction

**Task**: Classification

**Goal**: Predict passenger survival

---

## Step 1: Setup and Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.info())
print(train_df.head())
```

---

## Step 2: Exploratory Data Analysis

```python
# Check missing values
print(train_df.isnull().sum())

# Visualize target distribution
sns.countplot(x='Survived', data=train_df)
plt.show()

# Explore relationships
sns.heatmap(train_df.corr(), annot=True)
plt.show()
```

---

## Step 3: Data Preprocessing

```python
# Handle missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Feature engineering
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

# Encode categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Sex_encoded'] = le.fit_transform(train_df['Sex'])
```

---

## Step 4: Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare features
features = ['Pclass', 'Sex_encoded', 'Age', 'Fare', 'FamilySize', 'IsAlone']
X = train_df[features]
y = train_df['Survived']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)
```

---

## Step 5: Evaluation and Improvement

```python
from sklearn.metrics import accuracy_score, classification_report

# Evaluate
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_val, y_pred))

# Feature importance
importances = model.feature_importances_
plt.barh(features, importances)
plt.show()
```

---

**Congratulations!** You've completed your first ML project!

