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

# Demo frame shaped like a Titanic extract. Swap in your local train table later.
rng = np.random.default_rng(42)
n = 400
train_df = pd.DataFrame({
    "Survived": rng.integers(0, 2, size=n),
    "Pclass": rng.integers(1, 4, size=n),
    "Sex": rng.choice(["male", "female"], size=n),
    "Age": rng.normal(30, 12, size=n).clip(1, 80),
    "SibSp": rng.integers(0, 4, size=n),
    "Parch": rng.integers(0, 3, size=n),
    "Fare": rng.normal(32, 20, size=n).clip(5, 250),
    "Embarked": rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1]),
})
# Inject a few missing values the way real Titanic CSVs do
train_df.loc[rng.choice(n, size=25, replace=False), "Age"] = np.nan
train_df.loc[rng.choice(n, size=3, replace=False), "Embarked"] = np.nan

print(train_df.info())
print(train_df.head())
```

---

## Step 2: Exploratory Data Analysis

```python
# Check missing values
print(train_df.isnull().sum())

# Visualize target distribution
sns.countplot(x="Survived", data=train_df)
plt.show()

# Explore numeric relationships
sns.heatmap(train_df.select_dtypes(include=[np.number]).corr(), annot=True)
plt.show()
```

---

## Step 3: Data Preprocessing

```python
# Handle missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Feature engineering
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)

# Encode categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df["Sex_encoded"] = le.fit_transform(train_df["Sex"])
```

---

## Step 4: Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare features
features = ["Pclass", "Sex_encoded", "Age", "Fare", "FamilySize", "IsAlone"]
X = train_df[features]
y = train_df["Survived"]

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

**Try next:** Open another project under [Module 16](../16-projects-beginner/README.md) or move to [Module 17 · Intermediate projects](../17-projects-intermediate/README.md).
