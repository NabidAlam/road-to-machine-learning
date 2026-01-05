# Beginner Projects Quick Reference Guide

Quick reference for building ML projects.

## Table of Contents

- [Project Workflow](#project-workflow)
- [Code Snippets](#code-snippets)
- [Common Tasks](#common-tasks)
- [Best Practices Checklist](#best-practices-checklist)

---

## Project Workflow

1. Problem Definition
2. Data Collection
3. EDA
4. Preprocessing
5. Model Training
6. Evaluation
7. Improvement

---

## Code Snippets

### Load Data

```python
df = pd.read_csv('data.csv')
print(df.info())
```

### Handle Missing Values

```python
df['column'].fillna(df['column'].median(), inplace=True)
```

### Train Model

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

---

## Common Tasks

### Classification

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```

### Regression

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

---

## Best Practices Checklist

- [ ] Do EDA first
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Split train/test properly
- [ ] Try multiple models
- [ ] Evaluate with appropriate metrics
- [ ] Document your work

---

**Remember**: Start simple and iterate!

