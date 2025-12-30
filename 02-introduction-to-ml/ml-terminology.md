# Machine Learning Terminology - Complete Reference

Comprehensive glossary of machine learning terms and concepts with detailed explanations and examples.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Data Terms](#data-terms)
- [Model Terms](#model-terms)
- [Training Terms](#training-terms)
- [Evaluation Terms](#evaluation-terms)
- [Algorithm Terms](#algorithm-terms)
- [Quick Reference](#quick-reference)

---

## Core Concepts

### Machine Learning (ML)

**Definition**: A method of data analysis that automates analytical model building, enabling computers to learn from data without being explicitly programmed.

**Key Idea**: Instead of writing rules, we show examples and let the computer learn patterns.

**Example:**
```python
# Traditional: Write rules
if email_contains("free money"):
    mark_as_spam()

# ML: Learn from examples
model.fit(spam_emails, labels)  # Learn what spam looks like
model.predict(new_email)  # Use learned knowledge
```

### Artificial Intelligence (AI)

**Definition**: The broader field of creating intelligent machines. ML is a subset of AI.

**Relationship:**
```
AI (Broad)
  └── Machine Learning
       └── Deep Learning
```

### Deep Learning

**Definition**: A subset of ML using neural networks with multiple layers to learn complex patterns.

**Example**: Image recognition, natural language processing, speech recognition.

---

## Data Terms

### Features (X)

**Definition**: Input variables used to make predictions. Also called:
- Predictors
- Independent variables
- Attributes
- Inputs

**Example:**
```python
# Features for house price prediction
features = ['size', 'bedrooms', 'location', 'age']
X = df[features]  # Feature matrix
```

### Labels/Targets (y)

**Definition**: The output variable we want to predict. Also called:
- Target variable
- Dependent variable
- Output
- Ground truth

**Example:**
```python
# Target for house price prediction
y = df['price']  # What we want to predict
```

### Training Data

**Definition**: Data used to teach the model. The model learns patterns from this data.

**Characteristics:**
- Typically 60-80% of total data
- Contains both features and labels (for supervised learning)
- Model sees this during training

**Example:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, y_train = Training data
```

### Validation Data

**Definition**: Data used to tune hyperparameters and select models during development.

**Characteristics:**
- Separate from training and test data
- Used to prevent overfitting
- Typically 10-20% of total data

### Test Data

**Definition**: Data used for final evaluation. Model has never seen this data.

**Characteristics:**
- Typically 10-20% of total data
- Simulates real-world performance
- Only used once at the end
- Never used for training or tuning

**Example:**
```python
# Test data - final evaluation only!
test_accuracy = model.score(X_test, y_test)
```

### Dataset

**Definition**: Collection of data used for ML. Can be:
- **Structured**: Tables (CSV, Excel)
- **Unstructured**: Images, text, audio
- **Semi-structured**: JSON, XML

### Sample/Instance

**Definition**: A single data point or row in the dataset.

**Example:**
```python
# One sample (one house)
sample = {
    'size': 1500,
    'bedrooms': 3,
    'location': 'suburb',
    'price': 250000
}
```

### Batch

**Definition**: A subset of training data processed together in one iteration.

**Why use batches:**
- Memory efficiency
- Faster training (parallel processing)
- More stable gradients

**Example:**
```python
# Process 32 samples at a time
batch_size = 32
for batch in batches:
    model.train(batch)
```

---

## Model Terms

### Model

**Definition**: The learned function that makes predictions. It's the "brain" that learned from data.

**Analogy**: Like a student who studied examples and can now answer new questions.

**Example:**
```python
# Model learns from data
model = LinearRegression()
model.fit(X_train, y_train)  # Learning phase

# Model makes predictions
predictions = model.predict(X_new)  # Prediction phase
```

### Algorithm

**Definition**: The method or procedure used to learn from data.

**Examples:**
- Linear Regression
- Decision Tree
- Random Forest
- Neural Network

**Note**: Algorithm is the method, Model is what it learned.

### Hyperparameters

**Definition**: Configuration settings for the algorithm (not learned from data).

**Examples:**
- Learning rate
- Number of trees in Random Forest
- Depth of Decision Tree
- Number of layers in Neural Network

**Key Point**: Set by the user, not learned by the model.

```python
# Hyperparameters (set by you)
model = RandomForestClassifier(
    n_estimators=100,  # Hyperparameter
    max_depth=10,      # Hyperparameter
    random_state=42    # Hyperparameter
)
```

### Parameters

**Definition**: Values learned by the model during training.

**Examples:**
- Weights in neural network
- Coefficients in linear regression
- Split points in decision tree

**Key Point**: Learned from data, not set by user.

```python
# Parameters (learned by model)
model.fit(X_train, y_train)
weights = model.coef_  # Learned parameters
```

### Weights

**Definition**: Parameters in a model that determine how features contribute to predictions.

**Example:**
```python
# Linear regression: y = w1*x1 + w2*x2 + b
# w1, w2 are weights
# b is bias
```

### Bias

**Definition**: A constant term added to predictions. Allows model to fit data that doesn't pass through origin.

**Example:**
```python
# y = mx + b
# b is the bias (intercept)
```

---

## Training Terms

### Training

**Definition**: The process of teaching the model by showing it examples and letting it learn patterns.

**Process:**
1. Initialize model
2. Show training examples
3. Model adjusts parameters
4. Repeat until model learns

**Example:**
```python
model.fit(X_train, y_train)  # Training phase
```

### Learning

**Definition**: The process of improving performance through experience (seeing more data).

### Epoch

**Definition**: One complete pass through the entire training dataset.

**Example:**
```python
# Train for 10 epochs
for epoch in range(10):
    model.train_one_epoch(training_data)
```

### Iteration

**Definition**: One update step of the model (processing one batch).

**Relationship:**
- 1 Epoch = Multiple iterations
- Iterations = Number of batches processed

### Loss Function / Cost Function

**Definition**: Function that measures how wrong the model's predictions are.

**Purpose**: Guides the model to improve by showing what's wrong.

**Examples:**
- **MSE** (Mean Squared Error): For regression
- **Cross-Entropy**: For classification
- **MAE** (Mean Absolute Error): For regression

```python
# Loss measures prediction error
predictions = model.predict(X_train)
loss = loss_function(predictions, y_train)
# Lower loss = better model
```

### Gradient

**Definition**: The direction and magnitude of steepest increase in loss. Used to update model parameters.

**In Training:**
- Calculate gradient
- Move opposite to gradient (reduce loss)
- Update parameters

### Gradient Descent

**Definition**: Optimization algorithm that finds minimum of loss function by following negative gradient.

**Process:**
1. Calculate gradient
2. Update parameters: `w = w - learning_rate * gradient`
3. Repeat until convergence

### Learning Rate

**Definition**: Step size in gradient descent. Controls how much parameters change each iteration.

**Too Small**: Slow convergence
**Too Large**: May overshoot or diverge
**Just Right**: Fast, stable convergence

```python
# Learning rate is a hyperparameter
model = SGDClassifier(learning_rate=0.01)
```

### Overfitting

**Definition**: Model memorizes training data but fails on new data.

**Signs:**
- High training accuracy
- Low test accuracy
- Large gap between train and test performance

**Example:**
```python
# Overfitting
train_accuracy = 0.99  # Perfect on training
test_accuracy = 0.60   # Poor on test
# Gap = 0.39 (overfitting!)
```

**Solutions:**
- More training data
- Simpler model
- Regularization
- Early stopping

### Underfitting

**Definition**: Model too simple to capture patterns in data.

**Signs:**
- Low training accuracy
- Low test accuracy
- Both are similar but both low

**Example:**
```python
# Underfitting
train_accuracy = 0.55  # Poor on training
test_accuracy = 0.53    # Poor on test
# Both low = underfitting
```

**Solutions:**
- More complex model
- Better features
- Remove regularization
- Train longer

### Generalization

**Definition**: Model's ability to perform well on new, unseen data.

**Goal**: Good generalization = model works in real world

**Measures:**
- Test accuracy
- Cross-validation score
- Performance on holdout set

### Regularization

**Definition**: Techniques to prevent overfitting by penalizing complex models.

**Types:**
- **L1 Regularization (Lasso)**: Penalizes sum of absolute weights
- **L2 Regularization (Ridge)**: Penalizes sum of squared weights
- **Dropout**: Randomly disable neurons (neural networks)

```python
# L2 regularization
model = Ridge(alpha=1.0)  # alpha controls regularization strength
```

---

## Evaluation Terms

### Accuracy

**Definition**: Proportion of correct predictions.

**Formula**: `Accuracy = (Correct Predictions) / (Total Predictions)`

**Use Case**: Balanced datasets

```python
accuracy = accuracy_score(y_true, y_pred)
# Example: 85 out of 100 correct = 0.85 accuracy
```

### Precision

**Definition**: Proportion of positive predictions that are actually positive.

**Formula**: `Precision = TP / (TP + FP)`

**Use Case**: When false positives are costly

**Example**: Spam detection - don't want to mark real emails as spam

### Recall / Sensitivity

**Definition**: Proportion of actual positives correctly identified.

**Formula**: `Recall = TP / (TP + FN)`

**Use Case**: When false negatives are costly

**Example**: Disease diagnosis - don't want to miss sick patients

### F1-Score

**Definition**: Harmonic mean of precision and recall. Balances both metrics.

**Formula**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**Use Case**: When need to balance precision and recall

### Confusion Matrix

**Definition**: Table showing performance of classification model.

**Structure:**
```
                Predicted
              Negative  Positive
Actual Negative   TN      FP
       Positive   FN      TP
```

**Components:**
- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive
- **FN** (False Negative): Incorrectly predicted negative

### ROC-AUC

**Definition**: Area Under the ROC Curve. Measures classifier's ability to distinguish classes.

**Range**: 0 to 1 (higher is better)
- 1.0: Perfect classifier
- 0.5: Random guessing
- <0.5: Worse than random

### MSE (Mean Squared Error)

**Definition**: Average squared difference between predictions and actual values.

**Formula**: `MSE = (1/n) * Σ(predicted - actual)²`

**Use Case**: Regression problems

### RMSE (Root Mean Squared Error)

**Definition**: Square root of MSE. In same units as target variable.

**Formula**: `RMSE = √MSE`

### MAE (Mean Absolute Error)

**Definition**: Average absolute difference between predictions and actual values.

**Formula**: `MAE = (1/n) * Σ|predicted - actual|`

### R² (R-squared / Coefficient of Determination)

**Definition**: Proportion of variance in target explained by model.

**Range**: -∞ to 1
- 1.0: Perfect fit
- 0.0: Model performs as well as mean
- <0: Model worse than mean

---

## Algorithm Terms

### Supervised Learning

**Definition**: Learning from labeled examples (input-output pairs).

**Types:**
- Regression: Predict continuous values
- Classification: Predict categories

### Unsupervised Learning

**Definition**: Learning from unlabeled data (only features, no labels).

**Types:**
- Clustering: Find groups
- Dimensionality Reduction: Reduce features

### Reinforcement Learning

**Definition**: Learning through interaction, receiving rewards/penalties.

### Ensemble

**Definition**: Combining multiple models to improve performance.

**Types:**
- **Bagging**: Train multiple models on different data subsets
- **Boosting**: Train models sequentially, each focuses on previous errors
- **Stacking**: Train meta-model on predictions of base models

### Cross-Validation

**Definition**: Technique to assess model performance by splitting data into folds.

**K-Fold CV**: Split data into k folds, train on k-1, test on 1, repeat k times.

**Benefits:**
- Better performance estimate
- Use all data for training and testing
- Reduce overfitting risk

---

## Quick Reference

### Common Abbreviations

- **ML**: Machine Learning
- **AI**: Artificial Intelligence
- **DL**: Deep Learning
- **NN**: Neural Network
- **CNN**: Convolutional Neural Network
- **RNN**: Recurrent Neural Network
- **SVM**: Support Vector Machine
- **KNN**: K-Nearest Neighbors
- **PCA**: Principal Component Analysis
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **ROC**: Receiver Operating Characteristic
- **AUC**: Area Under Curve
- **TP**: True Positive
- **TN**: True Negative
- **FP**: False Positive
- **FN**: False Negative

### Data Split Conventions

- **Training**: 60-80%
- **Validation**: 10-20%
- **Test**: 10-20%

### Model Performance Indicators

**Good Model:**
- Train and test accuracy close
- Test accuracy high
- Low loss on test set

**Overfitting:**
- Train accuracy >> Test accuracy
- Large gap between train/test

**Underfitting:**
- Train accuracy ≈ Test accuracy (both low)
- High loss on both

---

## Resources

- [ML Glossary in Resources](../resources/ml_glossary.md) - Comprehensive glossary
- [Scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html)

---

**Remember**: Understanding terminology helps you understand ML concepts and communicate effectively!

