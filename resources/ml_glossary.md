# Machine Learning Glossary

Comprehensive glossary of machine learning terms and concepts for beginners and practitioners.

## Table of Contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [J](#j)
- [K](#k)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [Q](#q)
- [R](#r)
- [S](#s)
- [T](#t)
- [U](#u)
- [V](#v)
- [W](#w)
- [X](#x)
- [Y](#y)
- [Z](#z)

---

## A

**Accuracy**
- The proportion of correct predictions made by a model
- Formula: (True Positives + True Negatives) / Total Predictions
- Note: Can be misleading with imbalanced datasets

**Activation Function**
- A function applied to the output of a neuron in a neural network
- Examples: ReLU, Sigmoid, Tanh
- Purpose: Introduces non-linearity into the model

**AdaBoost**
- Adaptive Boosting algorithm
- Combines multiple weak learners into a strong learner
- Weights misclassified examples more heavily in subsequent iterations

**Algorithm**
- A set of rules or instructions for solving a problem
- In ML: A method for learning patterns from data

**Anomaly Detection**
- Identifying rare items or events that differ significantly from the majority
- Applications: Fraud detection, network security, quality control

**API (Application Programming Interface)**
- A set of protocols for building software applications
- In ML: Interface for accessing model predictions (e.g., REST API)

**ARIMA (AutoRegressive Integrated Moving Average)**
- A statistical model for time series forecasting
- Components: AR (autoregressive), I (integrated), MA (moving average)

**Artificial Intelligence (AI)**
- The simulation of human intelligence by machines
- ML is a subset of AI

**Attention Mechanism**
- Allows models to focus on relevant parts of input
- Key component of Transformer architectures
- Used in NLP and computer vision

**AutoML**
- Automated Machine Learning
- Automates the process of model selection and hyperparameter tuning

---

## B

**Backpropagation**
- Algorithm for training neural networks
- Calculates gradients by propagating errors backward through the network
- Enables gradient descent optimization

**Bagging (Bootstrap Aggregating)**
- Ensemble method that trains multiple models on different subsets of data
- Predictions are averaged (regression) or voted (classification)
- Example: Random Forest

**Batch**
- A subset of training data used in one iteration
- Batch size: Number of samples processed before updating model weights
- Smaller batches: More updates, more noise
- Larger batches: Fewer updates, more stable

**Bias**
- Systematic error in model predictions
- High bias: Model is too simple (underfitting)
- In ethics: Unfair treatment of certain groups

**Bias-Variance Tradeoff**
- Fundamental problem in ML
- Bias: Error from oversimplifying assumptions
- Variance: Error from sensitivity to small fluctuations
- Goal: Balance both

**Big Data**
- Extremely large datasets that require special tools and techniques
- Characteristics: Volume, Velocity, Variety, Veracity

**Boosting**
- Ensemble method that trains models sequentially
- Each model focuses on examples previous models got wrong
- Examples: AdaBoost, Gradient Boosting, XGBoost

**Bootstrapping**
- Statistical resampling technique
- Creates multiple datasets by sampling with replacement
- Used in bagging and cross-validation

---

## C

**Categorical Variable**
- Variable that takes on a limited set of values
- Examples: Color (red, blue, green), Country, Gender
- Requires encoding (one-hot, label encoding) for ML models

**Classification**
- Supervised learning task of predicting discrete categories
- Examples: Spam/Not Spam, Image categories, Disease diagnosis
- Metrics: Accuracy, Precision, Recall, F1-Score

**Clustering**
- Unsupervised learning task of grouping similar data points
- Examples: Customer segmentation, Image compression
- Algorithms: K-Means, DBSCAN, Hierarchical

**CNN (Convolutional Neural Network)**
- Deep learning architecture for image data
- Uses convolutional layers to detect patterns
- Applications: Image classification, object detection

**Confusion Matrix**
- Table showing performance of a classification model
- Rows: Actual classes, Columns: Predicted classes
- Contains: True Positives, True Negatives, False Positives, False Negatives

**Cross-Validation**
- Technique for assessing model performance
- Splits data into k folds, trains on k-1, tests on 1
- Reduces overfitting risk, provides better performance estimate

**Curse of Dimensionality**
- Problems that arise when working with high-dimensional data
- Data becomes sparse, distances become similar
- Requires dimensionality reduction techniques

---

## D

**Data Augmentation**
- Technique to increase training data by creating modified versions
- Examples: Rotating images, adding noise, translating text
- Helps prevent overfitting

**Data Leakage**
- When information from test set leaks into training
- Results in overly optimistic performance estimates
- Must be carefully avoided

**Data Preprocessing**
- Cleaning and transforming raw data before modeling
- Steps: Handling missing values, encoding, scaling, normalization

**Decision Tree**
- Tree-like model for classification and regression
- Splits data based on feature values
- Easy to interpret, prone to overfitting

**Deep Learning**
- Subset of ML using neural networks with multiple layers
- Can learn complex patterns automatically
- Requires large amounts of data and computation

**Dimensionality Reduction**
- Reducing number of features while preserving information
- Methods: PCA, t-SNE, Autoencoders
- Benefits: Faster training, visualization, noise reduction

**Dropout**
- Regularization technique for neural networks
- Randomly sets some neurons to zero during training
- Prevents overfitting

---

## E

**Epoch**
- One complete pass through the entire training dataset
- Multiple epochs needed for model to learn
- Too many epochs can cause overfitting

**Ensemble Learning**
- Combining multiple models to improve performance
- Methods: Bagging, Boosting, Stacking, Voting
- Generally performs better than individual models

**Evaluation Metrics**
- Measures used to assess model performance
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: MSE, RMSE, MAE, R²

**Exploratory Data Analysis (EDA)**
- Initial investigation of data to discover patterns
- Uses visualizations and summary statistics
- Helps understand data before modeling

---

## F

**False Negative (FN)**
- Case where model predicts negative but actual is positive
- Example: Model says "not spam" but email is spam

**False Positive (FP)**
- Case where model predicts positive but actual is negative
- Example: Model says "spam" but email is not spam

**Feature**
- Individual measurable property of a phenomenon
- Also called: Variable, Attribute, Predictor
- Input to ML models

**Feature Engineering**
- Creating new features from existing ones
- Examples: Creating ratios, polynomial features, time-based features
- Critical for model performance

**Feature Importance**
- Measure of how much a feature contributes to predictions
- Methods: Permutation importance, SHAP values, tree-based importance
- Helps understand model behavior

**Feature Selection**
- Choosing subset of relevant features
- Benefits: Faster training, reduced overfitting, better interpretability
- Methods: Filter methods, wrapper methods, embedded methods

**F1-Score**
- Harmonic mean of precision and recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Balances precision and recall

**Fine-tuning**
- Adjusting a pre-trained model for a specific task
- Common in transfer learning
- Faster than training from scratch

---

## G

**Gradient**
- Vector of partial derivatives
- Points in direction of steepest ascent
- Used in optimization (gradient descent)

**Gradient Boosting**
- Boosting algorithm that minimizes loss function using gradients
- Examples: XGBoost, LightGBM, CatBoost
- Very powerful for structured data

**Gradient Descent**
- Optimization algorithm for finding minimum of a function
- Updates parameters in direction opposite to gradient
- Variants: SGD, Mini-batch GD, Adam

**Grid Search**
- Hyperparameter tuning method
- Tests all combinations of specified hyperparameter values
- Exhaustive but computationally expensive

---

## H

**Hyperparameter**
- Configuration settings for a model (not learned from data)
- Examples: Learning rate, number of trees, regularization strength
- Tuned using validation set or cross-validation

**Hyperparameter Tuning**
- Process of finding optimal hyperparameter values
- Methods: Grid Search, Random Search, Bayesian Optimization

---

## I

**Imbalanced Dataset**
- Dataset where classes are not equally represented
- Example: 99% negative, 1% positive
- Requires special techniques: SMOTE, class weights, different metrics

**Inference**
- Using a trained model to make predictions on new data
- Different from training (no weight updates)

**Instance**
- Single data point or example
- Also called: Sample, Observation, Row

---

## J

**Jaccard Similarity**
- Measures similarity between two sets
- Formula: \(|A ∩ B| / |A ∪ B|\)
- Common use: evaluating overlap in recommendations, clustering, and NLP set-based features

---

## K

**K-Fold Cross-Validation**
- Cross-validation with k folds
- Data split into k subsets, model trained k times
- Each fold used once as validation set

**K-Means**
- Clustering algorithm
- Groups data into k clusters
- Minimizes within-cluster variance

**KNN (K-Nearest Neighbors)**
- Instance-based learning algorithm
- Predicts based on k nearest training examples
- Simple but can be slow for large datasets

---

## L

**Label**
- The target variable or correct answer
- In supervised learning: What we're trying to predict
- Also called: Target, Output, Dependent Variable

**Learning Rate**
- Hyperparameter controlling step size in optimization
- Too high: May overshoot minimum
- Too low: Slow convergence

**Linear Regression**
- Algorithm for predicting continuous values
- Assumes linear relationship between features and target
- Simple, interpretable baseline

**Logistic Regression**
- Algorithm for binary classification
- Uses logistic function to output probabilities
- Despite name, it's a classification algorithm

**Loss Function**
- Function measuring how far predictions are from actual values
- Examples: MSE (regression), Cross-entropy (classification)
- Minimized during training

**LSTM (Long Short-Term Memory)**
- Type of RNN that can learn long-term dependencies
- Solves vanishing gradient problem
- Used for sequences: text, time series

---

## M

**Machine Learning (ML)**
- Subset of AI where systems learn from data
- Improves performance with experience
- Types: Supervised, Unsupervised, Reinforcement

**Mean Squared Error (MSE)**
- Average squared difference between predictions and actual values
- Formula: Σ(predicted - actual)² / n
- Penalizes large errors more

**Model**
- Mathematical representation learned from data
- Makes predictions based on input features
- Examples: Linear regression, Neural network, Decision tree

**Multiclass Classification**
- Classification with more than two classes
- Example: Classifying images into 10 categories
- Requires different metrics than binary classification

---

## N

**Neural Network**
- Computing system inspired by biological neural networks
- Consists of interconnected nodes (neurons)
- Can learn complex patterns

**Normalization**
- Scaling features to a standard range (usually 0-1)
- Different from standardization (z-score)
- Helps with gradient descent convergence

**NumPy**
- Python library for numerical computing
- Provides arrays and mathematical operations
- Foundation for many ML libraries

---

## O

**One-Hot Encoding**
- Converting categorical variables to binary vectors
- Each category becomes a binary feature
- Example: [Red, Blue, Green] → [1,0,0], [0,1,0], [0,0,1]

**Overfitting**
- Model learns training data too well, including noise
- Performs well on training but poorly on test data
- Solutions: Regularization, more data, simpler models

---

## P

**Pandas**
- Python library for data manipulation and analysis
- Provides DataFrames (like Excel tables)
- Essential for data preprocessing

**Parameter**
- Values learned by model during training
- Examples: Weights in neural network, coefficients in linear regression
- Different from hyperparameters

**PCA (Principal Component Analysis)**
- Dimensionality reduction technique
- Finds directions of maximum variance
- Reduces features while preserving information

**Precision**
- Proportion of positive predictions that are correct
- Formula: True Positives / (True Positives + False Positives)
- Important when false positives are costly

**Preprocessing**
- Preparing data before modeling
- Steps: Cleaning, encoding, scaling, feature engineering

---

## Q

**Quantization**
- Compressing model weights/activations to lower precision (e.g., FP16, INT8)
- Benefits: Faster inference and lower memory usage
- Common in production deployment and edge/CPU serving

---

## R

**Recall**
- Proportion of actual positives correctly identified
- Formula: True Positives / (True Positives + False Negatives)
- Important when false negatives are costly

**Regression**
- Supervised learning task of predicting continuous values
- Examples: House prices, temperature, sales
- Metrics: MSE, RMSE, MAE, R²

**Regularization**
- Technique to prevent overfitting
- Adds penalty for complex models
- Types: L1 (Lasso), L2 (Ridge), Dropout

**Reinforcement Learning**
- Learning through interaction with environment
- Agent receives rewards/penalties
- Applications: Game playing, robotics

**ROC-AUC**
- Area Under the ROC Curve
- Measures classifier's ability to distinguish classes
- Range: 0 to 1 (higher is better)

**RNN (Recurrent Neural Network)**
- Neural network for sequential data
- Has memory of previous inputs
- Used for: Text, time series, speech

---

## S

**Scikit-learn**
- Popular Python ML library
- Provides many algorithms and utilities
- Easy to use, well-documented

**Supervised Learning**
- Learning from labeled examples
- Types: Classification, Regression
- Requires labeled training data

**SVM (Support Vector Machine)**
- Classification algorithm
- Finds optimal boundary between classes
- Effective for high-dimensional data

---

## T

**Test Set**
- Data used to evaluate final model performance
- Not used during training or validation
- Should represent real-world data

**Time Series**
- Data points collected over time
- Examples: Stock prices, temperature, sales
- Requires special techniques (ARIMA, LSTM)

**Training Set**
- Data used to train the model
- Model learns patterns from this data
- Typically 60-80% of total data

**Transfer Learning**
- Using knowledge from one task for another
- Common in deep learning
- Example: Using ImageNet model for medical images

**Transformer**
- Neural network architecture using attention
- Revolutionized NLP (BERT, GPT)
- Can process sequences in parallel

**True Negative (TN)**
- Correctly predicted negative case
- Example: Model says "not spam" and email is not spam

**True Positive (TP)**
- Correctly predicted positive case
- Example: Model says "spam" and email is spam

---

## U

**Underfitting**
- Model is too simple to capture patterns
- Performs poorly on both training and test data
- Solutions: More complex model, more features, less regularization

**Unsupervised Learning**
- Learning from unlabeled data
- Types: Clustering, Dimensionality Reduction
- Finds hidden patterns

---

## V

**Validation Set**
- Data used to tune hyperparameters and select models
- Separate from training and test sets
- Typically 10-20% of total data

**Variance**
- Model's sensitivity to small fluctuations in training data
- High variance: Overfitting
- Low variance: Underfitting

---

## W

**Word Embedding**
- Dense vector representation of words
- Captures semantic relationships
- Examples: Word2Vec, GloVe, BERT embeddings

---

## X

**XGBoost**
- Extreme Gradient Boosting
- Powerful gradient boosting library
- Excellent performance on structured data
- Used in many Kaggle competitions

---

## Y

**YOLO (You Only Look Once)**
- Real-time object detection model family
- Predicts bounding boxes and class probabilities in one forward pass
- Common use: computer vision deployment where latency matters

---

## Z

**Z-score (Standard Score)**
- Number of standard deviations a value is from the mean
- Formula: \((x - μ) / σ\)
- Common use: outlier detection and feature standardization

---

## Additional Terms

**API**: Application Programming Interface

**BERT**: Bidirectional Encoder Representations from Transformers

**CNN**: Convolutional Neural Network

**EDA**: Exploratory Data Analysis

**FN**: False Negative

**FP**: False Positive

**GDPR**: General Data Protection Regulation

**GPU**: Graphics Processing Unit (used for deep learning)

**LSTM**: Long Short-Term Memory

**MLOps**: Machine Learning Operations

**NLP**: Natural Language Processing

**PCA**: Principal Component Analysis

**ROC**: Receiver Operating Characteristic

**RNN**: Recurrent Neural Network

**SGD**: Stochastic Gradient Descent

**SVM**: Support Vector Machine

**TN**: True Negative

**TP**: True Positive

---

## Quick Reference

### Common Metrics

**Classification:**
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)

**Regression:**
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of Determination

### Data Splits

- **Training**: 60-80% (learn patterns)
- **Validation**: 10-20% (tune hyperparameters)
- **Test**: 10-20% (final evaluation)

### Common Algorithms

**Supervised:**
- Linear/Logistic Regression
- Decision Trees
- Random Forest
- SVM
- KNN
- Neural Networks

**Unsupervised:**
- K-Means
- DBSCAN
- PCA
- t-SNE

---

**Note**: This glossary is a living document. Terms are added and updated as the field evolves. For detailed explanations, refer to the specific module guides in this repository.

