# Machine Learning Glossary

A plain-language glossary of core ML terms for beginners and practitioners. Use it when a module drops jargon and you need a clear meaning fast.

For LLM, agents, RAG, and serving terms, see the [AI Engineering Glossary](ai_engineering_glossary.md). For common wrong mental models, see [AI Myths Busted](ai_myths_busted.md).

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
- [Quick Reference](#quick-reference)

---

## A

**Accuracy**  
Share of predictions that are correct: (True Positives + True Negatives) / Total. It is easy to read, but it can look great on imbalanced data even when the model is weak on the rare class.

**Activation Function**  
A function applied to a neuron’s output. ReLU, Sigmoid, and Tanh are common choices. Without some nonlinearity, stacked layers stay linear and cannot learn richer patterns.

**AdaBoost**  
Adaptive Boosting. It builds a strong model from many weak ones and puts more weight on examples that earlier rounds got wrong.

**Algorithm**  
A clear set of steps for solving a problem. In ML. It is the method used to learn patterns from data.

**Anomaly Detection**  
Finding rare items that look different from the rest. Common in fraud detection, network security, and quality control.

**API (Application Programming Interface)**  
A defined way for software to talk to other software. In ML, this often means a REST endpoint that returns predictions.

**ARIMA (AutoRegressive Integrated Moving Average)**  
A classic statistical model for time series. It combines autoregression, differencing, and a moving-average term.

**Artificial Intelligence (AI)**  
Machines that perform tasks we usually associate with human intelligence. Machine learning is one part of that broader field.

**Attention Mechanism**  
Lets a model weigh which parts of the input matter most right now. It is central to Transformers and shows up in NLP and computer vision.

**AutoML**  
Automated Machine Learning. Tools that search model choices and hyperparameters for you so less of the pipeline is hand-tuned.

---

## B

**Backpropagation**  
The training method that sends error signals backward through a neural network and computes gradients so weights can be updated.

**Bagging (Bootstrap Aggregating)**  
Train several models on different resampled subsets, then average or vote. Random Forest is the usual example.

**Batch**  
A chunk of training examples used in one update step. Small batches update more often and look noisier. Large batches are steadier but need more memory.

**Bias**  
A systematic lean in predictions. High bias often means the model is too simple and underfits. In ethics, bias also means unfair treatment of groups.

**Bias-Variance Tradeoff**  
The classic balance in ML. Bias comes from assumptions that are too simple. Variance comes from fitting noise too closely. Good models keep both in check.

**Big Data**  
Very large datasets that need special storage and processing. People often talk about volume, velocity, variety, and veracity.

**Boosting**  
Train models in sequence so each new model focuses on what earlier ones missed. AdaBoost, Gradient Boosting, and XGBoost are common forms.

**Bootstrapping**  
Resample with replacement to create many datasets from one. Used in bagging and some uncertainty estimates.

---

## C

**Categorical Variable**  
A feature with a limited set of labels, such as color or country. Models usually need encoding (one-hot or label encoding) before they can use it.

**Classification**  
Supervised learning that predicts discrete classes, such as spam vs not spam. Common metrics include accuracy, precision, recall, and F1.

**Clustering**  
Unsupervised grouping of similar points. Useful for customer segments or compression. K-Means, DBSCAN, and hierarchical methods are typical tools.

**CNN (Convolutional Neural Network)**  
A deep network built for grid-like data such as images. Convolution layers pick up local patterns for tasks like classification and detection.

**Confusion Matrix**  
A table of actual vs predicted classes. It shows true positives, true negatives, false positives, and false negatives in one place.

**Cross-Validation**  
Split data into folds, train on most folds, and test on the held-out fold. It gives a more honest performance estimate and helps catch overfitting.

**Curse of Dimensionality**  
In high dimensions, data gets sparse and distances start looking alike. That is why people use dimensionality reduction.

---

## D

**Data Augmentation**  
Create extra training examples by changing existing ones, such as rotating images or paraphrasing text. It helps the model generalize and can reduce overfitting.

**Data Leakage**  
Test or future information sneaks into training. Scores look amazing and then fail in the real world. Guard against it carefully.

**Data Preprocessing**  
Clean and transform raw data before modeling. Typical steps include missing values, encoding, scaling, and normalization.

**Decision Tree**  
A tree of feature splits for classification or regression. Easy to read, but a single deep tree can overfit.

**Deep Learning**  
ML with multi-layer neural networks that learn features from data. It can model complex patterns, but it usually wants lots of data and compute.

**Dimensionality Reduction**  
Shrink the feature space while keeping useful signal. PCA, t-SNE, and autoencoders are common options. Benefits include speed, clearer plots, and less noise.

**Dropout**  
During training, randomly turn off some neurons. That stops the network from relying on a few paths and helps fight overfitting.

---

## E

**Epoch**  
One full pass over the training set. Models usually need several epochs. Too many can tip into overfitting.

**Ensemble Learning**  
Combine several models so the group beats any one of them. Bagging, boosting, stacking, and voting are the main styles.

**Evaluation Metrics**  
Numbers that say how good the model is. Classification often uses accuracy, precision, recall, F1, and ROC-AUC. Regression often uses MSE, RMSE, MAE, and R².

**Exploratory Data Analysis (EDA)**  
Look at the data first with plots and summaries. This is how you learn the shape of the problem before you fit models.

---

## F

**False Negative (FN)**  
The model says negative, but the true label is positive. Example: calling a spam email “not spam.”

**False Positive (FP)**  
The model says positive, but the true label is negative. Example: flagging a normal email as spam.

**Feature**  
One measurable input property. Also called a variable, attribute, or predictor.

**Feature Engineering**  
Build better inputs from what you already have, such as ratios, polynomials, or time features. This step often moves the needle more than swapping algorithms.

**Feature Importance**  
How much each feature drives predictions. People use permutation importance, SHAP, or tree-based scores to explain the model.

**Feature Selection**  
Keep the useful features and drop the rest. Training gets faster, overfitting risk drops, and the model is easier to explain.

**F1-Score**  
The harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall). Use it when you care about both sides of the tradeoff.

**Fine-tuning**  
Start from a pre-trained model and adapt it to your task. It is usually faster than training from scratch and is common in transfer learning.

---

## G

**Gradient**  
A vector of partial derivatives that points toward the steepest increase. Optimization steps usually move opposite to it.

**Gradient Boosting**  
Boosting that follows the loss gradient at each step. XGBoost, LightGBM, and CatBoost are strong on tabular data.

**Gradient Descent**  
Move parameters opposite the gradient to reduce loss. Variants include SGD, mini-batch GD, and Adam.

**Grid Search**  
Try every combination in a hyperparameter grid. Thorough, but expensive when the grid is large.

---

## H

**Hyperparameter**  
A setting you choose before training, such as learning rate or tree count. It is not learned as a weight. Tune it on validation data or with cross-validation.

**Hyperparameter Tuning**  
Search for better hyperparameter values with grid search, random search, or Bayesian optimization.

---

## I

**Imbalanced Dataset**  
Classes are uneven, such as 99% negative and 1% positive. Use tools like SMOTE, class weights, and metrics that respect the rare class.

**Inference**  
Run a trained model on new data. Weights stay fixed. This is prediction time, not training time.

**Instance**  
One example or row. Also called a sample or observation.

---

## J

**Jaccard Similarity**  
Overlap between two sets: |A ∩ B| / |A ∪ B|. Useful for recommendations, clustering checks, and some NLP features.

---

## K

**K-Fold Cross-Validation**  
Split into k folds and rotate which fold is validation. You train k times and average the results.

**K-Means**  
Cluster points into k groups by reducing within-cluster variance.

**KNN (K-Nearest Neighbors)**  
Predict from the k closest training examples. Simple and strong as a baseline, but it can get slow on large datasets.

---

## L

**Label**  
The answer you want the model to predict. Also called the target, output, or dependent variable.

**Learning Rate**  
How large each optimization step is. Too high and you overshoot. Too low and training crawls.

**Linear Regression**  
Predict a continuous target with a linear relationship. Simple, readable, and a solid baseline.

**Logistic Regression**  
Predict class probabilities with a logistic function. The name says regression, but it is mainly used for classification.

**Loss Function**  
A score for how far predictions are from the truth. MSE is common for regression. Cross-entropy is common for classification. Training tries to minimize it.

**LSTM (Long Short-Term Memory)**  
An RNN design that keeps long-range context better and reduces vanishing gradients. Used for text and time series.

---

## M

**Machine Learning (ML)**  
Systems that improve from data rather than only from hand-written rules. Main branches include supervised, unsupervised, and reinforcement learning.

**Mean Squared Error (MSE)**  
Average of squared prediction errors: Σ(predicted − actual)² / n. Large mistakes get punished more.

**Model**  
The function learned from data that maps features to predictions. Linear regression, trees, and neural nets are all models.

**Multiclass Classification**  
More than two classes, such as ten image labels. Metrics and encoding choices differ from binary problems.

---

## N

**Neural Network**  
Layers of connected units inspired by biological neurons. With enough capacity and data, they can learn complex functions.

**Normalization**  
Scale features into a fixed range, often 0 to 1. Different from z-score standardization. It often helps optimization.

**NumPy**  
The core Python library for arrays and numerical work. Many ML libraries sit on top of it.

---

## O

**One-Hot Encoding**  
Turn categories into binary columns. Red, Blue, Green become three flags such as [1,0,0], [0,1,0], [0,0,1].

**Overfitting**  
The model memorizes training quirks and then fails on new data. Regularization, more data, and simpler models are common fixes.

---

## P

**Pandas**  
Python library for tables and data wrangling. DataFrames are the everyday tool for preprocessing.

**Parameter**  
A value the model learns during training, such as a weight or coefficient. Not the same as a hyperparameter.

**PCA (Principal Component Analysis)**  
A linear way to reduce dimensions by keeping directions with the most variance.

**Precision**  
Of the predicted positives, how many were right: TP / (TP + FP). Matter most when false alarms are expensive.

**Preprocessing**  
Get data ready for modeling through cleaning, encoding, scaling, and feature work.

---

## Q

**Quantization**  
Store weights or activations in lower precision such as FP16 or INT8. Inference gets faster and lighter, which helps production and edge devices.

---

## R

**Recall**  
Of the real positives, how many you caught: TP / (TP + FN). Matter most when misses are expensive.

**Regression**  
Predict continuous values such as price or temperature. Common metrics are MSE, RMSE, MAE, and R².

**Regularization**  
Add pressure against overly complex fits so the model generalizes better. L1, L2, and dropout are common forms.

**Reinforcement Learning**  
An agent learns by acting in an environment and collecting rewards or penalties. Used in games and robotics.

**ROC-AUC**  
Area under the ROC curve. It summarizes how well a classifier ranks positives above negatives. Higher is better, with 1.0 as perfect.

**RNN (Recurrent Neural Network)**  
A network with memory of earlier steps, built for sequences such as text, time series, and speech.

---

## S

**Scikit-learn**  
A widely used Python ML library with solid algorithms and utilities. Friendly for classical ML workflows.

**Supervised Learning**  
Learn from labeled examples. Classification and regression are the two main task types.

**SVM (Support Vector Machine)**  
Find a boundary that separates classes with a strong margin. Often useful in high-dimensional spaces.

---

## T

**Test Set**  
Held-out data for the final score. Do not tune on it. It should look like the real deployment data.

**Time Series**  
Observations ordered in time, such as prices or sensor readings. Models need to respect order and seasonality.

**Training Set**  
The data the model learns from. Often about 60% to 80% of the full set.

**Transfer Learning**  
Reuse knowledge from one task on another. A common case is adapting an ImageNet model to medical images.

**Transformer**  
An architecture built on attention. It changed NLP with models like BERT and GPT and can process sequence positions in parallel.

**True Negative (TN)**  
Correct negative prediction. Example: “not spam,” and it really is not spam.

**True Positive (TP)**  
Correct positive prediction. Example: “spam,” and it really is spam.

---

## U

**Underfitting**  
The model is too weak to capture the pattern. Training and test scores both stay poor. Try a richer model, better features, or less regularization.

**Unsupervised Learning**  
Learn structure without labels. Clustering and dimensionality reduction are the usual starting points.

---

## V

**Validation Set**  
Data used to tune hyperparameters and pick models. Keep it separate from train and test. Often about 10% to 20% of the data.

**Variance**  
How much the model jumps when the training sample changes. High variance lines up with overfitting. Very low variance can signal underfitting.

---

## W

**Word Embedding**  
A dense vector for a word that reflects meaning and neighborhood. Word2Vec, GloVe, and contextual embeddings from BERT are common examples.

---

## X

**XGBoost**  
Extreme Gradient Boosting. A fast, strong library for tabular problems and a regular top performer in competitions.

---

## Y

**YOLO (You Only Look Once)**  
A family of real-time object detectors. Boxes and classes come from one forward pass, which helps when latency matters.

---

## Z

**Z-score (Standard Score)**  
How many standard deviations a value sits from the mean: (x − μ) / σ. Used for outliers and feature standardization.

---

## Quick Reference

### Common Metrics

**Classification**
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)

**Regression**
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R²: Coefficient of Determination

### Data Splits

- **Training:** about 60% to 80%, used to learn patterns
- **Validation:** about 10% to 20%, used to tune choices
- **Test:** about 10% to 20%, used for the final check

### Common Algorithms

**Supervised:** Linear or logistic regression, decision trees, random forest, SVM, KNN, neural networks

**Unsupervised:** K-Means, DBSCAN, PCA, t-SNE

### Acronyms

API, BERT, CNN, EDA, FN, FP, GDPR, GPU, LSTM, MLOps, NLP, PCA, ROC, RNN, SGD, SVM, TN, TP

---

This glossary grows with the curriculum. For deeper walkthroughs, open the matching module guides in this repository.
