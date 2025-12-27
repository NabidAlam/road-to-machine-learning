# Project 4: Spam Email Detection

Classify emails as spam or not spam using text features.

## Difficulty
Beginner

## Time Estimate
2-3 days

## Skills You'll Practice
- Text Classification
- NLP Basics
- Feature Engineering from Text
- Text Preprocessing

## Learning Objectives

By completing this project, you will learn to:
- Preprocess text data
- Extract features from text
- Apply classification to text data
- Handle imbalanced text datasets
- Evaluate text classification models

## Dataset

**Option 1: SMS Spam Collection Dataset**
- [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Simple CSV format
- Two columns: label (ham/spam) and text

**Option 2: Email Spam Dataset**
- [Enron Spam Dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam)
- More complex, real-world emails

## Project Steps

### Step 1: Load and Explore Data
- Load the dataset
- Check class distribution (spam vs ham)
- Analyze text length statistics
- Visualize word frequencies
- Check for missing values

### Step 2: Text Preprocessing
- Convert to lowercase
- Remove punctuation
- Remove stopwords
- Tokenize text
- Stemming or lemmatization
- Remove special characters and numbers

### Step 3: Feature Extraction
- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- N-grams (unigrams, bigrams)
- Text length features
- Word count features

### Step 4: Model Training
- Split data into train/test sets
- Train multiple models:
  - Naive Bayes (great for text!)
  - Logistic Regression
  - Random Forest
  - SVM
- Compare model performance

### Step 5: Model Evaluation
- Calculate accuracy, precision, recall, F1-score
- Create confusion matrix
- Plot ROC curve
- Analyze misclassified examples

### Step 6: Model Improvement
- Tune hyperparameters
- Try different feature extraction methods
- Handle class imbalance
- Final model selection

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis:
   - Text preprocessing pipeline
   - Feature extraction
   - Model training and evaluation
   - Results and conclusions

2. **Python Script** (optional):
   - Function to predict spam/ham for new emails

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted spam, how many are actually spam
- **Recall**: Of actual spam, how many were detected
- **F1-Score**: Balance between precision and recall

## Text Preprocessing Pipeline

```python
# Example preprocessing steps
1. Lowercase conversion
2. Remove URLs, emails, phone numbers
3. Remove punctuation
4. Remove stopwords
5. Tokenization
6. Stemming/Lemmatization
```

## Feature Extraction Methods

- **Count Vectorizer**: Word frequency
- **TF-IDF**: Term frequency-inverse document frequency
- **N-grams**: Word pairs, triplets
- **Character n-grams**: For catching typos

## Tips

- Naive Bayes works very well for text classification
- TF-IDF usually performs better than simple word counts
- Try different n-gram ranges (1-2, 1-3)
- Visualize most common words in spam vs ham
- Handle class imbalance if needed
- Test on your own emails!

## Resources

- [Kaggle SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [NLTK Documentation](https://www.nltk.org/)

## Next Steps

After completing this project:
- Try more advanced NLP techniques
- Experiment with word embeddings
- Move to [Project 5: Wine Quality](../project-05-wine-quality/README.md)

