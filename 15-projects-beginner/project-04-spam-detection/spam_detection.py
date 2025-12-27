"""
Spam Email Detection Project
Classify emails as spam or not spam using text features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("Spam Email Detection Project")
print("="*60)

# Note: You need to download the dataset
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
print("\nNote: Download SMS Spam Collection dataset from Kaggle")
print("Place it in the data/ directory or update the path below.\n")

# Load data (update path as needed)
try:
    # Try different possible column names
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    # Common column names: v1, v2 or label, text
    if 'v1' in df.columns:
        df.columns = ['label', 'text'] + list(df.columns[2:])
    df = df[['label', 'text']]
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Creating sample data for demonstration...")
    # Create sample spam/ham messages
    spam_messages = [
        "WINNER!! You have won a $1000 prize! Click here to claim.",
        "URGENT: Your account will be closed. Verify now!",
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Congratulations! You've been selected for a free gift.",
        "Limited time offer! Buy now and save 50%!"
    ]
    ham_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Thanks for the update. I'll review it and get back to you.",
        "Can you send me the report by end of day?",
        "See you at the meeting at 3pm.",
        "Just wanted to check in and see how you're doing."
    ]
    df = pd.DataFrame({
        'label': ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages),
        'text': spam_messages + ham_messages
    })
    print("Using sample data for demonstration purposes.")

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check class distribution
print("\n" + "="*60)
print("Class Distribution")
print("="*60)
print(df['label'].value_counts())
print(f"\nSpam percentage: {(df['label'] == 'spam').mean():.2%}")

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{10,}', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Preprocess text
print("\n" + "="*60)
print("Text Preprocessing")
print("="*60)
df['text_cleaned'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete!")
print(f"\nSample cleaned text:")
print(df[['text', 'text_cleaned']].head())

# Text length analysis
df['text_length'] = df['text_cleaned'].str.len()
df['word_count'] = df['text_cleaned'].str.split().str.len()

print("\n" + "="*60)
print("Text Statistics")
print("="*60)
print(df.groupby('label')[['text_length', 'word_count']].mean())

# Encode labels
df['label_encoded'] = (df['label'] == 'spam').astype(int)

# Prepare data
X = df['text_cleaned']
y = df['label_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature extraction methods
vectorizers = {
    'Count Vectorizer': CountVectorizer(max_features=5000, ngram_range=(1, 2)),
    'TF-IDF Vectorizer': TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
}

# Train models with different vectorizers
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "="*60)
print("Model Training and Evaluation")
print("="*60)

for vec_name, vectorizer in vectorizers.items():
    print(f"\n{'='*60}")
    print(f"Using {vec_name}")
    print(f"{'='*60}")
    
    # Transform text
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    for model_name, model in models.items():
        full_name = f"{model_name} ({vec_name})"
        
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[full_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }
        
        print(f"\n{full_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")

# Compare all models
print("\n" + "="*60)
print("Model Comparison")
print("="*60)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
}).sort_values('Accuracy', ascending=False)
print(comparison_df.to_string(index=False))

# Best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Classification report
print("\n" + "="*60)
print("Classification Report (Best Model)")
print("="*60)
print(classification_report(y_test, best_predictions, target_names=['Ham', 'Spam']))

# Test on new messages
print("\n" + "="*60)
print("Testing on Sample Messages")
print("="*60)

# Use TF-IDF and Naive Bayes (typically best for text)
best_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_best = best_vec.fit_transform(X_train)
X_test_best = best_vec.transform(X_test)
best_model = MultinomialNB()
best_model.fit(X_train_best, y_train)

test_messages = [
    "WINNER!! You have won a prize! Click here!",
    "Hey, are we meeting tomorrow?",
    "Free entry to win tickets!",
    "Thanks for the update."
]

for msg in test_messages:
    msg_cleaned = preprocess_text(msg)
    msg_vec = best_vec.transform([msg_cleaned])
    prediction = best_model.predict(msg_vec)[0]
    probability = best_model.predict_proba(msg_vec)[0]
    label = "Spam" if prediction == 1 else "Ham"
    print(f"\nMessage: {msg}")
    print(f"Prediction: {label}")
    print(f"Confidence: Ham={probability[0]:.3f}, Spam={probability[1]:.3f}")

print("\n" + "="*60)
print("Project Complete!")
print("="*60)
print("\nNext steps:")
print("1. Download actual SMS Spam dataset from Kaggle")
print("2. Try different n-gram ranges")
print("3. Experiment with word embeddings")
print("4. Try advanced NLP techniques")

