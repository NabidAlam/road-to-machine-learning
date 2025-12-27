# Natural Language Processing Complete Guide

Comprehensive guide to processing and understanding human language.

## Table of Contents

- [Text Preprocessing](#text-preprocessing)
- [Word Embeddings](#word-embeddings)
- [RNNs and LSTMs](#rnns-and-lstms)
- [Transformers](#transformers)
- [Practice Exercises](#practice-exercises)

---

## Text Preprocessing

### Basic Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)

# Example
text = "This is a sample text for preprocessing!"
processed = preprocess_text(text)
print(processed)
```

---

## Word Embeddings

### Word2Vec

```python
from gensim.models import Word2Vec

# Prepare sentences
sentences = [['I', 'love', 'machine', 'learning'],
             ['Machine', 'learning', 'is', 'awesome']]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get word vector
vector = model.wv['machine']
print(f"Vector shape: {vector.shape}")

# Find similar words
similar = model.wv.most_similar('machine', topn=5)
print(similar)
```

### Using Pre-trained Embeddings

```python
# Load GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f"Loaded {len(embeddings_index)} word vectors")
```

---

## RNNs and LSTMs

### LSTM for Sentiment Analysis

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=100),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, validation_split=0.2)
```

---

## Transformers

### Using Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict
text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"Positive: {predictions[0][1]:.3f}, Negative: {predictions[0][0]:.3f}")
```

---

## Practice Exercises

### Exercise 1: Sentiment Analysis

**Task:** Build sentiment analysis model using LSTM.

**Solution:**
```python
# Load IMDB dataset
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Build model
model = keras.Sequential([
    layers.Embedding(10000, 128),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

## Key Takeaways

1. **Preprocessing**: Clean and normalize text
2. **Embeddings**: Dense vector representations
3. **RNNs/LSTMs**: Handle sequences
4. **Transformers**: State-of-the-art NLP

---

## Next Steps

- Practice with text datasets
- Experiment with transformers
- Move to [13-model-deployment](../13-model-deployment/README.md)

**Remember**: NLP is rapidly evolving with transformers!

