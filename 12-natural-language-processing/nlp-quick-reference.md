# NLP Quick Reference Guide

Quick reference for NLP techniques, code snippets, and best practices.

## Table of Contents

- [Text Preprocessing](#text-preprocessing)
- [Word Embeddings](#word-embeddings)
- [Model Architectures](#model-architectures)
- [Transformers](#transformers)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Text Preprocessing

### Basic Pipeline

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)
```

---

## Word Embeddings

### Word2Vec

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)
vector = model.wv['word']
```

### Pre-trained Embeddings

```python
# GloVe, FastText, etc.
# Use in Keras Embedding layer
```

---

## Model Architectures

### LSTM

```python
model = keras.Sequential([
    layers.Embedding(vocab_size, 128),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
```

### Transformers

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
```

---

## Common Issues & Solutions

### Issue 1: Out-of-Vocabulary

**Solution**: Use FastText or BERT subword tokenization

### Issue 2: Long Sequences

**Solution**: Truncate or use models for long sequences

---

## Best Practices Checklist

- [ ] Preprocess text properly
- [ ] Use appropriate sequence length
- [ ] Try pre-trained embeddings
- [ ] Use transformers for best results
- [ ] Handle variable-length sequences

---

**Remember**: Transformers are state-of-the-art - use them!

