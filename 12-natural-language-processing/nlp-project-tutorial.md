# Complete NLP Project Tutorial

Step-by-step walkthrough of building a sentiment analysis system.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading](#step-1-data-loading)
- [Step 2: Text Preprocessing](#step-2-text-preprocessing)
- [Step 3: Build LSTM Model](#step-3-build-lstm-model)
- [Step 4: Train Model](#step-4-train-model)
- [Step 5: Use Transformers](#step-5-use-transformers)
- [Step 6: Compare Results](#step-6-compare-results)

---

## Project Overview

**Project**: Sentiment Analysis on Movie Reviews

**Dataset**: IMDB Movie Reviews

**Goals**: Classify reviews as positive or negative

---

## Step 1: Data Loading

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)
```

---

## Step 2: Text Preprocessing

```python
# Already tokenized in IMDB dataset
# For custom data, use Tokenizer
```

---

## Step 3: Build LSTM Model

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Embedding(10000, 128),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Step 4: Train Model

```python
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Step 5: Use Transformers

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This movie is amazing!")
print(result)
```

---

## Step 6: Compare Results

```python
print("LSTM Accuracy:", lstm_acc)
print("Transformer Accuracy:", transformer_acc)
```

---

**Congratulations!** You've built a complete NLP system!

