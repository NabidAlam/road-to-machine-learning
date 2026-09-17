# Complete NLP Project Tutorial

Step-by-step walkthrough of building a sentiment analysis system on IMDB movie reviews.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading](#step-1-data-loading)
- [Step 2: Text Preprocessing](#step-2-text-preprocessing)
- [Step 3: Build LSTM Model](#step-3-build-lstm-model)
- [Step 4: Train and Evaluate LSTM](#step-4-train-and-evaluate-lstm)
- [Step 5: Use Transformers](#step-5-use-transformers)
- [Step 6: Compare Results](#step-6-compare-results)

---

## Project Overview

**Project**: Sentiment Analysis on Movie Reviews

**Dataset**: IMDB Movie Reviews (built into Keras)

**Goals**: Classify reviews as positive or negative using an LSTM and compare with a transformer pipeline.

---

## Step 1: Data Loading

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))
```

The IMDB dataset ships as integer sequences. Each number is a word index. That is why we pad sequences to the same length before training.

---

## Step 2: Text Preprocessing

For this dataset, Keras already tokenized the text. You load integer sequences directly.

If you use your own raw text later, tokenize with a `Tokenizer`, fit it on training text only, then convert sentences to sequences and pad them the same way.

```python
# Example for custom text (not needed for IMDB load above)
# from tensorflow.keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
# tokenizer.fit_on_texts(train_texts)
# train_seq = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=200)
```

---

## Step 3: Build LSTM Model

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=200),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
```

---

## Step 4: Train and Evaluate LSTM

```python
# Small subset for a quick CPU run. Use full x_train / y_train for a longer run.
n_train, n_test = 2000, 500
history = model.fit(
    x_train[:n_train],
    y_train[:n_train],
    epochs=1,
    batch_size=128,
    validation_split=0.2,
    verbose=0,
)

test_loss, lstm_acc = model.evaluate(x_test[:n_test], y_test[:n_test], verbose=0)
print(f"LSTM test accuracy: {lstm_acc:.4f}")
```

Expect roughly 85–88% test accuracy after a few epochs on CPU. Your exact number depends on hardware and random seed.

---

## Step 5: Use Transformers

Use a pre-trained sentiment pipeline for a quick baseline on raw text.

```python snippet-skip
from transformers import pipeline

# Downloads a pretrained model; skip in CI replay
classifier = pipeline("sentiment-analysis", truncation=True)

samples = [
    "This movie is amazing!",
    "Terrible plot and bad acting.",
]

for text in samples:
    print(text, "->", classifier(text))
```

For a fair numeric comparison on IMDB, you would run the pipeline on decoded test reviews and compute accuracy. That step is slower on CPU. The snippet above shows how the API works on sample sentences.

Optional extension. Decode a small test batch, score with the pipeline, and count matches with `y_test` to get `transformer_acc`.

---

## Step 6: Compare Results

```python
print(f"LSTM test accuracy:      {lstm_acc:.4f}")
print("Transformer pipeline:    run optional batch eval above for transformer_acc")
```

If you completed the optional transformer batch evaluation, print both numbers side by side. LSTM trains on the full IMDB training set. A general sentiment model may behave differently on movie review wording.

---

## Next steps

1. Try bidirectional LSTM or a small CNN on the same padded sequences.
2. Add early stopping and plot train vs validation loss.
3. Move to the full NLP module guide for transformers, BERT, and production patterns.

**Module guide:** [nlp.md](nlp.md)
