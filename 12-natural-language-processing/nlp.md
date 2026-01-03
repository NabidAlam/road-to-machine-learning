# Natural Language Processing Complete Guide

Comprehensive guide to processing and understanding human language.

## Table of Contents

- [Introduction to NLP](#introduction-to-nlp)
- [Text Preprocessing](#text-preprocessing)
- [Word Embeddings](#word-embeddings)
- [RNNs and LSTMs](#rnns-and-lstms)
- [Transformers](#transformers)
- [NLP Tasks](#nlp-tasks)
- [Practice Exercises](#practice-exercises)

---

## Introduction to NLP

### What is NLP?

Natural Language Processing enables machines to understand, interpret, and generate human language.

**Key Challenges:**
- **Ambiguity**: Words have multiple meanings
- **Context**: Meaning depends on context
- **Variability**: Same meaning, different words
- **Structure**: Grammar and syntax

**Applications:**
- Sentiment analysis
- Machine translation
- Chatbots
- Text classification
- Named Entity Recognition (NER)
- Question answering

### NLP Pipeline

```
Raw Text → Preprocessing → Tokenization → Embedding → Model → Output
```

## Text Preprocessing

### Why Preprocess Text?

- **Normalization**: Consistent format
- **Noise Removal**: Remove irrelevant information
- **Feature Extraction**: Prepare for models
- **Dimensionality Reduction**: Reduce vocabulary size

### Basic Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text, remove_stopwords=True, stem=True, lemmatize=False):
    """
    Comprehensive text preprocessing
    
    Parameters:
    - text: Input text string
    - remove_stopwords: Remove common words
    - stem: Apply stemming
    - lemmatize: Apply lemmatization (overrides stem)
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]
    
    # Stemming or Lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    elif stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)

# Example
text = "This is a sample text for preprocessing! Visit https://example.com or email test@example.com"
processed = preprocess_text(text)
print(f"Original: {text}")
print(f"Processed: {processed}")

# Compare stemming vs lemmatization
words = ['running', 'flies', 'better']
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("\nStemming vs Lemmatization:")
for word in words:
    print(f"  {word}: stem={stemmer.stem(word)}, lemma={lemmatizer.lemmatize(word)}")
```

### Advanced Preprocessing

```python
def advanced_preprocessing(text):
    """Advanced preprocessing with multiple techniques"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Handle contractions
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        # Add more contractions
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text
```

---

## Word Embeddings

### Why Word Embeddings?

- **Dense Representations**: Capture semantic meaning
- **Similarity**: Similar words have similar vectors
- **Context**: Capture word relationships
- **Dimensionality**: Lower than one-hot encoding

### Word2Vec

Learn word vectors from context using neural networks.

**Two Architectures:**
- **Skip-gram**: Predict context from word
- **CBOW**: Predict word from context

```python
from gensim.models import Word2Vec
import numpy as np

# Prepare sentences (list of tokenized sentences)
sentences = [
    ['I', 'love', 'machine', 'learning'],
    ['Machine', 'learning', 'is', 'awesome'],
    ['Deep', 'learning', 'is', 'powerful'],
    ['I', 'enjoy', 'studying', 'machine', 'learning']
]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,      # Dimension of word vectors
    window=5,             # Context window size
    min_count=1,          # Minimum word frequency
    workers=4,            # Number of threads
    sg=1                  # Skip-gram (1) or CBOW (0)
)

# Get word vector
vector = model.wv['machine']
print(f"Vector shape: {vector.shape}")
print(f"Vector (first 10): {vector[:10]}")

# Find similar words
similar = model.wv.most_similar('machine', topn=5)
print("\nWords similar to 'machine':")
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Word arithmetic (king - man + woman ≈ queen)
result = model.wv.most_similar(positive=['machine', 'learning'], negative=[], topn=3)
print("\nWord arithmetic:")
for word, score in result:
    print(f"  {word}: {score:.3f}")

# Save and load model
model.save('word2vec.model')
loaded_model = Word2Vec.load('word2vec.model')
```

### Using Pre-trained Embeddings

```python
# Load GloVe embeddings
def load_glove_embeddings(file_path):
    """Load GloVe word embeddings"""
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Example usage
# embeddings_index = load_glove_embeddings('glove.6B.100d.txt')
# print(f"Loaded {len(embeddings_index)} word vectors")

# Create embedding matrix for Keras
def create_embedding_matrix(word_index, embeddings_index, embedding_dim=100):
    """Create embedding matrix for Keras Embedding layer"""
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

# Use in model
# embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index)
# embedding_layer = layers.Embedding(
#     vocab_size,
#     embedding_dim,
#     weights=[embedding_matrix],
#     trainable=False  # Freeze pre-trained embeddings
# )
```

### FastText

Handles out-of-vocabulary words using subword information.

```python
try:
    from gensim.models import FastText
    
    # Train FastText
    fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1)
    
    # Can get vectors for unseen words
    unseen_vector = fasttext_model.wv['unseenword']
    print("FastText can handle out-of-vocabulary words!")
except ImportError:
    print("Install gensim: pip install gensim")
```

---

## RNNs and LSTMs

### Why RNNs for Text?

- **Sequential Data**: Text is ordered sequence
- **Context**: Previous words inform current word
- **Variable Length**: Handle different text lengths

### Vanilla RNN

Basic recurrent unit (suffers from vanishing gradient).

```python
# Simple RNN layer
rnn_layer = layers.SimpleRNN(64, return_sequences=True)
```

### LSTM (Long Short-Term Memory)

Solves vanishing gradient problem with gates.

**LSTM Gates:**
- **Forget Gate**: What to forget
- **Input Gate**: What to store
- **Output Gate**: What to output

### LSTM for Sentiment Analysis

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Prepare text data
texts = ["I love this movie", "This movie is terrible", ...]
labels = [1, 0, ...]  # 1 = positive, 0 = negative

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convert to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to same length
max_length = 100
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

print(f"Vocabulary size: {len(word_index)}")
print(f"Sequence shape: {X.shape}")

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_length),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ReduceLROnPlateau(patience=2)
    ]
)
```

### Bidirectional LSTM

Process sequences in both directions.

```python
# Bidirectional LSTM
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    layers.Dense(1, activation='sigmoid')
])
```

### GRU (Gated Recurrent Unit)

Simpler alternative to LSTM.

```python
# GRU model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_length),
    layers.GRU(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])
```

---

## Transformers

### Why Transformers?

- **Attention Mechanism**: Focus on relevant parts
- **Parallel Processing**: Faster than RNNs
- **State-of-the-Art**: Best performance on most NLP tasks
- **Pre-trained Models**: Leverage large-scale training

### Transformer Architecture

**Key Components:**
- **Self-Attention**: Relate different positions
- **Multi-Head Attention**: Multiple attention mechanisms
- **Position Encoding**: Inject position information
- **Feed-Forward Networks**: Process attended features

### Using Hugging Face

Hugging Face provides easy access to pre-trained transformer models.

```python
try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              AutoModel, pipeline)
    import torch
    
    # Method 1: Using pipeline (easiest)
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this movie!")
    print(result)
    
    # Method 2: Using model directly
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Predict
    text = "I love this movie!"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    print(f"Positive: {predictions[0][1]:.3f}, Negative: {predictions[0][0]:.3f}")
    
    # Method 3: Fine-tuning for custom task
    from transformers import Trainer, TrainingArguments
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
except ImportError:
    print("Install transformers: pip install transformers")
```

### Popular Transformer Models

| Model | Use Case | Size | Notes |
|-------|----------|------|-------|
| **BERT** | General NLP | Large | Bidirectional encoder |
| **GPT** | Text generation | Large | Autoregressive |
| **DistilBERT** | Fast inference | Medium | Smaller BERT |
| **RoBERTa** | Better BERT | Large | Improved training |
| **T5** | Text-to-text | Large | Unified framework |

### Fine-tuning Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Fine-tune on your dataset
# trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
# trainer.train()
```

---

## NLP Tasks

### Sentiment Analysis

Classify text as positive, negative, or neutral.

```python
# Using transformers
classifier = pipeline("sentiment-analysis")
result = classifier("This product is amazing!")
print(result)
```

### Text Classification

Categorize documents into classes.

```python
# Multi-class text classification
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=500),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Named Entity Recognition (NER)

Identify entities (person, location, organization).

```python
try:
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    result = ner_pipeline("Apple is located in Cupertino, California.")
    print(result)
except ImportError:
    print("Install transformers: pip install transformers")
```

### Question Answering

Answer questions from context.

```python
qa_pipeline = pipeline("question-answering")
result = qa_pipeline(
    question="What is machine learning?",
    context="Machine learning is a subset of artificial intelligence..."
)
print(result)
```

## Practice Exercises

### Exercise 1: Sentiment Analysis with LSTM

**Task:** Build sentiment analysis model using LSTM on IMDB dataset.

**Solution:**
```python
from tensorflow.keras.datasets import imdb

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build improved LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_length),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=2)]
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Exercise 2: Text Classification with Transformers

**Task:** Fine-tune BERT for text classification.

**Solution:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Load pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Tokenize and train
# (See Hugging Face documentation for complete example)
```

---

## Resources and Further Learning

### Books

1. **"Speech and Language Processing"** - Jurafsky & Martin
   - [Free Online (3rd Edition)](https://web.stanford.edu/~jurafsky/slp3/)
   - Comprehensive NLP textbook covering all major topics

2. **"Natural Language Processing with Python"** - Bird, Klein & Loper
   - [NLTK Book](https://www.nltk.org/book/)
   - Practical guide using NLTK

3. **"Deep Learning for Natural Language Processing"** - Palash Goyal, Sumit Pandey, Karan Jain
   - Covers modern deep learning approaches to NLP

### Important Papers

1. **"Attention Is All You Need"** - Vaswani et al., 2017 (Transformers)
2. **"BERT: Pre-training of Deep Bidirectional Transformers"** - Devlin et al., 2018
3. **"GPT-3: Language Models are Few-Shot Learners"** - Brown et al., 2020
4. **"Word2Vec"** - Mikolov et al., 2013 (Word embeddings)
5. **"GloVe: Global Vectors for Word Representation"** - Pennington et al., 2014
6. **"Sequence to Sequence Learning with Neural Networks"** - Sutskever et al., 2014
7. **"Long Short-Term Memory"** - Hochreiter & Schmidhuber, 1997 (LSTM)

### Online Courses

1. **CS224N: Natural Language Processing with Deep Learning** - Stanford
   - [Course Website](https://web.stanford.edu/class/cs224n/)
   - Comprehensive deep learning for NLP course

2. **Natural Language Processing Specialization** - Coursera (DeepLearning.AI)
   - [Course Link](https://www.coursera.org/specializations/natural-language-processing)
   - Practical NLP with transformers

3. **Fast.ai NLP Course**
   - [Course Website](https://www.fast.ai/)
   - Practical deep learning for NLP

### Datasets

1. **Text Classification**:
   - [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
   - [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

2. **Sentiment Analysis**:
   - [Sentiment140](http://help.sentiment140.com/for-students)
   - [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

3. **Named Entity Recognition**:
   - [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
   - [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)

4. **Question Answering**:
   - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
   - [MS MARCO](https://microsoft.github.io/msmarco/)

5. **Machine Translation**:
   - [WMT](http://www.statmt.org/wmt20/)
   - [OPUS](https://opus.nlpl.eu/)

### Tools and Libraries

1. **NLTK**: Natural Language Toolkit
   - [Documentation](https://www.nltk.org/)
   - Comprehensive NLP library

2. **spaCy**: Industrial-strength NLP
   - [Documentation](https://spacy.io/)
   - Fast and efficient NLP library

3. **Hugging Face Transformers**: Pre-trained transformer models
   - [Documentation](https://huggingface.co/docs/transformers/)
   - Easy access to state-of-the-art models

4. **Gensim**: Topic modeling and word embeddings
   - [Documentation](https://radimrehurek.com/gensim/)
   - Word2Vec, Doc2Vec, LDA

5. **TextBlob**: Simple NLP library
   - [Documentation](https://textblob.readthedocs.io/)
   - Easy-to-use text processing

---

## Key Takeaways

1. **Preprocessing**: Essential for good NLP models
2. **Word Embeddings**: Capture semantic meaning (Word2Vec, GloVe, FastText)
3. **RNNs/LSTMs**: Handle sequential data, good for sequences
4. **Transformers**: State-of-the-art, use pre-trained models
5. **Hugging Face**: Easy access to transformer models
6. **Fine-tuning**: Adapt pre-trained models to your task

---

## Best Practices

### Text Preprocessing
- Normalize text (lowercase, remove punctuation)
- Handle special cases (URLs, emails, contractions)
- Choose between stemming and lemmatization
- Remove stopwords when appropriate

### Model Selection
- **Small datasets**: Use pre-trained embeddings + simple models
- **Medium datasets**: Use LSTM/GRU
- **Large datasets**: Use transformers
- **Always try**: Pre-trained transformer models first

### Training
- Use appropriate sequence length
- Handle variable-length sequences with padding
- Use dropout for regularization
- Monitor for overfitting

---

## Next Steps

- Practice with text datasets (IMDB, news articles)
- Experiment with different transformer models
- Try fine-tuning on your own dataset
- Explore advanced NLP tasks (NER, QA, translation)
- Move to [13-model-deployment](../13-model-deployment/README.md)

**Remember**: NLP is rapidly evolving with transformers! Use pre-trained models when possible.

