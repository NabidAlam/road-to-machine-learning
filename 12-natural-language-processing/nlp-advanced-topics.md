# Advanced Natural Language Processing Topics

Comprehensive guide to advanced NLP techniques and architectures.

## Table of Contents

- [Advanced Transformer Architectures](#advanced-transformer-architectures)
- [Advanced Text Preprocessing](#advanced-text-preprocessing)
- [Sequence-to-Sequence Models](#sequence-to-sequence-models)
- [Advanced Embeddings](#advanced-embeddings)
- [Model Optimization](#model-optimization)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Transformer Architectures

### BERT (Bidirectional Encoder)

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors='pt')
outputs = model(**encoded)
```

### GPT (Generative Pre-trained Transformer)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Advanced Text Preprocessing

### Handling Special Cases

```python
def advanced_preprocessing(text):
    # Handle emojis
    import emoji
    text = emoji.demojize(text)
    
    # Handle mentions and hashtags
    text = re.sub(r'@\w+', '<MENTION>', text)
    text = re.sub(r'#\w+', '<HASHTAG>', text)
    
    # Handle numbers
    text = re.sub(r'\d+', '<NUMBER>', text)
    
    return text
```

---

## Sequence-to-Sequence Models

### Encoder-Decoder Architecture

```python
# Encoder
encoder_inputs = keras.Input(shape=(None,))
encoder_embedding = layers.Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = layers.LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(None,))
decoder_embedding = layers.Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

---

## Advanced Embeddings

### Contextual Embeddings

```python
# BERT provides contextual embeddings
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Same word, different contexts
text1 = "I deposited money in the bank"
text2 = "I sat by the river bank"

# Get contextual embeddings
inputs1 = tokenizer(text1, return_tensors='pt')
inputs2 = tokenizer(text2, return_tensors='pt')

outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

# Embeddings for "bank" will be different!
```

---

## Model Optimization

### Model Quantization

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('model_name')

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Out-of-Vocabulary Words

**Solution**: Use FastText or subword tokenization (BERT)

### Pitfall 2: Long Sequences

**Solution**: Truncate or use models that handle long sequences

### Pitfall 3: Imbalanced Classes

**Solution**: Use class weights or resampling

---

## Key Takeaways

1. **Transformers**: State-of-the-art for most NLP tasks
2. **Pre-trained Models**: Leverage large-scale training
3. **Contextual Embeddings**: Better than static embeddings
4. **Fine-tuning**: Adapt models to your task

---

**Remember**: Transformers have revolutionized NLP - use them!

