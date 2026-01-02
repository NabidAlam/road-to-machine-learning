# Transformer Fine-Tuning Comprehensive Guide

Complete guide to fine-tuning transformer models (T5, BERT, GPT) using the Hugging Face library for various NLP tasks.

## Table of Contents

- [Introduction to Transformer Fine-Tuning](#introduction-to-transformer-fine-tuning)
- [Understanding Transformer Architecture](#understanding-transformer-architecture)
- [Fine-Tuning T5 for Text Summarization](#fine-tuning-t5-for-text-summarization)
- [Fine-Tuning BERT for Classification](#fine-tuning-bert-for-classification)
- [Fine-Tuning GPT for Text Generation](#fine-tuning-gpt-for-text-generation)
- [Optimization Techniques](#optimization-techniques)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to Transformer Fine-Tuning

### What is Fine-Tuning?

**Fine-tuning** is the process of taking a pre-trained transformer model and adapting it to a specific task by training it further on task-specific data.

**Why Fine-Tune?**
- **Transfer Learning**: Leverage knowledge from large pre-trained models
- **Task-Specific Adaptation**: Adapt to your specific use case
- **Data Efficiency**: Achieve good performance with less data
- **Cost Effective**: Faster than training from scratch

### Pre-Trained Models

**Popular Models:**
- **BERT**: Bidirectional Encoder Representations from Transformers
- **GPT**: Generative Pre-trained Transformer
- **T5**: Text-to-Text Transfer Transformer
- **RoBERTa**: Robustly Optimized BERT
- **DistilBERT**: Distilled version of BERT

### Hugging Face Transformers

The Hugging Face `transformers` library provides easy access to pre-trained models and fine-tuning capabilities.

**Installation:**
```bash
pip install transformers torch datasets accelerate
```

---

## Understanding Transformer Architecture

### Key Components

**1. Encoder (BERT, T5 Encoder)**
- Self-attention mechanism
- Bidirectional context
- Good for understanding tasks

**2. Decoder (GPT, T5 Decoder)**
- Masked self-attention
- Autoregressive generation
- Good for generation tasks

**3. Encoder-Decoder (T5)**
- Combines encoder and decoder
- Text-to-text framework
- Versatile for many tasks

### Attention Mechanism

**Self-Attention:**
- Allows model to focus on relevant parts of input
- Computes relationships between all positions
- Enables parallel processing

**Multi-Head Attention:**
- Multiple attention heads capture different relationships
- Each head learns different patterns
- Combined for richer representations

---

## Fine-Tuning T5 for Text Summarization

### What is T5?

**T5 (Text-to-Text Transfer Transformer)** is a unified framework that treats all NLP tasks as text-to-text problems.

**Key Features:**
- Encoder-decoder architecture
- Pre-trained on large text corpus
- Can be fine-tuned for summarization, translation, Q&A, etc.

### Fine-Tuning T5 for Summarization

**Step 1: Load Pre-Trained Model and Tokenizer**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # Options: t5-small, t5-base, t5-large, t5-3b, t5-11b
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**Step 2: Prepare Dataset**

```python
from datasets import load_dataset

# Load dataset (example: CNN/DailyMail for summarization)
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Or use your own dataset
# dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv"})

# Preprocess function
def preprocess_function(examples):
    # Prefix for T5 (required for summarization)
    inputs = ["summarize: " + article for article in examples["article"]]
    targets = examples["highlights"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss function
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

**Step 3: Fine-Tune with Trainer**

```python
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-summarization",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,  # Set to True to push to Hugging Face Hub
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune
trainer.train()

# Save model
trainer.save_model("./t5-summarization-final")
tokenizer.save_pretrained("./t5-summarization-final")
```

**Step 4: Generate Summaries**

```python
# Load fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("./t5-summarization-final")
tokenizer = T5Tokenizer.from_pretrained("./t5-summarization-final")
model = model.to(device)

# Generate summary
def summarize(text, max_length=128, min_length=30):
    # Add prefix
    input_text = "summarize: " + text
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Example
article = """
Artificial intelligence has transformed many industries, from healthcare to finance.
Machine learning models can now diagnose diseases, predict stock prices, and even
drive cars. However, with great power comes great responsibility. We must ensure
that AI systems are fair, transparent, and beneficial to all of humanity.
"""

summary = summarize(article)
print("Summary:", summary)
```

---

## Fine-Tuning BERT for Classification

### What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based model that uses bidirectional context to understand language.

**Key Features:**
- Encoder-only architecture
- Bidirectional context
- Pre-trained on masked language modeling and next sentence prediction
- Excellent for classification, NER, Q&A

### Fine-Tuning BERT for Text Classification

**Step 1: Load Pre-Trained Model**

```python
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Option 1: Use BERT directly
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

# Option 2: Use Auto classes (more flexible)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**Step 2: Prepare Dataset**

```python
from datasets import load_dataset
import pandas as pd

# Load dataset (example: IMDB for sentiment analysis)
dataset = load_dataset("imdb")

# Or use your own CSV
# df = pd.read_csv("your_data.csv")
# dataset = Dataset.from_pandas(df)

# Preprocess function
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Rename columns if needed
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

**Step 3: Fine-Tune**

```python
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-classification",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune
trainer.train()

# Save model
trainer.save_model("./bert-classification-final")
tokenizer.save_pretrained("./bert-classification-final")
```

**Step 4: Make Predictions**

```python
# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained("./bert-classification-final")
tokenizer = BertTokenizer.from_pretrained("./bert-classification-final")
model = model.to(device)
model.eval()

def predict(text):
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return predicted_class, probabilities[0].cpu().numpy()

# Example
text = "This movie was absolutely fantastic! I loved every minute of it."
predicted_class, probabilities = predict(text)
print(f"Predicted Class: {predicted_class}")
print(f"Probabilities: {probabilities}")
```

---

## Fine-Tuning GPT for Text Generation

### What is GPT?

**GPT (Generative Pre-trained Transformer)** is an autoregressive language model that generates text one token at a time.

**Key Features:**
- Decoder-only architecture
- Autoregressive generation
- Pre-trained on large text corpus
- Good for text generation, completion, dialogue

### Fine-Tuning GPT-2 for Text Generation

**Step 1: Load Pre-Trained Model**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2
model_name = "gpt2"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**Step 2: Prepare Dataset**

```python
from datasets import load_dataset

# Load dataset (example: your text generation dataset)
# dataset = load_dataset("your_dataset")

# Or create from text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return {"text": texts}

# Preprocess function
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

**Step 3: Fine-Tune**

```python
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is not masked language model
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-generation",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune
trainer.train()

# Save model
trainer.save_model("./gpt2-generation-final")
tokenizer.save_pretrained("./gpt2-generation-final")
```

**Step 4: Generate Text**

```python
# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./gpt2-generation-final")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-generation-final")
model = model.to(device)
model.eval()

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    # Tokenize prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example
prompt = "The future of artificial intelligence"
generated = generate_text(prompt, max_length=150)
print("Generated Text:")
print(generated)
```

---

## Optimization Techniques

### Learning Rate Scheduling

```python
from transformers import get_linear_schedule_with_warmup

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Create scheduler
num_training_steps = len(train_dataset) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()
```

### Gradient Accumulation

For larger effective batch sizes when GPU memory is limited:

```python
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,  # Small batch size
    gradient_accumulation_steps=4,   # Effective batch size = 4 * 4 = 16
    # ... other arguments
)
```

### Mixed Precision Training

For faster training and lower memory usage:

```python
training_args = TrainingArguments(
    output_dir="./model",
    fp16=True,  # Enable mixed precision
    # ... other arguments
)
```

### Freezing Layers

Freeze early layers and only fine-tune later layers:

```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last few layers
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

# Unfreeze classifier head
for param in model.classifier.parameters():
    param.requires_grad = True
```

### Learning Rate Finder

Find optimal learning rate:

```python
from transformers import TrainerCallback

class LRFinderCallback(TrainerCallback):
    def __init__(self):
        self.lrs = []
        self.losses = []
    
    def on_step_end(self, args, state, control, **kwargs):
        self.lrs.append(args.learning_rate)
        self.losses.append(state.log_history[-1].get('loss', 0))

# Use in training
trainer = Trainer(
    # ... other arguments
    callbacks=[LRFinderCallback()]
)
```

---

## Best Practices

### 1. Data Preparation

- **Clean Data**: Remove noise, handle missing values
- **Balanced Dataset**: Ensure balanced classes for classification
- **Appropriate Size**: Fine-tuning typically needs less data than training from scratch
- **Validation Split**: Always keep validation set

### 2. Model Selection

- **Start Small**: Use smaller models (t5-small, bert-base) for faster iteration
- **Scale Up**: Move to larger models if needed
- **Task-Specific**: Choose model architecture suited to your task

### 3. Hyperparameter Tuning

- **Learning Rate**: Start with 2e-5, adjust based on results
- **Batch Size**: As large as GPU memory allows
- **Epochs**: 3-5 epochs usually sufficient for fine-tuning
- **Warmup Steps**: 10% of total training steps

### 4. Monitoring

- **Track Metrics**: Monitor loss, accuracy, F1-score
- **Use TensorBoard**: Visualize training curves
- **Early Stopping**: Stop if validation loss stops improving

### 5. Evaluation

- **Multiple Metrics**: Use task-appropriate metrics
- **Test Set**: Always evaluate on held-out test set
- **Error Analysis**: Analyze failure cases

### 6. Deployment

- **Model Optimization**: Quantize or use smaller models for production
- **Inference Speed**: Optimize for latency if needed
- **Version Control**: Track model versions

---

## Resources

### Official Documentation

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Course](https://huggingface.co/course) - Free comprehensive NLP course
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Tutorials

- [Fine-Tuning T5 for Summarization](https://huggingface.co/docs/transformers/tasks/summarization)
- [Fine-Tuning BERT for Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Fine-Tuning GPT for Generation](https://huggingface.co/docs/transformers/tasks/language_modeling)

### Books

- "Natural Language Processing with Transformers" by Lewis Tunstall et al.
- "Hands-On Machine Learning" by Aurélien Géron (Chapter on NLP)

### Community

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Hugging Face GitHub](https://github.com/huggingface/transformers)

---

**Remember**: Fine-tuning transformers is a powerful technique for adapting pre-trained models to your specific tasks. Start with smaller models and simpler tasks, then gradually move to more complex scenarios. Always monitor your training and validate on held-out data!

