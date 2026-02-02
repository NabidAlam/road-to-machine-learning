# Deep Learning Frameworks Quick Reference Guide

Quick reference for TensorFlow/Keras and PyTorch code snippets and best practices.

## Table of Contents

- [Framework Selection](#framework-selection)
- [Keras Code Snippets](#keras-code-snippets)
- [PyTorch Code Snippets](#pytorch-code-snippets)
- [Common Patterns](#common-patterns)
- [Framework Comparison](#framework-comparison)
- [Common Issues & Solutions](#common-issues-solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Framework Selection

### Quick Decision Tree

```
Need deep learning framework?
│
├─ Beginner or rapid prototyping?
│  └─ YES → Use Keras
│
├─ Need maximum flexibility?
│  └─ YES → Use PyTorch
│
├─ Production deployment?
│  └─ YES → Use Keras/TensorFlow
│
└─ Research or custom architectures?
   └─ YES → Use PyTorch
```

### When to Use Each

| Use Case | Recommended Framework | Reason |
|----------|----------------------|--------|
| **Learning** | Keras | Easier to learn |
| **Rapid Prototyping** | Keras | Faster to code |
| **Production** | Keras/TensorFlow | Better deployment tools |
| **Research** | PyTorch | More flexible |
| **Custom Architectures** | PyTorch | Better control |
| **Standard Models** | Keras | Simpler code |

---

## Keras Code Snippets

### Basic Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
```

### Functional API

```python
# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(x_train, y_train, epochs=50, callbacks=callbacks)
```

### Save/Load

```python
# Save
model.save('model.h5')
model.save_weights('weights.h5')

# Load
model = keras.models.load_model('model.h5')
model.load_weights('weights.h5')
```

---

## PyTorch Code Snippets

### Basic Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model()
```

### Training Loop

```python
# DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Save/Load

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## Common Patterns

### Data Preprocessing

**Keras:**
```python
# Normalize
x_train = x_train.astype('float32') / 255.0

# Reshape
x_train = x_train.reshape(-1, 28, 28, 1)
```

**PyTorch:**
```python
# Normalize
x_train = x_train.float() / 255.0

# Reshape
x_train = x_train.view(-1, 784)
```

### GPU Usage

**Keras:**
```python
# Automatic GPU usage if available
# No code needed - TensorFlow handles it
```

**PyTorch:**
```python
# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

### Transfer Learning

**Keras:**
```python
base_model = keras.applications.VGG16(weights='imagenet', include_top=False)
base_model.trainable = False
# Add custom layers
```

**PyTorch:**
```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace classifier
```

---

## Framework Comparison

### Code Comparison

| Task | Keras | PyTorch |
|------|-------|---------|
| **Model Definition** | Sequential/Functional | Class-based |
| **Training** | `model.fit()` | Manual loop |
| **Loss** | String or function | `nn.Module` |
| **Optimizer** | String or object | `optim` object |
| **Callbacks** | Built-in | Manual implementation |
| **GPU** | Automatic | `.to(device)` |

### Feature Comparison

| Feature | Keras | PyTorch |
|---------|-------|---------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Debugging** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Research** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Common Issues & Solutions

### Issue 1: Out of Memory

**Keras:**
```python
# Reduce batch size
model.fit(x_train, y_train, batch_size=32)  # Instead of 128

# Use mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
```

**PyTorch:**
```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=32)

# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing
```

### Issue 2: Model Not Learning

**Keras:**
```python
# Check learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # Lower LR

# Check data normalization
x_train = (x_train - x_train.mean()) / x_train.std()
```

**PyTorch:**
```python
# Check learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Check model.eval() not called during training
model.train()  # During training
```

### Issue 3: Overfitting

**Keras:**
```python
# Add regularization
layers.Dropout(0.5)
layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))

# Early stopping
keras.callbacks.EarlyStopping(patience=5)
```

**PyTorch:**
```python
# Add dropout
nn.Dropout(0.5)

# L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Issue 4: Slow Training

**Keras:**
```python
# Use GPU
# Check: tf.config.list_physical_devices('GPU')

# Increase batch size
model.fit(x_train, y_train, batch_size=256)
```

**PyTorch:**
```python
# Use GPU
model = model.cuda()
x = x.cuda()

# Increase batch size
train_loader = DataLoader(dataset, batch_size=256, num_workers=4)
```

---

## Best Practices Checklist

### Data Preparation
- [ ] Normalize/standardize data
- [ ] Shuffle training data
- [ ] Use validation split
- [ ] Handle class imbalance if needed

### Model Building
- [ ] Start with simple architecture
- [ ] Use appropriate activation functions
- [ ] Add dropout for regularization
- [ ] Use batch normalization if needed

### Training
- [ ] Set appropriate learning rate
- [ ] Use callbacks (early stopping, checkpointing)
- [ ] Monitor training and validation metrics
- [ ] Use learning rate scheduling
- [ ] Save best model

### Evaluation
- [ ] Evaluate on test set
- [ ] Check for overfitting
- [ ] Visualize predictions
- [ ] Compare with baseline

### Keras Specific
- [ ] Use validation_split or validation_data
- [ ] Implement callbacks
- [ ] Use model.save() for deployment
- [ ] Leverage pre-trained models

### PyTorch Specific
- [ ] Set model.train() for training
- [ ] Set model.eval() for inference
- [ ] Use torch.no_grad() for inference
- [ ] Save state_dict() not entire model
- [ ] Use DataLoader for efficiency

---

## Quick Code Templates

### Keras Complete Pipeline

```python
# 1. Load and preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

# 2. Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# 3. Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
```

### PyTorch Complete Pipeline

```python
# 1. Load and preprocess
x_train = torch.FloatTensor(x_train) / 255.0
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)

# 2. Build model
model = Model()

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train
for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. Evaluate
model.eval()
with torch.no_grad():
    # ... evaluation code ...
```

---

## Key Takeaways

1. **Keras**: Easier, faster to prototype, great for production
2. **PyTorch**: More flexible, better for research, Pythonic
3. **Learn both**: Maximum flexibility and career options
4. **Choose wisely**: Based on project needs
5. **Best practices**: Normalize data, use callbacks, monitor training
6. **GPU**: Both support GPU acceleration
7. **Transfer learning**: Use pre-trained models to save time

---

## Next Steps

- Practice with both frameworks
- Build same model in both to compare
- Experiment with transfer learning
- Learn advanced techniques
- Move to computer vision module

**Remember**: Frameworks are tools - understanding fundamentals is key!

