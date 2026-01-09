# Deep Learning Frameworks Complete Guide

Comprehensive guide to TensorFlow/Keras and PyTorch for building deep learning models.

## Table of Contents

- [Introduction](#introduction)
- [TensorFlow/Keras](#tensorflowkeras)
- [PyTorch](#pytorch)
- [Framework Comparison](#framework-comparison)
- [Model Building](#model-building)
- [Training](#training)
- [Callbacks and Monitoring](#callbacks-and-monitoring)
- [Model Saving and Loading](#model-saving-and-loading)
- [Transfer Learning](#transfer-learning)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Use Frameworks?

Deep learning frameworks provide:
- **Automatic Differentiation**: No manual gradient computation
- **GPU Acceleration**: Easy GPU utilization
- **Pre-built Layers**: Ready-to-use components
- **Optimization**: Built-in optimizers and techniques
- **Ecosystem**: Pre-trained models, tools, and community

### Framework Overview

**TensorFlow/Keras:**
- Industry standard, production-ready
- Great documentation and community
- TensorFlow Serving for deployment
- Keras: High-level API, easy to use

**PyTorch:**
- Research-friendly, flexible
- Dynamic computation graphs
- Pythonic, intuitive API
- Strong research community

**Choosing a Framework:**
- **Keras**: Beginners, rapid prototyping, production
- **PyTorch**: Research, custom architectures, flexibility
- **Both**: Learn both for maximum flexibility

---

## TensorFlow/Keras

### Installation

```python
# Install TensorFlow
pip install tensorflow

# For GPU support (optional)
pip install tensorflow-gpu

# Verify installation
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### Sequential API

Simple linear stack of layers - easiest way to build models.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='hidden1'),
    layers.Dropout(0.2, name='dropout1'),
    layers.Dense(64, activation='relu', name='hidden2'),
    layers.Dropout(0.2, name='dropout2'),
    layers.Dense(10, activation='softmax', name='output')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Visualize model architecture
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
```

### Functional API

More flexible - supports complex architectures (multi-input, multi-output, shared layers).

```python
# Functional API (more flexible)
inputs = keras.Input(shape=(784,), name='input')
x = layers.Dense(128, activation='relu', name='hidden1')(inputs)
x = layers.Dropout(0.2, name='dropout1')(x)
x = layers.Dense(64, activation='relu', name='hidden2')(x)
x = layers.Dropout(0.2, name='dropout2')(x)
outputs = layers.Dense(10, activation='softmax', name='output')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mlp_model')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Multi-input example
input1 = keras.Input(shape=(784,), name='input1')
input2 = keras.Input(shape=(10,), name='input2')
x1 = layers.Dense(64, activation='relu')(input1)
x2 = layers.Dense(64, activation='relu')(input2)
merged = layers.concatenate([x1, x2])
outputs = layers.Dense(10, activation='softmax')(merged)
multi_input_model = keras.Model(inputs=[input1, input2], outputs=outputs)
```

### Model Subclassing

Maximum flexibility - define custom models by subclassing.

```python
class MLP(keras.Model):
    def __init__(self, hidden_units=[128, 64], num_classes=10, dropout_rate=0.2):
        super().__init__()
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)

# Create and compile
model = MLP(hidden_units=[128, 64], num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Functional API

```python
# Functional API (more flexible)
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Data Preprocessing

```python
# Load data (MNIST example)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# Alternative: Use ImageDataGenerator for augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

# For image data (if using Conv2D)
# x_train = x_train.reshape(-1, 28, 28, 1)
# train_generator = datagen.flow(x_train, y_train, batch_size=32, subset='training')
# val_generator = datagen.flow(x_train, y_train, batch_size=32, subset='validation')
```

### Training

```python
# Train model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7)
    ]
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predictions: {predicted_classes}")
print(f"True labels: {y_test[:10]}")
```

---

## PyTorch

### Installation

```python
# Install PyTorch (CPU version)
pip install torch torchvision

# For GPU support (CUDA), visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### Tensors

PyTorch's fundamental data structure - similar to NumPy arrays but with GPU support.

```python
import torch
import numpy as np

# Create tensors
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# From NumPy
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)

# To NumPy
arr = x.numpy()

# GPU tensors
if torch.cuda.is_available():
    x_gpu = x.cuda()
    # or
    x_gpu = x.to('cuda')

# Operations
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = torch.matmul(x, y)  # Matrix multiplication
```

### Building Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Sequential
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10)
)

# Method 2: Class-based (more flexible)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.linear_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

model = NeuralNetwork(input_size=784, hidden_sizes=[128, 64], num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Training Loop

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Prepare data
train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    # Calculate averages
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy', linewidth=2)
plt.plot(val_accs, label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Framework Comparison

### Keras vs PyTorch

| Feature | Keras | PyTorch |
|---------|-------|---------|
| **Ease of Use** | Very easy | Moderate |
| **Flexibility** | Good | Excellent |
| **Debugging** | Moderate | Excellent |
| **Production** | Excellent | Good |
| **Research** | Good | Excellent |
| **Community** | Large | Large |
| **Learning Curve** | Gentle | Steeper |

### When to Use Each

**Use Keras when:**
- You're a beginner
- You need rapid prototyping
- You're building standard architectures
- You need production deployment
- You prefer high-level APIs

**Use PyTorch when:**
- You need maximum flexibility
- You're doing research
- You need dynamic computation graphs
- You prefer Pythonic code
- You need fine-grained control

## Callbacks and Monitoring

### Keras Callbacks

```python
# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Model Checkpointing
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Learning Rate Reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)

# CSV Logger
csv_logger = keras.callbacks.CSVLogger('training.log')

# Use in training
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint, reduce_lr, tensorboard, csv_logger]
)
```

### PyTorch Monitoring

```python
from torch.utils.tensorboard import SummaryWriter

# TensorBoard writer
writer = SummaryWriter('runs/experiment_1')

# Log during training
for epoch in range(num_epochs):
    # ... training code ...
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Val', val_acc, epoch)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

writer.close()

# View with: tensorboard --logdir=runs
```

## Model Saving and Loading

### Keras

```python
# Save entire model (architecture + weights + optimizer state)
model.save('model.h5')
model.save('model.keras')  # New format in Keras 3

# Save only weights
model.save_weights('weights.h5')

# Save model as SavedModel (for TensorFlow Serving)
model.save('saved_model', save_format='tf')

# Load entire model
model = keras.models.load_model('model.h5')

# Load only weights (need to create model first)
model = create_model()  # Define architecture
model.load_weights('weights.h5')

# Load and compile
model = keras.models.load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### PyTorch

```python
# Save model state dict (recommended)
torch.save(model.state_dict(), 'model.pth')

# Save entire model (not recommended - includes file paths)
torch.save(model, 'model_full.pth')

# Save checkpoint (with optimizer state, epoch, etc.)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load state dict
model = NeuralNetwork()  # Create model architecture
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Transfer Learning

### Keras Transfer Learning

```python
# Load pre-trained model (e.g., VGG16)
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning: Unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### PyTorch Transfer Learning

```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes

# Fine-tuning: Unfreeze last few layers
for param in list(model.parameters())[-10:]:
    param.requires_grad = True

# Use different learning rates
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': [p for p in model.parameters() if p.requires_grad and p not in model.fc.parameters()], 'lr': 1e-5}
])
```

---

## Practice Exercises

### Exercise 1: MNIST with Keras

**Task:** Build and train CNN for MNIST classification.

**Solution:**
```python
# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2, batch_size=128)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Exercise 2: Same Model in PyTorch

**Task:** Build the same CNN in PyTorch.

**Solution:**
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
# ... training code ...
```

---

## PyTorch Fundamentals Deep Dive

### Understanding Deep Learning and Why PyTorch?

**What is Deep Learning?**
- Subset of machine learning using neural networks with multiple layers
- Learns hierarchical representations from data
- Excels at pattern recognition in complex data

**Why Use Machine/Deep Learning?**
- **Automatic Feature Learning**: No manual feature engineering
- **Scalability**: Handles large, complex datasets
- **Performance**: State-of-the-art results in many domains
- **Flexibility**: Adapts to various problem types

**The Number One Rule of ML:**
> **Data > Algorithms > Models**
> 
> Having good, representative data is more important than choosing the best algorithm or model architecture.

**Machine Learning vs Deep Learning:**

| Aspect | Machine Learning | Deep Learning |
|--------|------------------|--------------|
| **Feature Engineering** | Manual | Automatic |
| **Data Requirements** | Can work with smaller datasets | Needs large datasets |
| **Interpretability** | More interpretable | Less interpretable (black box) |
| **Computational Cost** | Lower | Higher (requires GPUs) |
| **Best For** | Structured data, smaller problems | Complex patterns, unstructured data |

**What Can Deep Learning Be Used For?**
- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Translation, sentiment analysis, chatbots
- **Speech Recognition**: Voice assistants, transcription
- **Recommendation Systems**: Product recommendations, content filtering
- **Time Series**: Forecasting, anomaly detection
- **Generative AI**: Text, image, code generation

**What is/Why PyTorch?**
- **Pythonic**: Feels like NumPy, intuitive API
- **Dynamic Graphs**: Build and modify graphs on-the-fly
- **Research-Friendly**: Easy to experiment and prototype
- **Strong Community**: Widely used in research and industry
- **GPU Acceleration**: Seamless CUDA integration
- **Ecosystem**: torchvision, torchaudio, torchtext

### PyTorch Tensors: The Foundation

**What are Tensors?**
Tensors are multi-dimensional arrays - the fundamental data structure in PyTorch.

```python
import torch
import numpy as np

# Creating tensors
# Scalar (0D tensor)
scalar = torch.tensor(42)
print(f"Scalar: {scalar}, Shape: {scalar.shape}")

# Vector (1D tensor)
vector = torch.tensor([1, 2, 3, 4])
print(f"Vector: {vector}, Shape: {vector.shape}")

# Matrix (2D tensor)
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"Matrix: {matrix}, Shape: {matrix.shape}")

# 3D tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor: {tensor_3d}, Shape: {tensor_3d.shape}")

# Random tensors
random_tensor = torch.rand(3, 4)  # Uniform [0, 1)
normal_tensor = torch.randn(3, 4)  # Normal distribution
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
```

**Tensor Data Types:**

```python
# Different data types
float32_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
int64_tensor = torch.tensor([1, 2], dtype=torch.int64)
bool_tensor = torch.tensor([True, False], dtype=torch.bool)

# Check data type
print(f"Data type: {float32_tensor.dtype}")
print(f"Device: {float32_tensor.device}")  # cpu or cuda
print(f"Shape: {float32_tensor.shape}")
print(f"Requires grad: {float32_tensor.requires_grad}")  # For autograd
```

**Tensor Attributes:**

```python
tensor = torch.randn(3, 4)

print(f"Shape: {tensor.shape}")  # or tensor.size()
print(f"Number of dimensions: {tensor.ndim}")
print(f"Number of elements: {tensor.numel()}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Layout: {tensor.layout}")  # Usually 'strided'
```

**Manipulating Tensors:**

```python
# Reshaping
x = torch.randn(4, 4)
x_reshaped = x.reshape(2, 8)  # or x.view(2, 8)
x_flattened = x.flatten()  # Flatten to 1D

# Stacking
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
stacked = torch.stack([a, b])  # Shape: (2, 3)

# Concatenation
concat = torch.cat([a, b], dim=0)  # Shape: (6,)

# Squeezing and Unsqueezing
x = torch.randn(1, 3, 1, 4)
x_squeezed = x.squeeze()  # Remove dims of size 1 -> (3, 4)
x_unsqueezed = x.unsqueeze(0)  # Add dimension at index 0

# Permuting (transposing)
x = torch.randn(2, 3, 4)
x_permuted = x.permute(2, 0, 1)  # New shape: (4, 2, 3)
```

**Matrix Multiplication:**

```python
# Matrix multiplication
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = torch.matmul(x, y)  # or x @ y
print(f"Result shape: {z.shape}")  # (3, 5)

# Element-wise operations
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
element_wise = a * b  # Element-wise multiplication
element_wise_sum = a + b  # Element-wise addition
```

**Finding Min, Max, Mean, Sum:**

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Min/Max
print(f"Min: {x.min()}")
print(f"Max: {x.max()}")
print(f"Min along dim 0: {x.min(dim=0)}")  # Returns (values, indices)
print(f"Argmin: {x.argmin()}")  # Index of minimum
print(f"Argmax: {x.argmax()}")  # Index of maximum

# Mean/Sum
print(f"Mean: {x.mean()}")
print(f"Sum: {x.sum()}")
print(f"Mean along dim 1: {x.mean(dim=1)}")
print(f"Sum along dim 0: {x.sum(dim=0)}")
```

**PyTorch and NumPy Interoperability:**

```python
# NumPy to PyTorch
numpy_array = np.array([1, 2, 3, 4])
torch_tensor = torch.from_numpy(numpy_array)  # Shares memory

# PyTorch to NumPy
torch_tensor = torch.tensor([1, 2, 3, 4])
numpy_array = torch_tensor.numpy()  # Shares memory if on CPU

# Note: If tensor is on GPU, need to move to CPU first
# numpy_array = torch_tensor.cpu().numpy()
```

**Reproducibility:**

```python
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# For CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Make operations deterministic (may slow down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Accessing GPU and Device-Agnostic Code:**

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move tensor to device
x = torch.randn(3, 4)
x_gpu = x.to(device)  # or x.cuda() if CUDA available

# Device-agnostic code (best practice)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork().to(device)
data = data.to(device)

# Check CUDA device properties
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## PyTorch Workflow: End-to-End

### The PyTorch Workflow Pattern

**Standard Workflow:**
1. **Prepare Data**: Load and preprocess
2. **Create Model**: Define architecture
3. **Setup Loss & Optimizer**: Choose loss function and optimizer
4. **Training Loop**: Forward pass, backward pass, update weights
5. **Evaluation**: Test on validation/test set
6. **Save/Load Model**: Persist trained models

### Creating a Dataset with Linear Regression

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create synthetic linear regression dataset
weight = 0.7  # True weight
bias = 0.3     # True bias

# Generate data
X = torch.randn(100, 1)
y = weight * X + bias + torch.randn(100, 1) * 0.1  # Add noise

# Visualize
plt.scatter(X, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Dataset')
plt.show()
```

### Creating Training and Test Sets

```python
from torch.utils.data import TensorDataset, DataLoader, random_split

# Create dataset
dataset = TensorDataset(X, y)

# Split into train and test (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
```

### Creating Your First PyTorch Model

```python
# Method 1: Using nn.Sequential
model_sequential = nn.Sequential(
    nn.Linear(in_features=1, out_features=1)
)

# Method 2: Using nn.Module (recommended)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear_layer(x)

model = LinearRegressionModel()
print(model)
print(f"Model parameters: {list(model.parameters())}")
```

**Important Model Building Classes:**

```python
# nn.Module: Base class for all neural network modules
# nn.Linear: Fully connected layer
# nn.Conv2d: 2D convolution
# nn.ReLU: ReLU activation
# nn.Sequential: Container for sequential layers
# nn.ModuleList: List of modules
# nn.ModuleDict: Dictionary of modules
```

**Checking Model Internals:**

```python
# Model state dict
print("Model state dict:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Model structure
print("\nModel structure:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
```

### Making Predictions

```python
# Set model to evaluation mode
model.eval()

# Make predictions (no gradient tracking)
with torch.no_grad():
    sample_input = torch.tensor([[0.5]])
    prediction = model(sample_input)
    print(f"Prediction for input 0.5: {prediction.item()}")
```

### Training a Model with PyTorch

```python
# Setup
loss_fn = nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()  # Zero gradients
        loss.backward()         # Compute gradients
        optimizer.step()        # Update weights
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

### Testing Loop

```python
# Evaluation
model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# Visualize predictions
model.eval()
with torch.no_grad():
    y_pred = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='True')
plt.plot(X, y_pred, 'r-', label='Predicted', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Model Predictions')
plt.show()
```

### Saving and Loading Models

```python
# Save model state dict (recommended)
torch.save(model.state_dict(), 'linear_model.pth')

# Save entire model (not recommended for production)
torch.save(model, 'linear_model_full.pth')

# Load model
# Method 1: Load state dict
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load('linear_model.pth'))
loaded_model.eval()

# Method 2: Load entire model
loaded_model_full = torch.load('linear_model_full.pth')
loaded_model_full.eval()

# Verify loaded model works
with torch.no_grad():
    test_input = torch.tensor([[0.5]])
    prediction = loaded_model(test_input)
    print(f"Loaded model prediction: {prediction.item()}")
```

---

## PyTorch Neural Network Classification

### Introduction to Classification

**Classification vs Regression:**
- **Regression**: Predict continuous values (e.g., house price)
- **Classification**: Predict discrete categories (e.g., cat vs dog)

**Classification Input and Outputs:**
- **Input**: Features (e.g., image pixels, text embeddings)
- **Output**: Class probabilities or class labels
- **Binary Classification**: 2 classes (e.g., spam/not spam)
- **Multi-class Classification**: Multiple classes (e.g., digits 0-9)

### Architecture of Classification Neural Network

```python
class ClassificationModel(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# For binary classification
binary_model = ClassificationModel(input_features=2, hidden_units=10, output_features=1)

# For multi-class classification
multiclass_model = ClassificationModel(input_features=784, hidden_units=128, output_features=10)
```

### Creating Classification Data

```python
from sklearn.datasets import make_circles
import torch

# Create binary classification dataset (circles)
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

# Convert to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Reshape y for binary classification
y = y.unsqueeze(1)  # Shape: (1000, 1)

# Split into train and test
from torch.utils.data import TensorDataset, DataLoader, random_split
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Building Classification Model

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # No activation - raw logits
        return x

model = BinaryClassifier()
```

### Using torch.nn.Sequential

```python
# Simpler way using Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
```

### Loss, Optimizer, and Evaluation Functions

```python
# Loss function for binary classification
loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
# Note: Use this when output doesn't have sigmoid activation

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Evaluation function
def accuracy_fn(y_true, y_pred):
    """Calculate accuracy"""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

### From Logits to Probabilities to Labels

```python
# Model outputs logits (raw predictions)
logits = model(X)

# Convert logits to probabilities (for binary classification)
probabilities = torch.sigmoid(logits)

# Convert probabilities to prediction labels
predictions = torch.round(probabilities)  # 0 or 1

# For multi-class classification
logits = model(X)  # Shape: (batch_size, num_classes)
probabilities = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
predictions = torch.argmax(probabilities, dim=1)  # Shape: (batch_size,)
```

### Training and Testing Loops for Classification

```python
# Training loop
num_epochs = 100
train_losses = []
train_accs = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        
        # Calculate accuracy
        probabilities = torch.sigmoid(logits)
        predictions = torch.round(probabilities)
        correct = (predictions == batch_y).sum().item()
        epoch_correct += correct
        epoch_total += len(batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    avg_acc = (epoch_correct / epoch_total) * 100
    train_losses.append(avg_loss)
    train_accs.append(avg_acc)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")

# Testing loop
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        
        probabilities = torch.sigmoid(logits)
        predictions = torch.round(probabilities)
        correct = (predictions == batch_y).sum().item()
        
        test_loss += loss.item()
        test_correct += correct
        test_total += len(batch_y)

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = (test_correct / test_total) * 100
print(f"\nTest Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.2f}%")
```

### The Missing Piece: Non-Linearity

**Why Non-Linearity?**
- Without activation functions, multiple layers = single layer (linear transformation)
- Non-linearity enables learning complex patterns
- ReLU is most common for hidden layers

```python
# Model without non-linearity (won't learn non-linear patterns)
linear_model = nn.Sequential(
    nn.Linear(2, 10),
    nn.Linear(10, 10),
    nn.Linear(10, 1)
)

# Model with non-linearity (can learn non-linear patterns)
nonlinear_model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),  # Non-linearity!
    nn.Linear(10, 10),
    nn.ReLU(),  # Non-linearity!
    nn.Linear(10, 1)
)
```

### Multi-Class Classification

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load MNIST dataset
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Multi-class classification model
class MNISTClassifier(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layer_stack(x)

model = MNISTClassifier(input_shape=28*28, hidden_units=128, output_shape=10)

# Loss for multi-class classification
loss_fn = nn.CrossEntropyLoss()  # Includes softmax
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (similar to binary, but different loss)
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)  # batch_y is class indices, not one-hot
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## PyTorch Computer Vision

### Introduction to Computer Vision

**What is Computer Vision?**
- Field of AI that enables machines to interpret visual information
- Tasks: Image classification, object detection, segmentation, etc.

**Computer Vision Input and Outputs:**
- **Input**: Images (tensors of shape `[batch, channels, height, width]`)
- **Output**: Predictions (class labels, bounding boxes, masks, etc.)

### What is a Convolutional Neural Network (CNN)?

**Why CNNs for Images?**
- **Translation Invariance**: Detect features anywhere in image
- **Parameter Sharing**: Same filters across image (fewer parameters)
- **Local Patterns**: Detect edges, shapes, textures hierarchically

**CNN Architecture:**
```
Input Image → Conv Layers → Pooling → Conv Layers → Pooling → Fully Connected → Output
```

### TorchVision

```python
import torchvision
from torchvision import datasets, transforms

# TorchVision provides:
# - Pre-trained models
# - Common datasets
# - Image transformations
# - Utilities for computer vision
```

### Getting Computer Vision Dataset

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

# Fashion-MNIST dataset
train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),  # Convert PIL Image to tensor
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# Visualize sample
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

### Mini-Batches and DataLoaders

```python
from torch.utils.data import DataLoader

# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Shuffle training data
    num_workers=0  # Number of subprocesses for data loading
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Don't shuffle test data
    num_workers=0
)

# Check batch shape
for batch_X, batch_y in train_loader:
    print(f"Batch shape: {batch_X.shape}")  # [batch_size, channels, height, width]
    print(f"Labels shape: {batch_y.shape}")  # [batch_size]
    break
```

### Training and Testing Loops for Batched Data

```python
def train_step(model, dataloader, loss_fn, optimizer, device):
    """Training step for one epoch"""
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):
    """Testing step"""
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc
```

### Running Experiments on GPU

```python
# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Training loop with GPU
for epoch in range(num_epochs):
    train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
```

### Creating a CNN

```python
class CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, 
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)  # 7*7 from pooling
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

model = CNN(input_shape=1, hidden_units=10, output_shape=10)
```

### Breaking Down nn.Conv2d and nn.MaxPool2d

```python
# Conv2d parameters
conv_layer = nn.Conv2d(
    in_channels=1,      # Input channels (grayscale=1, RGB=3)
    out_channels=10,    # Number of filters
    kernel_size=3,      # Filter size (3x3)
    stride=1,           # Step size
    padding=1           # Padding to maintain size
)

# MaxPool2d parameters
pool_layer = nn.MaxPool2d(
    kernel_size=2,      # Pooling window size
    stride=2            # Step size (usually same as kernel_size)
)

# Example: Understanding output shapes
x = torch.randn(1, 1, 28, 28)  # [batch, channels, height, width]
print(f"Input shape: {x.shape}")

# After Conv2d
x = conv_layer(x)
print(f"After Conv2d: {x.shape}")  # [1, 10, 28, 28] (with padding=1)

# After MaxPool2d
x = pool_layer(x)
print(f"After MaxPool2d: {x.shape}")  # [1, 10, 14, 14]
```

### Making Predictions on Random Test Samples

```python
def make_predictions(model, data, device):
    """Make predictions on random samples"""
    model.eval()
    image, label = data[torch.randint(0, len(data), size=(1,)).item()]
    
    with torch.no_grad():
        image = image.unsqueeze(dim=0).to(device)
        pred_logit = model(image)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
    
    return image, label, pred_label, pred_prob

# Make predictions
image, true_label, pred_label, pred_prob = make_predictions(model, test_data, device)
print(f"True label: {true_label}")
print(f"Predicted label: {pred_label.item()}")
print(f"Prediction probability: {pred_prob.max().item():.2%}")
```

---

## PyTorch Custom Datasets

### Introduction to Custom Datasets

**Why Custom Datasets?**
- Your own data (images, text, etc.)
- Specific format or structure
- Need custom preprocessing

### Creating Custom Dataset Class

```python
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self._get_label(image_path)  # Implement based on your needs
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def _get_label(self, image_path):
        # Example: Extract label from filename or directory
        # Implement based on your data structure
        return 0
```

### Data Augmentation

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Apply to dataset
train_data = CustomImageDataset(image_dir='train', transform=train_transform)
test_data = CustomImageDataset(image_dir='test', transform=test_transform)
```

---

## TensorFlow Fundamentals Deep Dive

### Deep Learning 101 with TensorFlow

**Why TensorFlow?**
- **Production-Ready**: Industry standard for deployment
- **Scalability**: Handles large-scale production systems
- **Ecosystem**: TensorFlow Serving, TensorFlow Lite, TensorFlow.js
- **Keras Integration**: High-level API for easy model building
- **Strong Community**: Extensive documentation and support

**TensorFlow Tensors:**

```python
import tensorflow as tf

# Creating tensors
scalar = tf.constant(42)
vector = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]])
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Random tensors
random_tensor = tf.random.normal(shape=(3, 4))  # Normal distribution
uniform_tensor = tf.random.uniform(shape=(3, 4), minval=0, maxval=1)

# Zeros and ones
zeros = tf.zeros(shape=(3, 4))
ones = tf.ones(shape=(3, 4))
```

**Tensor Variables:**

```python
# tf.Variable: Mutable tensors (for model parameters)
weights = tf.Variable(tf.random.normal(shape=(10, 5)))
bias = tf.Variable(tf.zeros(shape=(5,)))

# Update variable
weights.assign_add(tf.random.normal(shape=(10, 5)) * 0.1)
```

**Tensor Operations:**

```python
# Matrix multiplication
x = tf.random.normal(shape=(3, 4))
y = tf.random.normal(shape=(4, 5))
z = tf.matmul(x, y)  # or x @ y

# Element-wise operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
element_wise = a * b  # Element-wise multiplication
element_wise_sum = a + b

# Aggregation
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.reduce_mean(x))  # Mean of all elements
print(tf.reduce_sum(x, axis=0))  # Sum along axis 0
print(tf.reduce_max(x, axis=1))  # Max along axis 1
```

**Tensor Attributes:**

```python
tensor = tf.random.normal(shape=(3, 4))
print(f"Shape: {tensor.shape}")
print(f"Rank: {tf.rank(tensor)}")  # Number of dimensions
print(f"Size: {tf.size(tensor)}")  # Total number of elements
print(f"Dtype: {tensor.dtype}")
```

**TensorFlow and NumPy:**

```python
import numpy as np

# NumPy to TensorFlow
numpy_array = np.array([1, 2, 3, 4])
tf_tensor = tf.constant(numpy_array)

# TensorFlow to NumPy
tf_tensor = tf.constant([1, 2, 3, 4])
numpy_array = tf_tensor.numpy()
```

**GPU Acceleration:**

```python
# Check GPU availability
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Use GPU if available
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        # GPU operations
        x = tf.random.normal(shape=(1000, 1000))
        y = tf.random.normal(shape=(1000, 1000))
        z = x @ y
```

### TensorFlow Regression

**Regression Inputs and Outputs:**
- **Input**: Features (e.g., house size, bedrooms)
- **Output**: Continuous value (e.g., house price)

**Regression Architecture:**

```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple regression model
model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=(10,)),  # 10 features
    layers.Dense(50, activation='relu'),
    layers.Dense(1)  # Single output (regression)
])

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)
```

**Improving Regression Models:**

```python
# Method 1: Add more layers
model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=(10,)),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
])

# Method 2: Change learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# Method 3: Feature scaling (normalize inputs)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Evaluating Regression Models:**

```python
# Visualize predictions
import matplotlib.pyplot as plt

predictions = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Predictions')
plt.show()

# Metrics
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
```

### TensorFlow Classification

**Classification Inputs and Outputs:**
- **Input**: Features (e.g., image pixels)
- **Output**: Class probabilities or labels

**Classification Architecture:**

```python
# Binary classification
binary_model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=(10,)),
    layers.Dense(50, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

binary_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Multi-class classification
multiclass_model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=(784,)),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

multiclass_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)
```

**Non-Linearity in Neural Networks:**

```python
# Model without non-linearity (linear)
linear_model = keras.Sequential([
    layers.Dense(10, input_shape=(2,)),
    layers.Dense(10),
    layers.Dense(1, activation='sigmoid')
])

# Model with non-linearity (can learn non-linear patterns)
nonlinear_model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),  # ReLU activation
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

**Tuning Learning Rate:**

```python
# Learning rate callback
lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20)
)

# Find ideal learning rate
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler]
)

# Plot learning rate vs loss
lrs = 1e-4 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history['loss'])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Finding Ideal Learning Rate')
plt.show()
```

**Classification Evaluation:**

```python
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred = tf.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
```

### TensorFlow Computer Vision

**CNNs for Image Classification:**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

**Building CNN:**

```python
# CNN model
model = keras.Sequential([
    # Conv Block 1
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2),
    
    # Conv Block 2
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    
    # Conv Block 3
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(2),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Transfer Learning with TensorFlow:**

```python
from tensorflow.keras.applications import ResNet50

# Load pre-trained model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Saving and Loading Models:**

```python
# Save model
model.save('model.h5')  # HDF5 format
model.save('model.keras')  # Keras 3 format
model.save('saved_model')  # SavedModel format (for TensorFlow Serving)

# Load model
loaded_model = keras.models.load_model('model.h5')

# Save only weights
model.save_weights('weights.h5')

# Load weights (need to create model first)
new_model = create_model()
new_model.load_weights('weights.h5')
```

---

## Key Takeaways

1. **Keras**: Easier to learn, great for beginners and production
2. **PyTorch**: More flexible, research-friendly, Pythonic
3. **Both powerful**: Choose based on preference and project needs
4. **Practice**: Build models in both frameworks to understand differences
5. **Transfer Learning**: Use pre-trained models to save time
6. **Callbacks/Monitoring**: Essential for effective training
7. **GPU Support**: Both frameworks support GPU acceleration

---

## Best Practices

### General
- Always normalize/standardize input data
- Use validation split to monitor overfitting
- Implement early stopping
- Save model checkpoints
- Monitor training with TensorBoard
- Use appropriate batch sizes (32-256)

### Keras
- Start with Sequential API, move to Functional when needed
- Use callbacks for training control
- Leverage pre-trained models from keras.applications
- Use model.save() for complete model persistence

### PyTorch
- Always set model.eval() for inference
- Use DataLoader for efficient data loading
- Implement proper training/validation loops
- Use torch.no_grad() for inference to save memory
- Save state_dict() not entire model

---

## Next Steps

- Practice with both frameworks
- Build same model in both to compare
- Experiment with transfer learning
- Learn about advanced architectures
- Move to [11-computer-vision](../11-computer-vision/README.md) for CNNs

**Remember**: Frameworks make it easier, but understanding fundamentals is key! Master both for maximum flexibility.

---

## Comprehensive Learning Resources

### Official Documentation

**TensorFlow/Keras:**
- [TensorFlow Official Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Keras Guide](https://keras.io/guides/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [TensorFlow Examples](https://github.com/tensorflow/examples)

**PyTorch:**
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Learning Path](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Examples](https://github.com/pytorch/examples)
- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [PyTorch Audio](https://pytorch.org/audio/stable/index.html)

### Free Online Courses

**Comprehensive Deep Learning Courses:**
- [Fast.ai Practical Deep Learning](https://www.fast.ai/) - Completely free, top-down approach, uses PyTorch
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning) - Free audit available, uses TensorFlow/Keras
- [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)](http://cs231n.stanford.edu/) - Free, comprehensive computer vision course
- [CS224n: Natural Language Processing with Deep Learning (Stanford)](https://web.stanford.edu/class/cs224n/) - Free NLP course

**TensorFlow-Specific:**
- [TensorFlow Developer Certificate Course (Coursera)](https://www.coursera.org/professional-certificates/tensorflow-in-practice) - Free audit available
- [TensorFlow 2.0 Complete Course (YouTube - Daniel Bourke)](https://www.youtube.com/watch?v=tpCFfeUEGs8) - Free comprehensive course
- [TensorFlow Tutorials (YouTube - Sentdex)](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEFN)

**PyTorch-Specific:**
- [PyTorch for Deep Learning (YouTube - Daniel Bourke)](https://www.youtube.com/watch?v=V_xro1bcAuA) - Free comprehensive course
- [PyTorch Tutorial (YouTube - Sentdex)](https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh)
- [Deep Learning with PyTorch (Udacity)](https://www.udacity.com/course/deep-learning-pytorch--ud188) - Free course

### Video Tutorials and YouTube Channels

**Educational Channels:**
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Visual explanations of neural networks
- [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer) - Clear explanations of ML/DL concepts
- [Sentdex](https://www.youtube.com/user/sentdex) - Practical Python and deep learning tutorials
- [Daniel Bourke](https://www.youtube.com/c/mrdbourke) - PyTorch and TensorFlow tutorials
- [Aladdin Persson](https://www.youtube.com/c/AladdinPersson) - PyTorch implementation tutorials

**Conference Talks:**
- [PyTorch Developer Conference](https://pytorch.org/events/) - Annual conference with talks and tutorials
- [TensorFlow Dev Summit](https://www.tensorflow.org/dev-summit) - Annual TensorFlow conference

### Books

**Deep Learning Fundamentals:**
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio & Aaron Courville
  - [Free Online](https://www.deeplearningbook.org/)
  - Comprehensive theoretical foundation

- **"Neural Networks and Deep Learning"** by Michael Nielsen
  - [Free Online](http://neuralnetworksanddeeplearning.com/)
  - Great for understanding fundamentals

**Practical Guides:**
- **"Hands-On Machine Learning"** by Aurélien Géron
  - Covers TensorFlow/Keras extensively
  - Practical, project-based approach

- **"Deep Learning with Python"** by François Chollet
  - Written by Keras creator
  - Excellent for Keras/TensorFlow

- **"Programming PyTorch for Deep Learning"** by Ian Pointer
  - Practical PyTorch guide

### Practice Platforms

**Kaggle:**
- [Kaggle Learn - Deep Learning](https://www.kaggle.com/learn/deep-learning) - Free micro-courses
- [Kaggle Competitions](https://www.kaggle.com/competitions) - Practice on real datasets
- [Kaggle Notebooks](https://www.kaggle.com/code) - Explore community notebooks

**Other Platforms:**
- [Papers With Code](https://paperswithcode.com/) - Latest research with implementations
- [Hugging Face](https://huggingface.co/) - Pre-trained models and datasets
- [Google Colab](https://colab.research.google.com/) - Free GPU access for practice
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and visualization

### Community and Forums

**Discussion Forums:**
- [PyTorch Forums](https://discuss.pytorch.org/) - Official PyTorch community
- [TensorFlow Forums](https://discuss.tensorflow.org/) - Official TensorFlow community
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch) - Tag: pytorch, tensorflow
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - ML research and discussions
- [Reddit r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/) - Learning-focused community

**GitHub Resources:**
- [Awesome PyTorch](https://github.com/bharathgs/Awesome-pytorch-list) - Curated PyTorch resources
- [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) - Curated TensorFlow resources
- [PyTorch Examples](https://github.com/pytorch/examples) - Official examples
- [TensorFlow Examples](https://github.com/tensorflow/examples) - Official examples

### Datasets for Practice

**Image Classification:**
- [MNIST](http://yann.lecun.com/exdb/mnist/) - Handwritten digits
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) - Natural images
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - Clothing items
- [ImageNet](https://www.image-net.org/) - Large-scale image database

**Computer Vision:**
- [COCO](https://cocodataset.org/) - Object detection and segmentation
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - Object detection
- [Open Images](https://storage.googleapis.com/openimages/web/index.html) - Large-scale dataset

**Text/NLP:**
- [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) - Sentiment analysis
- [AG News](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) - News classification
- [GLUE Benchmark](https://gluebenchmark.com/) - NLP tasks

### Tools and Libraries

**Development Tools:**
- [Jupyter Notebook](https://jupyter.org/) - Interactive development
- [VS Code](https://code.visualstudio.com/) - Popular IDE with ML extensions
- [PyCharm](https://www.jetbrains.com/pycharm/) - Python IDE

**Experiment Tracking:**
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Built-in visualization
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [MLflow](https://mlflow.org/) - ML lifecycle management

**Model Deployment:**
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - Model serving
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Mobile/edge deployment
- [TorchServe](https://pytorch.org/serve/) - PyTorch model serving
- [ONNX](https://onnx.ai/) - Model interoperability

### Cheat Sheets and Quick References

- [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
- [TensorFlow 2.0 Quick Reference](https://www.tensorflow.org/api_docs/python/tf)
- [Keras Cheat Sheet](https://www.datacamp.com/cheat-sheet/keras-cheat-sheet-deep-learning-in-python)

### Best Practices and Style Guides

- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [TensorFlow Best Practices](https://www.tensorflow.org/guide/best_practices)
- [Google's Machine Learning Style Guide](https://google.github.io/styleguide/pyguide.html)

### Research Papers (Important)

**Foundational Papers:**
- "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, 2012)
- "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG, 2014)
- "Deep Residual Learning for Image Recognition" (ResNet, 2015)
- "Attention Is All You Need" (Transformer, 2017)

**Framework Papers:**
- "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems" (2016)
- "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)

### Getting Help

**When Stuck:**
1. Check official documentation first
2. Search Stack Overflow with specific error messages
3. Ask on official forums (PyTorch/TensorFlow)
4. Check GitHub issues for similar problems
5. Review example code in official repositories

**Learning Path Recommendation:**
1. Start with one framework (Keras for beginners, PyTorch for research)
2. Complete a full project end-to-end
3. Learn the other framework
4. Build same project in both frameworks
5. Explore advanced topics (custom layers, distributed training, etc.)

---

**Remember**: The best way to learn is by doing! Start with simple projects and gradually increase complexity. Both frameworks are powerful - mastering both gives you maximum flexibility in your ML career.
