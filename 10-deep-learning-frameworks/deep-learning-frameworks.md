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

