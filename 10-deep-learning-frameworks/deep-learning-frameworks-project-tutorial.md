# Complete Deep Learning Framework Project Tutorial

Step-by-step walkthrough of building and training a neural network using both Keras and PyTorch.

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1: Keras Implementation](#part-1-keras-implementation)
- [Part 2: PyTorch Implementation](#part-2-pytorch-implementation)
- [Part 3: Comparison and Analysis](#part-3-comparison-and-analysis)

---

## Project Overview

**Project**: Image Classification with MNIST

**Dataset**: MNIST handwritten digits (28x28 grayscale images, 10 classes)

**Goals**:
1. Build same model in both Keras and PyTorch
2. Train and evaluate both models
3. Compare frameworks
4. Understand differences and similarities

**Type**: Image Classification

**Difficulty**: Intermediate

**Time**: 2-3 hours

---

## Part 1: Keras Implementation

### Step 1: Setup and Data Loading

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training set: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test set: {x_test.shape}, Labels: {y_test.shape}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(x_train[i], cmap='gray')
    axes[row, col].set_title(f'Label: {y_train[i]}', fontsize=12)
    axes[row, col].axis('off')
plt.suptitle('Sample MNIST Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Step 2: Data Preprocessing

```python
# Reshape and normalize
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# No need to one-hot encode (using sparse_categorical_crossentropy)
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Data range: [{x_train.min():.2f}, {x_train.max():.2f}]")
```

### Step 3: Build Model

```python
# Build Sequential model
model_keras = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,), name='hidden1'),
    layers.Dropout(0.2, name='dropout1'),
    layers.Dense(64, activation='relu', name='hidden2'),
    layers.Dropout(0.2, name='dropout2'),
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_Classifier')

# Compile model
model_keras.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model_keras.summary()

# Visualize architecture
keras.utils.plot_model(model_keras, to_file='keras_model.png', show_shapes=True)
```

### Step 4: Train Model

```python
# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_keras_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs/keras',
        histogram_freq=1
    )
]

# Train model
history_keras = model_keras.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

### Step 5: Evaluate and Visualize

```python
# Evaluate on test set
test_loss_keras, test_acc_keras = model_keras.evaluate(x_test, y_test, verbose=0)
print(f"\nKeras Model Results:")
print(f"Test Loss: {test_loss_keras:.4f}")
print(f"Test Accuracy: {test_acc_keras:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_keras.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history_keras.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Keras Training Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_keras.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history_keras.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Keras Training Accuracy', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Make predictions
predictions_keras = model_keras.predict(x_test[:10])
predicted_classes_keras = np.argmax(predictions_keras, axis=1)

print("\nSample Predictions (Keras):")
for i in range(10):
    print(f"  Image {i}: True={y_test[i]}, Predicted={predicted_classes_keras[i]}, "
          f"Confidence={predictions_keras[i][predicted_classes_keras[i]]:.3f}")
```

---

## Part 2: PyTorch Implementation

### Step 1: Setup and Data Loading

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Load MNIST (using same data from Keras)
# Convert to PyTorch tensors
x_train_torch = torch.FloatTensor(x_train)
y_train_torch = torch.LongTensor(y_train)
x_test_torch = torch.FloatTensor(x_test)
y_test_torch = torch.LongTensor(y_test)

print(f"Training set: {x_train_torch.shape}, Labels: {y_train_torch.shape}")
print(f"Test set: {x_test_torch.shape}, Labels: {y_test_torch.shape}")
```

### Step 2: Create DataLoaders

```python
# Create datasets
train_dataset = TensorDataset(x_train_torch, y_train_torch)
test_dataset = TensorDataset(x_test_torch, y_test_torch)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")
```

### Step 3: Build Model

```python
# Define model class
class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create model
model_pytorch = MNISTClassifier(input_size=784, hidden_sizes=[128, 64], num_classes=10)
print(model_pytorch)

# Count parameters
total_params = sum(p.numel() for p in model_pytorch.parameters())
trainable_params = sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_pytorch = model_pytorch.to(device)
print(f"Using device: {device}")
```

### Step 4: Train Model

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training
num_epochs = 20
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# Split validation set
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

best_val_loss = float('inf')
patience_counter = 0
patience = 5

for epoch in range(num_epochs):
    # Training phase
    model_pytorch.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward
        outputs = model_pytorch(batch_x)
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
    model_pytorch.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model_pytorch(batch_x)
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
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_pytorch.state_dict(), 'best_pytorch_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model_pytorch.load_state_dict(torch.load('best_pytorch_model.pth'))
            break
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
```

### Step 5: Evaluate and Visualize

```python
# Load best model
model_pytorch.load_state_dict(torch.load('best_pytorch_model.pth'))
model_pytorch.eval()

# Evaluate on test set
test_loss_pytorch = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model_pytorch(batch_x)
        loss = criterion(outputs, batch_y)
        
        test_loss_pytorch += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()

test_loss_pytorch /= len(test_loader)
test_acc_pytorch = 100 * test_correct / test_total

print(f"\nPyTorch Model Results:")
print(f"Test Loss: {test_loss_pytorch:.4f}")
print(f"Test Accuracy: {test_acc_pytorch:.2f}%")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('PyTorch Training Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Training Accuracy', linewidth=2)
axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('PyTorch Training Accuracy', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Make predictions
model_pytorch.eval()
with torch.no_grad():
    test_samples = x_test_torch[:10].to(device)
    predictions_pytorch = model_pytorch(test_samples)
    predicted_probs_pytorch = F.softmax(predictions_pytorch, dim=1)
    predicted_classes_pytorch = torch.argmax(predicted_probs_pytorch, dim=1).cpu().numpy()

print("\nSample Predictions (PyTorch):")
for i in range(10):
    print(f"  Image {i}: True={y_test[i]}, Predicted={predicted_classes_pytorch[i]}, "
          f"Confidence={predicted_probs_pytorch[i][predicted_classes_pytorch[i]]:.3f}")
```

---

## Part 3: Comparison and Analysis

### Framework Comparison

```python
print("=" * 60)
print("FRAMEWORK COMPARISON")
print("=" * 60)

print(f"\nModel Performance:")
print(f"  Keras - Test Accuracy: {test_acc_keras:.4f}")
print(f"  PyTorch - Test Accuracy: {test_acc_pytorch:.2f}%")

print(f"\nCode Complexity:")
print(f"  Keras: ~20 lines for model + training")
print(f"  PyTorch: ~50 lines for model + training")

print(f"\nEase of Use:")
print(f"  Keras: Very easy - high-level API")
print(f"  PyTorch: Moderate - more control, more code")

print(f"\nFlexibility:")
print(f"  Keras: Good for standard architectures")
print(f"  PyTorch: Excellent for custom architectures")

print(f"\nDebugging:")
print(f"  Keras: Moderate - less visibility into internals")
print(f"  PyTorch: Excellent - full Python control flow")

print(f"\nProduction:")
print(f"  Keras: Excellent - TensorFlow Serving")
print(f"  PyTorch: Good - TorchScript, ONNX")

print(f"\nResearch:")
print(f"  Keras: Good - functional API")
print(f"  PyTorch: Excellent - dynamic graphs, research-friendly")
```

### Side-by-Side Code Comparison

```python
print("\n" + "=" * 60)
print("CODE COMPARISON")
print("=" * 60)

print("\n1. Model Definition:")
print("\nKeras (Sequential):")
print("""
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
""")

print("PyTorch (Class-based):")
print("""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
""")

print("\n2. Training:")
print("\nKeras:")
print("""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
""")

print("PyTorch:")
print("""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
""")
```

### Key Takeaways

```python
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("""
1. Both frameworks achieve similar results
2. Keras is easier for beginners and rapid prototyping
3. PyTorch offers more control and flexibility
4. Choose based on:
   - Experience level (beginners → Keras)
   - Project needs (research → PyTorch, production → Keras)
   - Team preferences
   - Ecosystem requirements

5. Best practice: Learn both!
   - Start with Keras for quick projects
   - Use PyTorch when you need flexibility
   - Master both for maximum career flexibility
""")
```

---

## Key Takeaways

1. **Both frameworks work**: Achieve similar results with different approaches
2. **Keras advantages**: Easier, faster to prototype, great for production
3. **PyTorch advantages**: More flexible, better for research, Pythonic
4. **Learn both**: Maximum flexibility and career options
5. **Choose wisely**: Based on project needs, not just preference

---

**Congratulations!** You've built the same model in both frameworks and understand their differences!

