# Deep Learning Frameworks Complete Guide

Comprehensive guide to TensorFlow/Keras and PyTorch for building deep learning models.

## Table of Contents

- [TensorFlow/Keras](#tensorflowkeras)
- [PyTorch](#pytorch)
- [Model Building](#model-building)
- [Training](#training)
- [Model Saving and Loading](#model-saving-and-loading)
- [Practice Exercises](#practice-exercises)

---

## TensorFlow/Keras

### Installation

```python
pip install tensorflow
```

### Sequential API

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

# Summary
model.summary()
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

### Training

```python
# Load data (MNIST example)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# Train
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")
```

---

## PyTorch

### Installation

```python
pip install torch torchvision
```

### Building Models

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)
```

### Training Loop

```python
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## Model Saving and Loading

### Keras

```python
# Save
model.save('model.h5')
model.save_weights('weights.h5')

# Load
model = keras.models.load_model('model.h5')
model.load_weights('weights.h5')
```

### PyTorch

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
```

---

## Practice Exercises

### Exercise 1: MNIST with Keras

**Task:** Build and train CNN for MNIST.

**Solution:**
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

## Key Takeaways

1. **Keras**: Easier, great for beginners
2. **PyTorch**: More flexible, research-friendly
3. **Both powerful**: Choose based on preference
4. **Practice**: Build models in both frameworks

---

## Next Steps

- Practice with both frameworks
- Move to [11-computer-vision](../11-computer-vision/README.md) for CNNs

**Remember**: Frameworks make it easier, but understanding fundamentals is key!

