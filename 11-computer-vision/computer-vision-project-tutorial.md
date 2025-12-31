# Complete Computer Vision Project Tutorial

Step-by-step walkthrough of building an image classification system with CNNs.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Exploration](#step-1-data-loading-and-exploration)
- [Step 2: Data Preprocessing](#step-2-data-preprocessing)
- [Step 3: Build CNN Model](#step-3-build-cnn-model)
- [Step 4: Train with Data Augmentation](#step-4-train-with-data-augmentation)
- [Step 5: Transfer Learning](#step-5-transfer-learning)
- [Step 6: Evaluate and Improve](#step-6-evaluate-and-improve)

---

## Project Overview

**Project**: CIFAR-10 Image Classification

**Dataset**: CIFAR-10 (32x32 color images, 10 classes)

**Goals**:
1. Build CNN from scratch
2. Apply data augmentation
3. Use transfer learning
4. Achieve high accuracy

**Type**: Image Classification

**Difficulty**: Intermediate

**Time**: 2-3 hours

---

## Step 1: Data Loading and Exploration

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")
print(f"Classes: {np.unique(y_train)}")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(x_train[i])
    axes[row, col].set_title(f'{class_names[y_train[i][0]]}', fontsize=10)
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
```

---

## Step 2: Data Preprocessing

```python
# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels (optional, can use sparse_categorical_crossentropy)
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

print(f"Data range: [{x_train.min():.2f}, {x_train.max():.2f}]")
```

---

## Step 3: Build CNN Model

```python
# Build CNN
model = keras.Sequential([
    # Data augmentation
    layers.RandomRotation(0.1, input_shape=(32, 32, 3)),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip('horizontal'),
    
    # Convolutional blocks
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

---

## Step 4: Train with Data Augmentation

```python
# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_cifar10_model.h5', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
]

# Train
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## Step 5: Transfer Learning

```python
from tensorflow.keras.applications import ResNet50

# Load pre-trained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)  # Note: ResNet expects 224x224, but we'll adapt
)

base_model.trainable = False

# Add classifier
transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

transfer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
transfer_model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

---

## Step 6: Evaluate and Improve

```python
# Make predictions
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)

# Visualize predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(x_test[i])
    true_label = class_names[y_test[i][0]]
    pred_label = class_names[predicted_classes[i]]
    color = 'green' if true_label == pred_label else 'red'
    axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            fontsize=9, color=color)
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
```

---

**Congratulations!** You've built a complete image classification system!

