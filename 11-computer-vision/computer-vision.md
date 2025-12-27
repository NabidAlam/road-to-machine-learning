# Computer Vision Complete Guide

Comprehensive guide to Convolutional Neural Networks (CNNs) and image processing.

## Table of Contents

- [Introduction to CNNs](#introduction-to-cnns)
- [Building CNNs](#building-cnns)
- [Transfer Learning](#transfer-learning)
- [Data Augmentation](#data-augmentation)
- [Object Detection Basics](#object-detection-basics)
- [Practice Exercises](#practice-exercises)

---

## Introduction to CNNs

### Why CNNs for Images?

- **Translation Invariance**: Detect features anywhere
- **Parameter Sharing**: Fewer parameters than fully connected
- **Local Patterns**: Detect edges, shapes, objects

### CNN Architecture

```
Input Image → Conv Layers → Pooling → Conv Layers → Pooling → Fully Connected → Output
```

---

## Building CNNs

### Simple CNN with Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

# CNN for image classification
model = keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and classify
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### CNN with PyTorch

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN()
```

---

## Transfer Learning

### Using Pre-trained Models

```python
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16
base_model = VGG16(
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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Data Augmentation

### Image Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Apply to training data
train_generator = datagen.flow(x_train, y_train, batch_size=32)
```

---

## Practice Exercises

### Exercise 1: CIFAR-10 Classification

**Task:** Build CNN to classify CIFAR-10 images.

**Solution:**
```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
```

---

## Key Takeaways

1. **CNNs**: Best for image data
2. **Transfer Learning**: Use pre-trained models
3. **Data Augmentation**: Increase dataset size
4. **Practice**: Work with real image datasets

---

## Next Steps

- Practice with image datasets
- Experiment with architectures
- Move to [12-natural-language-processing](../12-natural-language-processing/README.md)

**Remember**: CNNs revolutionized computer vision!

