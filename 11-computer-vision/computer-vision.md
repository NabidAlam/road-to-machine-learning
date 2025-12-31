# Computer Vision Complete Guide

Comprehensive guide to Convolutional Neural Networks (CNNs) and image processing.

## Table of Contents

- [Introduction to CNNs](#introduction-to-cnns)
- [Building CNNs](#building-cnns)
- [CNN Architectures](#cnn-architectures)
- [Transfer Learning](#transfer-learning)
- [Data Augmentation](#data-augmentation)
- [Image Preprocessing](#image-preprocessing)
- [Object Detection Basics](#object-detection-basics)
- [Practice Exercises](#practice-exercises)

---

## Introduction to CNNs

### Why CNNs for Images?

Convolutional Neural Networks are specifically designed for image data and excel at detecting spatial patterns.

**Key Advantages:**
- **Translation Invariance**: Detect features anywhere in the image
- **Parameter Sharing**: Same filters used across image (fewer parameters)
- **Local Patterns**: Detect edges, shapes, textures, objects
- **Hierarchical Features**: Low-level (edges) → High-level (objects)

**Why Not Fully Connected?**
- Too many parameters (e.g., 28x28 image = 784 inputs → millions of parameters)
- Doesn't leverage spatial structure
- Not translation invariant

### CNN Architecture

```
Input Image → Conv Layers → Pooling → Conv Layers → Pooling → Fully Connected → Output
     ↓              ↓            ↓           ↓            ↓            ↓            ↓
  (H×W×C)    Feature Maps   Downsample  Feature Maps  Downsample   Features   Predictions
```

**Key Components:**
1. **Convolutional Layers**: Detect features using filters
2. **Pooling Layers**: Reduce spatial dimensions
3. **Fully Connected Layers**: Final classification

### How Convolution Works

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Example: Edge detection
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]])

# Vertical edge detector
vertical_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

# Apply convolution
edge_detected = ndimage.convolve(image, vertical_filter)

print("Original Image:")
print(image)
print("\nEdge Detector Filter:")
print(vertical_filter)
print("\nConvolved Output:")
print(edge_detected)
```

---

## Building CNNs

### Understanding Convolutional Layers

**Convolution Operation:**
- **Filter/Kernel**: Small matrix that slides over image
- **Stride**: How many pixels filter moves each step
- **Padding**: Add zeros around image (same/valid)
- **Output Size**: `(input_size - filter_size + 2*padding) / stride + 1`

**Key Parameters:**
- **Filters**: Number of feature maps to learn
- **Kernel Size**: Size of filter (e.g., 3x3, 5x5)
- **Stride**: Step size (default: 1)
- **Padding**: 'same' (preserve size) or 'valid' (no padding)

### Simple CNN with Keras

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# CNN for image classification
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.BatchNormalization(name='bn1'),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.BatchNormalization(name='bn2'),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
    layers.Dropout(0.25, name='dropout1'),
    
    # Flatten and classify
    layers.Flatten(name='flatten'),
    layers.Dense(64, activation='relu', name='fc1'),
    layers.Dropout(0.5, name='dropout2'),
    layers.Dense(10, activation='softmax', name='output')
], name='CNN_Classifier')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Visualize architecture
keras.utils.plot_model(model, to_file='cnn_architecture.png', show_shapes=True)

# Calculate parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")
```

### CNN with PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(F.relu(self.conv3(x)))
        
        # Fully connected
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Understanding Pooling Layers

**Max Pooling**: Takes maximum value in each window
- Reduces spatial dimensions
- Provides translation invariance
- Reduces parameters

**Average Pooling**: Takes average value in each window
- Alternative to max pooling
- Sometimes better for certain tasks

```python
# Visualize pooling effect
import numpy as np

# Example feature map
feature_map = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])

# Max pooling (2x2)
max_pooled = np.array([[6, 8],
                      [14, 16]])

# Average pooling (2x2)
avg_pooled = np.array([[3.5, 5.5],
                      [11.5, 13.5]])

print("Original feature map:")
print(feature_map)
print("\nMax pooled:")
print(max_pooled)
print("\nAverage pooled:")
print(avg_pooled)
```

---

## CNN Architectures

### LeNet-5

Early CNN architecture for digit recognition.

```python
def create_lenet5():
    model = keras.Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model
```

### Modern Architectures

**ResNet**: Residual connections solve vanishing gradient problem

```python
# ResNet block with skip connection
def resnet_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.ReLU()(x)
    return x
```

## Transfer Learning

### Why Transfer Learning?

- **Limited Data**: Pre-trained models learned from millions of images
- **Faster Training**: Start from good weights
- **Better Performance**: Often outperforms training from scratch
- **Time Saving**: Don't need to train from scratch

### Using Pre-trained Models

```python
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# Method 1: Feature Extraction (Freeze base)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze all layers

# Add custom classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Method 2: Fine-tuning (Unfreeze some layers)
base_model.trainable = True
# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Available Pre-trained Models

| Model | Parameters | Top-1 Accuracy | Use Case |
|-------|------------|----------------|----------|
| **VGG16** | 138M | 71.3% | Good baseline |
| **ResNet50** | 25M | 76.0% | Deep, accurate |
| **MobileNetV2** | 3.4M | 71.3% | Mobile/edge devices |
| **EfficientNet** | 5.3M | 77.1% | Best accuracy/size ratio |
| **InceptionV3** | 23M | 78.0% | High accuracy |

```python
# Compare different architectures
models_to_try = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'MobileNetV2': MobileNetV2
}

for name, model_class in models_to_try.items():
    base_model = model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    
    print(f"{name}: {model.count_params():,} parameters")
```

---

## Data Augmentation

### Why Data Augmentation?

- **More Training Data**: Artificially increase dataset size
- **Reduce Overfitting**: Model sees more variations
- **Better Generalization**: Model learns robust features
- **Handle Imbalance**: Augment minority classes more

### Image Augmentation Techniques

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Comprehensive augmentation
datagen = ImageDataGenerator(
    rotation_range=20,           # Rotate images up to 20 degrees
    width_shift_range=0.2,       # Shift horizontally by 20%
    height_shift_range=0.2,      # Shift vertically by 20%
    shear_range=0.2,             # Apply shearing transformation
    zoom_range=0.2,              # Zoom in/out by 20%
    horizontal_flip=True,        # Flip horizontally
    vertical_flip=False,         # Don't flip vertically (for most images)
    fill_mode='nearest',         # Fill pixels outside boundaries
    brightness_range=[0.8, 1.2], # Adjust brightness
    rescale=1./255               # Normalize pixel values
)

# Visualize augmented images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

# Original image
original = x_train[0]
axes[0].imshow(original, cmap='gray')
axes[0].set_title('Original', fontsize=10, fontweight='bold')
axes[0].axis('off')

# Generate augmented images
augmented = datagen.flow(np.expand_dims(original, 0), batch_size=1)
for i in range(1, 10):
    aug_img = next(augmented)[0].squeeze()
    axes[i].imshow(aug_img, cmap='gray')
    axes[i].set_title(f'Augmented {i}', fontsize=10, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Using Augmentation in Training

```python
# Method 1: Using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow(
    x_train, y_train,
    batch_size=32,
    subset='training'
)

val_generator = train_datagen.flow(
    x_train, y_train,
    batch_size=32,
    subset='validation'
)

# Train with generator
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Method 2: Using layers (Keras 3+)
augmentation_layers = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip('horizontal'),
    layers.RandomZoom(0.1)
])

# Add to model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    augmentation_layers,
    # ... rest of model
])
```

### Advanced Augmentation with Albumentations

```python
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # More advanced augmentations
    transform = A.Compose([
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Apply to image
    augmented = transform(image=image)['image']
except ImportError:
    print("Install albumentations: pip install albumentations")
```

---

## Practice Exercises

### Exercise 1: CIFAR-10 Classification

**Task:** Build CNN to classify CIFAR-10 images with data augmentation.

**Solution:**
```python
from tensorflow.keras.datasets import cifar10

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build improved CNN
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

# Train with callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
]

history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=128,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Exercise 2: Transfer Learning Project

**Task:** Use pre-trained model for custom image classification.

**Solution:**
```python
# Load pre-trained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base
base_model.trainable = False

# Add classifier
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Train feature extraction phase
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Fine-tuning phase
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5)
```

---

## Key Takeaways

1. **CNNs**: Best for image data - leverage spatial structure
2. **Architecture Matters**: Deeper networks learn more complex features
3. **Transfer Learning**: Use pre-trained models for better performance
4. **Data Augmentation**: Essential for preventing overfitting
5. **Batch Normalization**: Stabilizes training in deep networks
6. **Pooling**: Reduces spatial dimensions and parameters
7. **Practice**: Work with real image datasets (MNIST, CIFAR-10, ImageNet)

---

## Best Practices

### CNN Design
- Start with simple architecture
- Use small filters (3x3) in multiple layers
- Add batch normalization after conv layers
- Use dropout to prevent overfitting
- Use data augmentation

### Training
- Use appropriate learning rate (start with 0.001)
- Use learning rate scheduling
- Monitor training and validation metrics
- Use early stopping
- Save best model checkpoints

### Transfer Learning
- Start with feature extraction (freeze base)
- Fine-tune with lower learning rate
- Unfreeze only top layers initially
- Use appropriate pre-trained model for your task

---

## Next Steps

- Practice with image datasets (MNIST, CIFAR-10, custom)
- Experiment with different architectures
- Try transfer learning on your own dataset
- Learn about advanced architectures (ResNet, EfficientNet)
- Explore object detection and segmentation
- Move to [12-natural-language-processing](../12-natural-language-processing/README.md)

**Remember**: CNNs revolutionized computer vision! Practice with real datasets to master them.

