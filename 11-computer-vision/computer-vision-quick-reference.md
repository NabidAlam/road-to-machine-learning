# Computer Vision Quick Reference Guide

Quick reference for CNN architectures, code snippets, and best practices.

## Table of Contents

- [Image Fundamentals](#image-fundamentals)
- [Convolution Basics](#convolution-basics)
- [CNN Architecture](#cnn-architecture)
- [Code Snippets](#code-snippets)
- [Architecture Comparison](#architecture-comparison)
- [Object Detection](#object-detection)
- [Segmentation](#segmentation)
- [Transfer Learning](#transfer-learning)
- [Training Optimization](#training-optimization)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Image Fundamentals

### Image Structure

- **Grayscale**: `(height, width, 1)` - Single channel, values 0-255
- **RGB**: `(height, width, 3)` - Three channels (Red, Green, Blue)
- **Normalized**: Scale to 0-1 range: `image / 255.0`

### Common Image Sizes

- **MNIST**: 28×28 grayscale
- **CIFAR-10**: 32×32 RGB
- **ImageNet**: 224×224 RGB (standard for transfer learning)

## Convolution Basics

### Convolution Operation

- **Filter/Kernel**: Small matrix (e.g., 3×3) that slides over image
- **Stride**: How many pixels filter moves (default: 1)
- **Padding**: 
  - `'valid'`: No padding (output smaller)
  - `'same'`: Padding to preserve size
  - For 3×3 kernel: `padding=1` gives same size

### Output Size Formula

```
Output = (Input - Filter + 2×Padding) / Stride + 1
```

### Edge Detection Filters

```python
# Sobel (vertical edges)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Canny (OpenCV)
edges = cv2.Canny(image, threshold1=50, threshold2=150)
```

### Pooling

- **Max Pooling**: Takes maximum value (most common)
- **Average Pooling**: Takes average value
- **Global Pooling**: Pools entire feature map to single value

## CNN Architecture

### Architecture Pattern

```
Input → Conv → BN → ReLU → Pool → Conv → BN → ReLU → Pool → FC → Output
```

### Key Components

- **Conv2D**: Feature detection (learns filters)
- **MaxPooling2D**: Downsampling (reduces spatial dimensions)
- **BatchNormalization**: Stabilize training
- **Dropout**: Prevent overfitting
- **Flatten**: Convert to 1D
- **Dense**: Final classification

---

## Code Snippets

### Basic CNN (Keras)

```python
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Basic CNN (PyTorch)

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x
```

### Transfer Learning

```python
# Keras
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False
model = keras.Sequential([base_model, layers.GlobalAveragePooling2D(), layers.Dense(10, activation='softmax')])
```

---

## Architecture Comparison

| Architecture | Year | Parameters | Top-1 Acc | Key Innovation | Use Case |
|--------------|------|------------|-----------|----------------|----------|
| **LeNet-5** | 1998 | 60K | - | First practical CNN | Digit recognition |
| **AlexNet** | 2012 | 60M | 63% | ReLU, Dropout, GPU training | ImageNet breakthrough |
| **VGG16** | 2014 | 138M | 71% | Small 3×3 filters, deep | Good baseline |
| **ResNet50** | 2015 | 25M | 76% | Residual connections | General purpose |
| **MobileNetV2** | 2018 | 3.4M | 71% | Depthwise separable conv | Mobile devices |
| **EfficientNet** | 2019 | 5.3M | 77% | Compound scaling | Best efficiency |

### Key Architecture Details

**LeNet-5**: 2 conv + 2 pooling + 3 FC layers, tanh activation

**AlexNet**: 5 conv + 3 FC, ReLU, dropout, data augmentation

**VGG16**: 13 conv + 3 FC, all 3×3 filters, very deep

**ResNet**: Skip connections solve vanishing gradient, enables 100+ layers

---

## Common Issues & Solutions

### Issue 1: Overfitting

**Solution**: Data augmentation, dropout, early stopping

### Issue 2: Slow Training

**Solution**: Use GPU, reduce image size, efficient architectures

### Issue 3: Low Accuracy

**Solution**: Transfer learning, more data, better architecture

---

## Object Detection

### YOLO (Quick Start)

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Detect
results = model('image.jpg')

# Train
model.train(data='dataset.yaml', epochs=100)
```

### Detection vs Classification

- **Classification**: "What is in the image?" (single label)
- **Detection**: "What and where?" (bounding boxes + labels)
- **Segmentation**: "Which pixels belong to what?" (pixel-level)

## Segmentation

### Types

- **Semantic**: Classify each pixel (no instance distinction)
- **Instance**: Identify and segment individual objects
- **Panoptic**: Combines semantic + instance

### U-Net Architecture

```python
# Encoder-Decoder with skip connections
# Encoder: Downsample (VGG16, ResNet)
# Decoder: Upsample + concatenate skip connections
# Output: Pixel-wise classification
```

---

## Training Optimization

### Callbacks

```python
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

### Batch Normalization

```python
# Add after Conv2D, before activation
layers.Conv2D(32, 3),
layers.BatchNormalization(),
layers.ReLU()
```

### Data Augmentation

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

---

## Best Practices Checklist

- [ ] Normalize pixel values (0-1 or standardized)
- [ ] Use data augmentation (rotation, flip, zoom)
- [ ] Add batch normalization after conv layers
- [ ] Use dropout for regularization (0.25-0.5)
- [ ] Try transfer learning (VGG16, ResNet50, EfficientNet)
- [ ] Monitor training curves (loss, accuracy)
- [ ] Use appropriate image size (224×224 for ImageNet models)
- [ ] Use callbacks (early stopping, LR scheduling)
- [ ] Start with simple architecture, then go deeper
- [ ] Use GPU for training (much faster)

---

## Quick Tips

1. **Start with transfer learning** - Pre-trained models work great
2. **Use data augmentation** - Essential for preventing overfitting
3. **Batch normalization** - Stabilizes training in deep networks
4. **Monitor learning curves** - Diagnose overfitting/underfitting
5. **Use appropriate architecture** - MobileNet for mobile, ResNet for accuracy
6. **Normalize images** - Scale to 0-1 or use ImageNet stats
7. **Use callbacks** - Early stopping saves time
8. **Try different optimizers** - Adam is good default
9. **Visualize feature maps** - Understand what model learns
10. **Test on diverse data** - Ensure generalization

---

**Remember**: CNNs excel at image data - leverage spatial structure! Start simple, use transfer learning, and iterate.

