# Computer Vision Quick Reference Guide

Quick reference for CNN architectures, code snippets, and best practices.

## Table of Contents

- [CNN Basics](#cnn-basics)
- [Code Snippets](#code-snippets)
- [Architecture Comparison](#architecture-comparison)
- [Transfer Learning](#transfer-learning)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## CNN Basics

### Architecture Pattern

```
Input → Conv → BN → ReLU → Pool → Conv → BN → ReLU → Pool → FC → Output
```

### Key Components

- **Conv2D**: Feature detection
- **MaxPooling2D**: Downsampling
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

| Architecture | Parameters | Accuracy | Use Case |
|--------------|------------|----------|----------|
| **LeNet** | 60K | Low | Simple tasks |
| **VGG16** | 138M | Medium | Good baseline |
| **ResNet50** | 25M | High | General purpose |
| **MobileNetV2** | 3.4M | Medium | Mobile devices |
| **EfficientNet** | 5.3M | Very High | Best efficiency |

---

## Common Issues & Solutions

### Issue 1: Overfitting

**Solution**: Data augmentation, dropout, early stopping

### Issue 2: Slow Training

**Solution**: Use GPU, reduce image size, efficient architectures

### Issue 3: Low Accuracy

**Solution**: Transfer learning, more data, better architecture

---

## Best Practices Checklist

- [ ] Normalize pixel values (0-1)
- [ ] Use data augmentation
- [ ] Add batch normalization
- [ ] Use dropout for regularization
- [ ] Try transfer learning
- [ ] Monitor training curves
- [ ] Use appropriate image size

---

**Remember**: CNNs excel at image data - leverage spatial structure!

