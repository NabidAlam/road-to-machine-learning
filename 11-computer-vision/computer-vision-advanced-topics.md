# Advanced Computer Vision Topics

Comprehensive guide to advanced computer vision techniques and architectures.

## Table of Contents

- [Advanced CNN Architectures](#advanced-cnn-architectures)
- [Advanced Data Augmentation](#advanced-data-augmentation)
- [Object Detection and Segmentation](#object-detection-and-segmentation)
- [Image Generation](#image-generation)
- [Model Optimization](#model-optimization)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced CNN Architectures

### ResNet (Residual Networks)

Skip connections solve vanishing gradient problem.

```python
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3):
    """ResNet residual block"""
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (if dimensions don't match, use 1x1 conv)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Build ResNet-like model
inputs = keras.Input(shape=(224, 224, 3))
x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

# Residual blocks
x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)
```

### EfficientNet

Best accuracy/efficiency trade-off.

```python
from tensorflow.keras.applications import EfficientNetB0

# Use pre-trained EfficientNet
base_model = EfficientNetB0(
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
```

### MobileNet

Lightweight for mobile/edge devices.

```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    alpha=1.0  # Width multiplier (0.25, 0.5, 0.75, 1.0)
)
```

---

## Advanced Data Augmentation

### Cutout and Mixup

```python
def cutout(image, num_holes=1, hole_size=16):
    """Randomly cut out square regions"""
    h, w = image.shape[:2]
    for _ in range(num_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - hole_size // 2, 0, h)
        y2 = np.clip(y + hole_size // 2, 0, h)
        x1 = np.clip(x - hole_size // 2, 0, w)
        x2 = np.clip(x + hole_size // 2, 0, w)
        image[y1:y2, x1:x2] = 0
    return image

def mixup(x1, y1, x2, y2, alpha=0.2):
    """Mix two images and labels"""
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed
```

---

## Object Detection and Segmentation

### YOLO Implementation

```python
try:
    from ultralytics import YOLO
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Train on custom dataset
    model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16
    )
    
    # Detect objects
    results = model('image.jpg')
    
    # Export to different formats
    model.export(format='onnx')
    model.export(format='tensorflow')
    
except ImportError:
    print("Install: pip install ultralytics")
```

### Semantic Segmentation

```python
from tensorflow.keras.applications import VGG16

def create_segmentation_model(input_shape=(512, 512, 3), num_classes=21):
    """U-Net like segmentation model"""
    # Encoder (VGG16)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Decoder
    x = base_model.output
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = keras.Model(base_model.input, x)
    return model
```

---

## Image Generation

### GAN Basics

```python
def build_generator(latent_dim=100):
    """Generator network"""
    model = keras.Sequential([
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(3, 7, padding='same', activation='tanh')
    ])
    return model

def build_discriminator():
    """Discriminator network"""
    model = keras.Sequential([
        layers.Conv2D(64, 3, strides=2, padding='same', input_shape=(28, 28, 3)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

---

## Model Optimization

### Model Quantization

```python
import tensorflow_model_optimization as tfmot

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Quantization-aware training
qat_model = tfmot.quantization.keras.quantize_model(model)
```

### Model Pruning

```python
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting

**Solution**: Use data augmentation, dropout, early stopping

### Pitfall 2: Slow Training

**Solution**: Use GPU, reduce image size, use efficient architectures

### Pitfall 3: Poor Transfer Learning

**Solution**: Use appropriate pre-trained model, fine-tune properly

---

## Key Takeaways

1. **Advanced Architectures**: ResNet, EfficientNet for better performance
2. **Advanced Augmentation**: Cutout, Mixup for robustness
3. **Object Detection**: YOLO for fast detection
4. **Segmentation**: U-Net for pixel-level classification
5. **Optimization**: Quantization and pruning for deployment

---

**Remember**: Advanced techniques build on fundamentals - master basics first!

