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

### Introduction to Object Detection: From R-CNN to YOLO

**Object Detection** involves both locating objects (bounding boxes) and classifying them.

**Evolution of Object Detection Frameworks:**

1. **R-CNN (2014)**: Region-based CNN
2. **Fast R-CNN (2015)**: Faster training and inference
3. **Faster R-CNN (2015)**: End-to-end training with Region Proposal Network
4. **YOLO (2016)**: You Only Look Once - real-time detection

### R-CNN Family

**R-CNN (Region-based CNN):**
- Uses selective search to generate region proposals
- Each region is classified separately
- Slow but accurate

**Fast R-CNN:**
- Processes entire image once
- ROI pooling for region features
- Faster than R-CNN

**Faster R-CNN:**
- Adds Region Proposal Network (RPN)
- End-to-end trainable
- Good accuracy, moderate speed

```python
# Conceptual R-CNN approach
def rcnn_pipeline(image):
    """
    R-CNN pipeline (conceptual)
    1. Generate region proposals (selective search)
    2. Extract features for each region
    3. Classify each region
    4. Refine bounding boxes
    """
    # Step 1: Region proposals (simplified)
    regions = selective_search(image)  # Returns list of bounding boxes
    
    # Step 2-3: Classify each region
    detections = []
    for region in regions:
        features = extract_features(region)  # CNN feature extraction
        class_scores = classifier(features)  # Classification
        detections.append((region, class_scores))
    
    # Step 4: Non-maximum suppression
    final_detections = nms(detections)
    return final_detections
```

### YOLO: You Only Look Once

**YOLO Philosophy:**
- Single forward pass through network
- Predicts bounding boxes and classes simultaneously
- Much faster than R-CNN family

**YOLO Architecture:**
- Divides image into grid (e.g., 7x7 or 13x13)
- Each grid cell predicts:
  - Bounding box coordinates
  - Object confidence
  - Class probabilities

**YOLO Versions:**
- **YOLOv1 (2016)**: Original YOLO
- **YOLOv2/YOLO9000 (2017)**: Better accuracy
- **YOLOv3 (2018)**: Multi-scale detection
- **YOLOv4 (2020)**: Improved performance
- **YOLOv5 (2020)**: PyTorch implementation
- **YOLOv8 (2023)**: Latest version

```python
try:
    from ultralytics import YOLO
    import cv2
    
    # Load pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # nano version (fastest)
    # Other options: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    
    # Detect objects in image
    results = model('image.jpg')
    
    # Process results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Get confidence
            confidence = box.conf[0].cpu().numpy()
            # Get class
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            print(f"{class_name}: {confidence:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    
    # Visualize results
    annotated_image = results[0].plot()
    cv2.imwrite('detected.jpg', annotated_image)
    
    # Train on custom dataset
    model.train(
        data='dataset.yaml',  # Dataset configuration
        epochs=100,
        imgsz=640,
        batch=16,
        device=0  # GPU device
    )
    
    # Export to different formats
    model.export(format='onnx')  # ONNX format
    model.export(format='tensorflow')  # TensorFlow format
    model.export(format='torchscript')  # TorchScript format
    
except ImportError:
    print("Install: pip install ultralytics")
```

**Comparative Study: R-CNN vs. YOLO**

| Aspect | R-CNN Family | YOLO |
|--------|--------------|------|
| **Speed** | Slow (seconds per image) | Fast (real-time) |
| **Accuracy** | High | Good (slightly lower) |
| **Training** | Multi-stage | End-to-end |
| **Use Case** | High accuracy needed | Real-time applications |
| **Small Objects** | Better | Can struggle |

**Implementing YOLO for Real-time Object Detection:**

```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Real-time detection from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Semantic and Instance Segmentation: Techniques and Applications

**Segmentation** is pixel-level classification, more detailed than object detection.

**Types of Segmentation:**

1. **Semantic Segmentation**: Classify each pixel (no object distinction)
2. **Instance Segmentation**: Identify and segment individual objects
3. **Panoptic Segmentation**: Combines semantic + instance

### Semantic Segmentation

**Goal**: Assign class label to each pixel.

**Key Methods:**
- **FCN (Fully Convolutional Networks)**: First CNN for segmentation
- **U-Net**: Encoder-decoder with skip connections
- **DeepLab**: Atrous convolutions for better context
- **PSPNet**: Pyramid Scene Parsing Network

**U-Net Architecture:**

```python
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

def create_unet(input_shape=(512, 512, 3), num_classes=21):
    """
    U-Net architecture for semantic segmentation
    Encoder-Decoder with skip connections
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (downsampling path)
    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)
    
    # Block 4 (bottleneck)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(2)(conv4)
    
    # Bottom
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder (upsampling path) with skip connections
    # Block 4
    up4 = layers.UpSampling2D(2)(conv5)
    up4 = layers.Conv2D(512, 2, activation='relu', padding='same')(up4)
    merge4 = layers.Concatenate()([conv4, up4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge4)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 3
    up3 = layers.UpSampling2D(2)(conv6)
    up3 = layers.Conv2D(256, 2, activation='relu', padding='same')(up3)
    merge3 = layers.Concatenate()([conv3, up3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 2
    up2 = layers.UpSampling2D(2)(conv7)
    up2 = layers.Conv2D(128, 2, activation='relu', padding='same')(up2)
    merge2 = layers.Concatenate()([conv2, up2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 1
    up1 = layers.UpSampling2D(2)(conv8)
    up1 = layers.Conv2D(64, 2, activation='relu', padding='same')(up1)
    merge1 = layers.Concatenate()([conv1, up1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge1)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs, outputs, name='U-Net')
    return model

# Create model
unet_model = create_unet()
unet_model.summary()
```

### Instance Segmentation

**Goal**: Identify and segment each object instance separately.

**Key Methods:**
- **Mask R-CNN**: Extends Faster R-CNN with segmentation branch
- **YOLACT**: Real-time instance segmentation
- **SOLO**: Segmenting Objects by Locations

**Mask R-CNN:**

```python
# Using detectron2 (Facebook AI Research)
try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    import cv2
    
    # Setup configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    image = cv2.imread('image.jpg')
    outputs = predictor(image)
    
    # Access results
    instances = outputs["instances"]
    boxes = instances.pred_boxes
    scores = instances.scores
    classes = instances.pred_classes
    masks = instances.pred_masks  # Segmentation masks
    
    print(f"Detected {len(instances)} objects")
    
except ImportError:
    print("Install: pip install detectron2")
    print("Or use: pip install torch torchvision")
```

**Distinction Between Semantic and Instance Segmentation:**

```python
# Conceptual difference
def visualize_segmentation_types():
    """
    Semantic Segmentation: All cars are same class (no distinction)
    Instance Segmentation: Each car is separate instance (numbered)
    """
    # Semantic: [car, car, car, person, person]
    # Instance: [car_1, car_2, car_3, person_1, person_2]
    pass
```

### Application Cases

**Autonomous Vehicles:**
- Lane detection (semantic)
- Object detection (cars, pedestrians)
- Instance segmentation for tracking

**Medical Imaging:**
- Organ segmentation
- Tumor detection
- Cell counting

**Example: Medical Image Segmentation:**

```python
def medical_segmentation_example():
    """
    Example: Segmenting organs in CT scans
    """
    # Load medical image
    ct_scan = load_ct_scan('patient_scan.dcm')
    
    # Preprocess
    processed = preprocess_medical_image(ct_scan)
    
    # Segment using U-Net
    segmentation_mask = unet_model.predict(processed)
    
    # Post-process
    organs = extract_organs(segmentation_mask)
    
    return organs
```

---

## Image Generation

### Advanced Topics: GANs for Image Synthesis and Editing

**Generative Adversarial Networks (GANs)** consist of two networks competing:
- **Generator**: Creates fake images
- **Discriminator**: Distinguishes real from fake

**Training Process:**
1. Generator creates fake images
2. Discriminator tries to classify real vs fake
3. Both networks improve through adversarial training
4. Eventually, generator creates realistic images

### Introduction to Generative Adversarial Networks (GANs)

**Key Components:**

```python
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np

def build_generator(latent_dim=100, output_shape=(28, 28, 1)):
    """
    Generator network: Maps random noise to images
    """
    inputs = layers.Input(shape=(latent_dim,))
    
    # Project and reshape
    x = layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((7, 7, 256))(x)
    
    # Upsample to 14x14
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Upsample to 28x28
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Output layer
    outputs = layers.Conv2D(output_shape[2], 7, padding='same', activation='tanh')(x)
    
    model = Model(inputs, outputs, name='Generator')
    return model

def build_discriminator(input_shape=(28, 28, 1)):
    """
    Discriminator network: Classifies real vs fake images
    """
    inputs = layers.Input(shape=input_shape)
    
    # Downsample
    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Classify
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='Discriminator')
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()

print("Generator:")
generator.summary()
print("\nDiscriminator:")
discriminator.summary()
```

### Use Cases in Image Generation and Photo Editing

**1. Image Generation:**
- Create new images from scratch
- Style transfer
- Data augmentation

**2. Photo Editing:**
- Super-resolution (enhance image quality)
- Inpainting (fill missing regions)
- Style transfer
- Face aging/younging

**3. Data Augmentation:**
- Generate synthetic training data
- Balance imbalanced datasets

### Hands-on GAN Training Session

```python
class GAN(Model):
    """
    Combined GAN model
    """
    def __init__(self, generator, discriminator, latent_dim=100):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train Discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors, training=True)
        
        # Combine real and fake
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        # Add noise to labels (label smoothing)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        
        # Train Generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            predictions = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Create and compile GAN
gan = GAN(generator, discriminator)
gan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Training (conceptual)
# gan.fit(dataset, epochs=100, callbacks=[...])
```

**GAN Variants:**

1. **DCGAN**: Deep Convolutional GAN
2. **WGAN**: Wasserstein GAN (more stable training)
3. **StyleGAN**: High-quality face generation
4. **CycleGAN**: Unpaired image-to-image translation
5. **Pix2Pix**: Paired image-to-image translation

**Best Practices for GAN Training:**

```python
# Tips for stable GAN training
gan_training_tips = {
    "Generator": [
        "Use BatchNormalization (except output layer)",
        "Use LeakyReLU instead of ReLU",
        "Use tanh activation for output (scale inputs to [-1, 1])",
        "Use Adam optimizer with beta1=0.5"
    ],
    "Discriminator": [
        "Use dropout for regularization",
        "Use LeakyReLU",
        "Label smoothing (0.9 instead of 1.0)",
        "Train discriminator more than generator initially"
    ],
    "Training": [
        "Monitor both losses (should be balanced)",
        "Use learning rate scheduling",
        "Save generated images periodically",
        "Use gradient penalty (for WGAN)"
    ]
}
```

---

## Special Topics: Recent Breakthroughs and Research Directions

### Latest Innovations and Research in Computer Vision

**2020s Breakthroughs:**

1. **Vision Transformers (ViT, 2020)**
   - Apply Transformer architecture to images
   - Divide image into patches, treat as sequence
   - Competitive with CNNs on large datasets

2. **CLIP (2021)**
   - Contrastive Language-Image Pre-training
   - Learns from image-text pairs
   - Zero-shot image classification

3. **DALL-E 2 / Stable Diffusion (2022)**
   - Text-to-image generation
   - High-quality, controllable image synthesis

4. **Segment Anything Model (SAM, 2023)**
   - Foundation model for segmentation
   - Zero-shot segmentation

### Vision Transformers (ViT)

```python
# Conceptual Vision Transformer
def vision_transformer_concept():
    """
    Vision Transformer approach:
    1. Split image into patches (e.g., 16x16)
    2. Flatten patches and project to embeddings
    3. Add position embeddings
    4. Process with Transformer encoder
    5. Classify using [CLS] token
    """
    pass

# Using pre-trained ViT
try:
    from transformers import ViTModel, ViTImageProcessor
    from PIL import Image
    
    # Load pre-trained ViT
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Process image
    image = Image.open('image.jpg')
    inputs = processor(image, return_tensors='pt')
    
    # Get features
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    
except ImportError:
    print("Install: pip install transformers torch")
```

### Emerging Technologies and Their Potential Impacts

**1. Foundation Models:**
- Large pre-trained models (GPT for vision)
- Transfer to many tasks with minimal fine-tuning
- Examples: CLIP, SAM, Segment Anything

**2. Multimodal Learning:**
- Combine vision, language, audio
- Examples: CLIP, DALL-E, GPT-4V

**3. Efficient Models:**
- Mobile/edge deployment
- Examples: MobileNet, EfficientNet, Vision Transformer variants

**4. Self-Supervised Learning:**
- Learn from unlabeled data
- Examples: SimCLR, MoCo, BYOL

### Future Directions and Career Opportunities

**Research Directions:**
- **3D Vision**: 3D object detection, reconstruction
- **Video Understanding**: Action recognition, video generation
- **Robotics**: Vision for robot control
- **Medical AI**: Diagnostic imaging, surgery assistance
- **AR/VR**: Real-time scene understanding

**Career Opportunities:**
- **Computer Vision Engineer**: Build CV systems
- **Research Scientist**: Advance CV algorithms
- **ML Engineer**: Deploy CV models
- **Robotics Engineer**: Vision for robots
- **Medical AI Specialist**: Healthcare applications

**Skills Needed:**
- Deep learning frameworks (TensorFlow, PyTorch)
- Computer vision libraries (OpenCV, PIL)
- Mathematics (linear algebra, calculus)
- Software engineering
- Domain knowledge (for specific applications)

---

## Project: Building Blocks of a Self-Driving Car - Vision-Based Navigation Systems

### Design and Implementation of Computer Vision Systems for Self-Driving Cars

**Key Components:**

1. **Object Detection**: Detect cars, pedestrians, cyclists
2. **Lane Detection**: Identify and track lanes
3. **Traffic Sign Recognition**: Read signs and signals
4. **Semantic Segmentation**: Understand road scene
5. **Depth Estimation**: Measure distances

### Real-time Object Detection

```python
# Using YOLO for real-time object detection in autonomous vehicles
from ultralytics import YOLO
import cv2

class AutonomousVehicleVision:
    def __init__(self):
        self.object_detector = YOLO('yolov8n.pt')
        # Filter for relevant classes
        self.relevant_classes = {
            0: 'person',  # Pedestrian
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            11: 'stop sign'
        }
    
    def detect_objects(self, frame):
        """Detect relevant objects in frame"""
        results = self.object_detector(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in self.relevant_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    detections.append({
                        'class': self.relevant_classes[class_id],
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
        
        return detections
```

### Lane Tracking

```python
import cv2
import numpy as np

def detect_lanes(image):
    """
    Lane detection using computer vision
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Region of interest (road area)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough line transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=50)
    
    return lines

def draw_lanes(image, lines):
    """Draw detected lanes on image"""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image
```

### Traffic Sign Recognition

```python
# Traffic sign recognition using CNN
def create_traffic_sign_classifier():
    """CNN for traffic sign classification"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(43, activation='softmax')  # 43 traffic sign classes
    ])
    return model
```

### Final Presentation: Simulated Autonomous Driving Scenario

```python
class AutonomousDrivingSystem:
    """
    Complete vision system for autonomous vehicle
    """
    def __init__(self):
        self.object_detector = YOLO('yolov8n.pt')
        self.lane_detector = detect_lanes
        self.sign_classifier = create_traffic_sign_classifier()
    
    def process_frame(self, frame):
        """Process single frame for autonomous driving"""
        # 1. Object detection
        objects = self.detect_objects(frame)
        
        # 2. Lane detection
        lanes = self.detect_lanes(frame)
        frame_with_lanes = self.draw_lanes(frame.copy(), lanes)
        
        # 3. Traffic sign recognition (if signs detected)
        signs = self.detect_traffic_signs(frame)
        
        # 4. Decision making
        decision = self.make_decision(objects, lanes, signs)
        
        return {
            'frame': frame_with_lanes,
            'objects': objects,
            'lanes': lanes,
            'signs': signs,
            'decision': decision
        }
    
    def make_decision(self, objects, lanes, signs):
        """Make driving decision based on vision inputs"""
        # Simple decision logic
        if 'stop sign' in [s['class'] for s in signs]:
            return 'STOP'
        elif 'person' in [o['class'] for o in objects]:
            return 'SLOW_DOWN'
        elif lanes is None:
            return 'CAUTION'
        else:
            return 'CONTINUE'
```

**Key Takeaways:**
- Computer vision is crucial for autonomous vehicles
- Multiple CV tasks work together (detection, segmentation, recognition)
- Real-time performance is essential
- Safety and reliability are paramount

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

