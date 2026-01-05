# Advanced Deep Learning Frameworks Topics

Comprehensive guide to advanced techniques and best practices for TensorFlow/Keras and PyTorch.

## Table of Contents

- [Custom Layers and Models](#custom-layers-and-models)
- [Advanced Training Techniques](#advanced-training-techniques)
- [Model Optimization](#model-optimization)
- [Distributed Training](#distributed-training)
- [Model Deployment](#model-deployment)
- [Debugging and Profiling](#debugging-and-profiling)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Custom Layers and Models

### Keras Custom Layers

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config

# Use custom layer
model = keras.Sequential([
    DenseLayer(128, activation='relu', input_shape=(784,)),
    DenseLayer(64, activation='relu'),
    DenseLayer(10, activation='softmax')
])
```

### PyTorch Custom Layers

```python
import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

# Use in model
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomLinear(784, 128)
        self.layer2 = CustomLinear(128, 64)
        self.layer3 = CustomLinear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

---

## Advanced Training Techniques

### Mixed Precision Training

**Keras:**
```python
# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Build model (automatically uses float16)
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32')  # Output in float32
])
```

**PyTorch:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Gradient Accumulation

**Keras:**
```python
class GradientAccumulationModel(keras.Model):
    def __init__(self, accumulation_steps=4):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        
        if self.accumulation_counter == 0:
            self.accumulated_gradients = [tf.zeros_like(g) for g in gradients]
        
        for i, grad in enumerate(gradients):
            self.accumulated_gradients[i] += grad / self.accumulation_steps
        
        self.accumulation_counter += 1
        
        if self.accumulation_counter == self.accumulation_steps:
            self.optimizer.apply_gradients(
                zip(self.accumulated_gradients, self.trainable_variables)
            )
            self.accumulation_counter = 0
        
        return {'loss': loss}
```

**PyTorch:**
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Learning Rate Scheduling

**Keras:**
```python
# Cosine annealing
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000
)

# One-cycle policy
def one_cycle_lr(epoch, max_epochs):
    if epoch < max_epochs // 2:
        return 0.001 * (2 * epoch / max_epochs)
    else:
        return 0.001 * (2 - 2 * epoch / max_epochs)

lr_callback = keras.callbacks.LearningRateScheduler(one_cycle_lr)
```

**PyTorch:**
```python
# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# One-cycle
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # ... training code ...
        scheduler.step()  # Update learning rate
```

---

## Model Optimization

### Model Pruning

**Keras:**
```python
import tensorflow_model_optimization as tfmot

# Prune model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Train with pruning
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

# Strip pruning wrapper
model = tfmot.sparsity.keras.strip_pruning(model)
```

**PyTorch:**
```python
import torch.nn.utils.prune as prune

# Prune weights
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
        prune.remove(module, 'weight')  # Make permanent

# Or use structured pruning
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
```

### Quantization

**Keras:**
```python
import tensorflow_model_optimization as tfmot

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Quantization-aware training
qat_model = tfmot.quantization.keras.quantize_model(model)
qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
qat_model.fit(x_train, y_train, epochs=10)
```

**PyTorch:**
```python
# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Quantization-aware training
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Train model
torch.quantization.convert(model, inplace=True)
```

---

## Distributed Training

### Keras Multi-GPU

```python
# Strategy for multi-GPU
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training automatically uses all GPUs
model.fit(x_train, y_train, batch_size=128 * strategy.num_replicas_in_sync)
```

### PyTorch Distributed

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Training loop (same as before, but runs on multiple GPUs)
# Each process handles a portion of data
```

---

## Model Deployment

### TensorFlow Serving

```python
# Save model in SavedModel format
model.save('saved_model', save_format='tf')

# Serve with TensorFlow Serving
# docker run -p 8501:8501 --mount type=bind,source=/path/to/saved_model,target=/models/model -e MODEL_NAME=model -t tensorflow/serving

# Client code
import requests
import json

data = {'instances': x_test[:10].tolist()}
response = requests.post('http://localhost:8501/v1/models/model:predict', json=data)
predictions = json.loads(response.text)['predictions']
```

### PyTorch TorchScript

```python
# Trace model
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model.pt')

# Script model (for control flow)
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# Load and use
loaded_model = torch.jit.load('model.pt')
loaded_model.eval()
with torch.no_grad():
    output = loaded_model(example_input)
```

### ONNX Export

**Keras:**
```python
import tf2onnx
import onnx

onnx_model, _ = tf2onnx.convert.from_keras(model, output_path='model.onnx')
```

**PyTorch:**
```python
dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

---

## Debugging and Profiling

### Keras Debugging

```python
# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Check gradients
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check for NaN or Inf
    for grad in gradients:
        if tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad)):
            print("NaN or Inf detected in gradients!")
            break
    
    return loss, gradients

# Profile training
tf.profiler.experimental.start('logdir')
model.fit(x_train, y_train, epochs=1)
tf.profiler.experimental.stop()
```

### PyTorch Debugging

```python
# Check for NaN/Inf
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN/Inf in {name}")

# Use in training
for batch_x, batch_y in train_loader:
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    check_gradients(model)
    optimizer.step()
    optimizer.zero_grad()

# Profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()

print(prof.key_averages().table(sort_by='cuda_time_total'))
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting model.eval() in PyTorch

**Problem**: Dropout and BatchNorm behave differently in train/eval mode

**Solution**:
```python
# Always set eval mode for inference
model.eval()
with torch.no_grad():
    predictions = model(x_test)
```

### Pitfall 2: Memory Issues

**Problem**: Out of memory errors

**Solution**:
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Use gradient checkpointing
# Keras: Use model with gradient_checkpointing=True
# PyTorch: Use torch.utils.checkpoint

# Clear cache
torch.cuda.empty_cache()
```

### Pitfall 3: Data Type Mismatches

**Problem**: Type errors between tensors

**Solution**:
```python
# Keras: Ensure consistent dtypes
x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.int32)

# PyTorch: Use .to() method
x = x.to(torch.float32)
y = y.to(torch.long)
```

### Pitfall 4: Not Shuffling Data

**Problem**: Model sees data in same order

**Solution**:
```python
# Keras: shuffle=True in fit()
model.fit(x_train, y_train, shuffle=True)

# PyTorch: shuffle=True in DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Pitfall 5: Learning Rate Too High

**Problem**: Loss doesn't decrease or diverges

**Solution**:
```python
# Start with lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Instead of 0.001

# Use learning rate finder
# Or use ReduceLROnPlateau callback
```

---

## Key Takeaways

1. **Custom Components**: Create custom layers/models for specific needs
2. **Advanced Training**: Mixed precision, gradient accumulation, advanced scheduling
3. **Optimization**: Pruning and quantization for deployment
4. **Distributed Training**: Scale to multiple GPUs/machines
5. **Deployment**: TensorFlow Serving, TorchScript, ONNX for production
6. **Debugging**: Use profiling and gradient checking
7. **Common Issues**: Memory, data types, learning rates

---

## Next Steps

- Practice advanced techniques
- Experiment with custom architectures
- Learn about model optimization
- Explore deployment options
- Move to computer vision module

**Remember**: Advanced techniques require understanding fundamentals first!

