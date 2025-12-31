# Neural Networks Quick Reference Guide

Quick reference for neural network concepts, code snippets, and best practices.

## Table of Contents

- [Neural Network Basics](#neural-network-basics)
- [Code Snippets](#code-snippets)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Optimization](#optimization)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Neural Network Basics

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
     ↓              ↓                ↓
  Features    Feature Learning   Predictions
```

### Key Components

- **Neurons**: Processing units
- **Weights**: Learned parameters
- **Biases**: Offset parameters
- **Activation Functions**: Non-linearity
- **Loss Function**: Measures error
- **Optimizer**: Updates weights

---

## Code Snippets

### Basic Perceptron

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                
                if y_predicted != y[idx]:
                    update = self.learning_rate * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [1 if x >= 0 else 0 for x in linear_output]
```

### Basic MLP

```python
class MLP:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backward(self, activations, y):
        m = y.shape[0]
        gradients_w = []
        gradients_b = []
        
        error = activations[-1] - y
        delta = error * activations[-1] * (1 - activations[-1])
        
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * activations[i] * (1 - activations[i])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations = self.forward(X)
            grad_w, grad_b = self.backward(activations, y)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_w[i]
                self.biases[i] -= self.learning_rate * grad_b[i]
    
    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]
```

---

## Activation Functions

### Common Activations

```python
# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

# Tanh
def tanh(x):
    return np.tanh(x)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# Softmax (for multi-class)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Derivatives

```python
# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Tanh derivative
def tanh_derivative(x):
    return 1 - x ** 2

# ReLU derivative
def relu_derivative(x):
    return (x > 0).astype(float)
```

### When to Use

| Function | Range | Use Case | Code |
|----------|-------|----------|------|
| **Sigmoid** | (0, 1) | Output layer (binary) | `sigmoid(x)` |
| **Tanh** | (-1, 1) | Hidden layers | `tanh(x)` |
| **ReLU** | [0, ∞) | Hidden layers (most common) | `relu(x)` |
| **Leaky ReLU** | (-∞, ∞) | Alternative to ReLU | `leaky_relu(x)` |
| **Softmax** | (0, 1), sums to 1 | Output layer (multi-class) | `softmax(x)` |

---

## Loss Functions

### Regression Losses

```python
# Mean Squared Error
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### Classification Losses

```python
# Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy
def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

### Loss Function Selection

| Task | Loss Function | Code |
|------|---------------|------|
| **Regression** | MSE | `mse_loss(y_true, y_pred)` |
| **Binary Classification** | Binary CE | `binary_crossentropy(y_true, y_pred)` |
| **Multi-class Classification** | Categorical CE | `categorical_crossentropy(y_true, y_pred)` |

---

## Optimization

### Gradient Descent Variants

```python
# Batch Gradient Descent
def batch_gd(X, y, weights, learning_rate=0.01):
    predictions = X @ weights
    error = predictions - y
    gradient = X.T @ error / len(X)
    weights -= learning_rate * gradient
    return weights

# Stochastic Gradient Descent
def sgd(X, y, weights, learning_rate=0.01):
    for i in range(len(X)):
        prediction = X[i] @ weights
        error = prediction - y[i]
        gradient = X[i] * error
        weights -= learning_rate * gradient
    return weights

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, weights, batch_size=32, learning_rate=0.01):
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        predictions = batch_X @ weights
        error = predictions - batch_y
        gradient = batch_X.T @ error / len(batch_X)
        weights -= learning_rate * gradient
    return weights
```

### Weight Initialization

```python
# Xavier/Glorot (for tanh/sigmoid)
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# He (for ReLU)
def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

# Random (simple)
def random_init(fan_in, fan_out, scale=0.1):
    return np.random.randn(fan_in, fan_out) * scale
```

---

## Common Issues & Solutions

### Issue 1: Vanishing Gradients

**Problem**: Gradients become very small

**Solution**:
```python
# Use ReLU instead of sigmoid/tanh
activation = 'relu'

# Use proper initialization
weights = he_init(fan_in, fan_out)  # For ReLU
```

### Issue 2: Exploding Gradients

**Problem**: Gradients become very large

**Solution**:
```python
# Gradient clipping
max_grad = 1.0
gradient = np.clip(gradient, -max_grad, max_grad)

# Lower learning rate
learning_rate = 0.001  # Instead of 0.01
```

### Issue 3: Overfitting

**Problem**: Model memorizes training data

**Solution**:
```python
# Add L2 regularization
lambda_reg = 0.01
loss = mse_loss + lambda_reg * np.sum(weights ** 2)

# Use dropout
def dropout(X, dropout_rate=0.5):
    mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape)
    return X * mask / (1 - dropout_rate)
```

### Issue 4: Learning Rate Too High

**Problem**: Loss diverges or oscillates

**Solution**:
```python
# Lower learning rate
learning_rate = 0.001  # Start small

# Use learning rate scheduling
def get_lr(epoch, initial_lr=0.01, decay=0.95):
    return initial_lr * (decay ** epoch)
```

### Issue 5: Poor Weight Initialization

**Problem**: Network doesn't learn

**Solution**:
```python
# Use proper initialization
if activation == 'relu':
    weights = he_init(fan_in, fan_out)
else:
    weights = xavier_init(fan_in, fan_out)

# Don't initialize to zero
# Don't initialize too large
```

---

## Best Practices Checklist

### Data Preparation
- [ ] Scale/normalize features
- [ ] Handle missing values
- [ ] Split into train/validation/test
- [ ] Shuffle data

### Architecture
- [ ] Start with simple architecture
- [ ] Choose appropriate activation functions
- [ ] Initialize weights properly
- [ ] Use appropriate number of layers/neurons

### Training
- [ ] Use mini-batch gradient descent
- [ ] Set appropriate learning rate
- [ ] Monitor training and validation loss
- [ ] Use early stopping
- [ ] Save best model

### Regularization
- [ ] Add L1/L2 regularization if overfitting
- [ ] Use dropout for deep networks
- [ ] Consider batch normalization

### Debugging
- [ ] Check gradients (not vanishing/exploding)
- [ ] Visualize loss curves
- [ ] Verify forward/backward pass
- [ ] Test on simple data first

### Evaluation
- [ ] Evaluate on test set
- [ ] Use appropriate metrics
- [ ] Visualize predictions
- [ ] Compare with baseline

---

## Quick Code Templates

### Complete Training Loop

```python
# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Initialize network
nn = MLP(layers=[input_dim, hidden_dim, output_dim], learning_rate=0.01)

# 3. Train
for epoch in range(epochs):
    # Forward
    activations = nn.forward(X_train)
    
    # Compute loss
    loss = compute_loss(y_train, activations[-1])
    
    # Backward
    grad_w, grad_b = nn.backward(activations, y_train)
    
    # Update
    for i in range(len(nn.weights)):
        nn.weights[i] -= nn.learning_rate * grad_w[i]
        nn.biases[i] -= nn.learning_rate * grad_b[i]
    
    # Evaluate
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 4. Predict
predictions = nn.predict(X_test)
```

### Forward Pass Template

```python
def forward(X, weights, biases, activation='sigmoid'):
    activations = [X]
    for i in range(len(weights)):
        z = activations[-1] @ weights[i] + biases[i]
        if activation == 'relu':
            a = np.maximum(0, z)
        else:
            a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        activations.append(a)
    return activations
```

### Backward Pass Template

```python
def backward(activations, y, weights, activation='sigmoid'):
    m = y.shape[0]
    gradients_w = []
    gradients_b = []
    
    error = activations[-1] - y
    if activation == 'relu':
        delta = error * (activations[-1] > 0).astype(float)
    else:
        delta = error * activations[-1] * (1 - activations[-1])
    
    for i in range(len(weights) - 1, -1, -1):
        grad_w = activations[i].T @ delta / m
        grad_b = np.sum(delta, axis=0, keepdims=True) / m
        gradients_w.insert(0, grad_w)
        gradients_b.insert(0, grad_b)
        
        if i > 0:
            delta = delta @ weights[i].T
            if activation == 'relu':
                delta = delta * (activations[i] > 0).astype(float)
            else:
                delta = delta * activations[i] * (1 - activations[i])
    
    return gradients_w, gradients_b
```

---

## Key Takeaways

1. **Always scale features** before training
2. **Use proper weight initialization** (He for ReLU, Xavier for tanh/sigmoid)
3. **Choose activation functions** based on layer (ReLU for hidden, sigmoid/softmax for output)
4. **Use appropriate loss function** (MSE for regression, Cross-Entropy for classification)
5. **Monitor training** - check loss and accuracy curves
6. **Regularize** if overfitting (L1/L2, dropout)
7. **Debug systematically** - check gradients, verify forward/backward pass

---

## Next Steps

- Practice building networks from scratch
- Experiment with different architectures
- Learn about advanced optimizers (Adam, RMSprop)
- Move to deep learning frameworks module

**Remember**: Understanding fundamentals is crucial before using frameworks!

