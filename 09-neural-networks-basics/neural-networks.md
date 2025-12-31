# Neural Networks Basics Complete Guide

Comprehensive guide to understanding and building neural networks from scratch.

## Table of Contents

- [Introduction](#introduction)
- [Perceptron](#perceptron)
- [Multi-Layer Perceptron](#multi-layer-perceptron)
- [Activation Functions](#activation-functions)
- [Backpropagation](#backpropagation)
- [Gradient Descent](#gradient-descent)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What are Neural Networks?

Neural networks are computing systems inspired by biological neural networks. They learn patterns from data through interconnected nodes (neurons) that process information.

**Key Components:**
- **Neurons**: Basic processing units that receive inputs, apply weights, and produce outputs
- **Layers**: Groups of neurons that process information at different levels
- **Weights**: Learned parameters that determine the strength of connections
- **Biases**: Offset parameters that shift the activation function
- **Activation Functions**: Non-linear functions that introduce complexity

### Why Neural Networks?

**Advantages:**
- Can learn complex non-linear patterns
- Universal function approximators
- Automatic feature learning
- Handle high-dimensional data
- State-of-the-art performance in many tasks

**Applications:**
- Image classification
- Natural language processing
- Speech recognition
- Recommendation systems
- Game playing (AlphaGo, etc.)

### Neural Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
     ↓              ↓                ↓
  Features    Feature Learning   Predictions
```

**Basic Structure:**
1. **Input Layer**: Receives raw data
2. **Hidden Layers**: Process and transform data (can have multiple)
3. **Output Layer**: Produces final predictions

---

## Perceptron

### Single Perceptron

Simplest neural network - single neuron that performs binary classification.

**How it works:**
1. Compute weighted sum: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
2. Apply activation function: `y = f(z)`
3. Update weights based on error

**Mathematical Formulation:**
- Input: `x = [x₁, x₂, ..., xₙ]`
- Weights: `w = [w₁, w₂, ..., wₙ]`
- Bias: `b`
- Output: `y = f(w·x + b)`

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors = []  # Track errors for visualization
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors = []
        
        for iteration in range(self.n_iterations):
            error_count = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Update weights if prediction is wrong
                if y_predicted != y[idx]:
                    update = self.learning_rate * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update
                    error_count += 1
            
            self.errors.append(error_count)
            
            # Early stopping if no errors
            if error_count == 0:
                print(f"Converged at iteration {iteration}")
                break
    
    def activation(self, x):
        """Step function (Heaviside)"""
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])
    
    def plot_decision_boundary(self, X, y):
        """Visualize decision boundary"""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        colors = ['red' if label == 0 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')
        
        # Plot decision boundary
        if self.weights[1] != 0:
            x_boundary = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
            y_boundary = -(self.weights[0] * x_boundary + self.bias) / self.weights[1]
            plt.plot(x_boundary, y_boundary, 'g--', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title('Perceptron Decision Boundary', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example: AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_and, y_and)
predictions = perceptron.predict(X_and)

print("AND Gate Results:")
for i, (x, y_true, y_pred) in enumerate(zip(X_and, y_and, predictions)):
    print(f"  Input: {x}, Target: {y_true}, Prediction: {y_pred}, {'✓' if y_true == y_pred else '✗'}")

# Visualize learning
plt.figure(figsize=(10, 4))
plt.plot(perceptron.errors, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Number of Errors', fontsize=12)
plt.title('Perceptron Learning Progress', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot decision boundary
perceptron.plot_decision_boundary(X_and, y_and)
```

### Perceptron Limitations

**XOR Problem**: Perceptron cannot solve XOR (exclusive OR) because it's not linearly separable.

```python
# XOR gate - cannot be solved by single perceptron
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(learning_rate=0.1, n_iterations=1000)
perceptron_xor.fit(X_xor, y_xor)
predictions_xor = perceptron_xor.predict(X_xor)

print("\nXOR Gate Results (Perceptron fails):")
for i, (x, y_true, y_pred) in enumerate(zip(X_xor, y_xor, predictions_xor)):
    print(f"  Input: {x}, Target: {y_true}, Prediction: {y_pred}, {'✓' if y_true == y_pred else '✗'}")

# Solution: Need multiple layers (MLP)
print("\nSolution: Use Multi-Layer Perceptron (MLP) with hidden layer")
```

**Key Insight**: This limitation led to the development of multi-layer perceptrons (MLPs) with hidden layers.

---

## Multi-Layer Perceptron

### Building MLP from Scratch

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
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
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
        
        # Output layer error
        error = activations[-1] - y
        delta = error * self.sigmoid_derivative(activations[-1])
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations = self.forward(X)
            grad_w, grad_b = self.backward(activations, y)
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_w[i]
                self.biases[i] -= self.learning_rate * grad_b[i]
            
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]
```

---

## Activation Functions

### Why Activation Functions?

Activation functions introduce non-linearity into neural networks. Without them, multiple layers would be equivalent to a single layer (linear transformation).

**Key Properties:**
- **Non-linearity**: Enables learning complex patterns
- **Differentiability**: Required for backpropagation
- **Bounded outputs**: Some functions bound outputs to specific ranges

### Common Activation Functions

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)

# Sigmoid: f(x) = 1 / (1 + e^(-x))
sigmoid = 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to avoid overflow
sigmoid_derivative = sigmoid * (1 - sigmoid)

# Tanh: f(x) = tanh(x)
tanh = np.tanh(x)
tanh_derivative = 1 - tanh ** 2

# ReLU: f(x) = max(0, x)
relu = np.maximum(0, x)
relu_derivative = (x > 0).astype(float)

# Leaky ReLU: f(x) = max(0.01x, x)
leaky_relu = np.maximum(0.01 * x, x)
leaky_relu_derivative = np.where(x > 0, 1, 0.01)

# ELU: Exponential Linear Unit
elu = np.where(x > 0, x, 0.1 * (np.exp(x) - 1))
elu_derivative = np.where(x > 0, 1, 0.1 * np.exp(x))

# Plot functions and derivatives
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

functions = [
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('ReLU', relu, relu_derivative),
    ('Leaky ReLU', leaky_relu, leaky_relu_derivative),
    ('ELU', elu, elu_derivative)
]

for idx, (name, func, deriv) in enumerate(functions):
    row = idx // 3
    col = idx % 3
    
    axes[row, col].plot(x, func, 'b-', linewidth=2, label='Function')
    axes[row, col].plot(x, deriv, 'r--', linewidth=2, label='Derivative')
    axes[row, col].set_title(name, fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('x', fontsize=11)
    axes[row, col].set_ylabel('f(x)', fontsize=11)
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[row, col].axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()
```

### Activation Function Comparison

| Function | Range | Pros | Cons | Use Case |
|----------|-------|------|------|----------|
| **Sigmoid** | (0, 1) | Smooth, bounded | Vanishing gradient, not zero-centered | Output layer (binary classification) |
| **Tanh** | (-1, 1) | Zero-centered, smooth | Vanishing gradient | Hidden layers (better than sigmoid) |
| **ReLU** | [0, ∞) | Fast, no vanishing gradient | Dying ReLU problem | Most common for hidden layers |
| **Leaky ReLU** | (-∞, ∞) | Fixes dying ReLU | Not smooth at 0 | Alternative to ReLU |
| **ELU** | (-∞, ∞) | Smooth, handles negatives | Computationally expensive | Alternative to ReLU |
| **Softmax** | (0, 1), sums to 1 | Probabilistic output | Only for output layer | Multi-class classification |

### When to Use Each

```python
# Sigmoid: Output layer for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

# Tanh: Hidden layers (better than sigmoid)
def tanh(x):
    return np.tanh(x)

# ReLU: Most common for hidden layers
def relu(x):
    return np.maximum(0, x)

# Softmax: Output layer for multiclass classification
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example: Multi-class classification output
logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
probabilities = softmax(logits)
print("Softmax Output (probabilities):")
print(probabilities)
print(f"Sum per sample: {probabilities.sum(axis=1)}")  # Should be 1.0
```

### Vanishing Gradient Problem

**Problem**: Sigmoid and tanh have small derivatives for large inputs, causing gradients to vanish in deep networks.

**Solution**: Use ReLU or its variants, which have constant gradient (1) for positive inputs.

```python
# Demonstrate vanishing gradient
x_large = np.array([10, 5, 2, 0, -2, -5, -10])

print("Gradient values for large inputs:")
print(f"Sigmoid derivative: {sigmoid(x_large) * (1 - sigmoid(x_large))}")
print(f"Tanh derivative: {1 - np.tanh(x_large)**2}")
print(f"ReLU derivative: {(x_large > 0).astype(float)}")
```

---

## Backpropagation

### Understanding Backpropagation

Backpropagation is the algorithm for training neural networks using the chain rule of calculus. It efficiently computes gradients by propagating errors backward through the network.

**Key Concept**: Chain Rule
- If `y = f(g(x))`, then `dy/dx = (dy/dg) * (dg/dx)`
- Allows us to compute gradients layer by layer

**Process:**
1. **Forward Pass**: Compute predictions and store intermediate values
2. **Calculate Loss**: Compare predictions with targets
3. **Backward Pass**: Compute gradients using chain rule
4. **Update Weights**: Adjust weights using gradients

### Step-by-Step Example

```python
def backpropagation_example():
    """Detailed backpropagation walkthrough"""
    # Network: x -> z1 -> a1 -> z2 -> y_pred
    # Parameters: w1, b1, w2, b2
    
    # Forward pass
    x = 2.0
    w1 = 0.5
    b1 = 0.1
    w2 = 0.3
    b2 = 0.2
    y_true = 1.0
    
    # Layer 1
    z1 = x * w1 + b1  # 2.0 * 0.5 + 0.1 = 1.1
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid(1.1) ≈ 0.75
    
    # Layer 2 (output)
    z2 = a1 * w2 + b2  # 0.75 * 0.3 + 0.2 = 0.425
    y_pred = 1 / (1 + np.exp(-z2))  # sigmoid(0.425) ≈ 0.605
    
    # Loss (MSE)
    loss = (y_pred - y_true) ** 2  # (0.605 - 1.0)² ≈ 0.156
    
    print("Forward Pass:")
    print(f"  z1 = {z1:.3f}, a1 = {a1:.3f}")
    print(f"  z2 = {z2:.3f}, y_pred = {y_pred:.3f}")
    print(f"  Loss = {loss:.3f}")
    
    # Backward pass (chain rule)
    # Gradient w.r.t. loss
    dloss_dypred = 2 * (y_pred - y_true)  # 2 * (0.605 - 1.0) = -0.79
    
    # Gradient w.r.t. z2
    dypred_dz2 = y_pred * (1 - y_pred)  # sigmoid derivative
    dloss_dz2 = dloss_dypred * dypred_dz2  # -0.79 * 0.239 ≈ -0.189
    
    # Gradient w.r.t. w2
    dz2_dw2 = a1
    dloss_dw2 = dloss_dz2 * dz2_dw2  # -0.189 * 0.75 ≈ -0.142
    
    # Gradient w.r.t. b2
    dz2_db2 = 1
    dloss_db2 = dloss_dz2 * dz2_db2  # -0.189
    
    # Gradient w.r.t. a1
    dloss_da1 = dloss_dz2 * w2  # -0.189 * 0.3 ≈ -0.057
    
    # Gradient w.r.t. z1
    da1_dz1 = a1 * (1 - a1)  # sigmoid derivative
    dloss_dz1 = dloss_da1 * da1_dz1  # -0.057 * 0.188 ≈ -0.011
    
    # Gradient w.r.t. w1
    dz1_dw1 = x
    dloss_dw1 = dloss_dz1 * dz1_dw1  # -0.011 * 2.0 ≈ -0.022
    
    # Gradient w.r.t. b1
    dz1_db1 = 1
    dloss_db1 = dloss_dz1 * dz1_db1  # -0.011
    
    print("\nBackward Pass (Gradients):")
    print(f"  dLoss/dw2 = {dloss_dw2:.3f}")
    print(f"  dLoss/db2 = {dloss_db2:.3f}")
    print(f"  dLoss/dw1 = {dloss_dw1:.3f}")
    print(f"  dLoss/db1 = {dloss_db1:.3f}")
    
    # Update weights (gradient descent)
    learning_rate = 0.1
    w1_new = w1 - learning_rate * dloss_dw1
    b1_new = b1 - learning_rate * dloss_db1
    w2_new = w2 - learning_rate * dloss_dw2
    b2_new = b2 - learning_rate * dloss_db2
    
    print("\nUpdated Weights:")
    print(f"  w1: {w1:.3f} -> {w1_new:.3f}")
    print(f"  b1: {b1:.3f} -> {b1_new:.3f}")
    print(f"  w2: {w2:.3f} -> {w2_new:.3f}")
    print(f"  b2: {b2:.3f} -> {b2_new:.3f}")

backpropagation_example()
```

### Backpropagation in Matrix Form

```python
def backpropagation_matrix_form():
    """Backpropagation using matrix operations (efficient)"""
    # Example: 2-layer network
    # Input: (batch_size, input_dim)
    # Weights: (input_dim, hidden_dim), (hidden_dim, output_dim)
    
    batch_size = 4
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    
    # Random data
    X = np.random.randn(batch_size, input_dim)
    y = np.random.randn(batch_size, output_dim)
    
    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))
    
    learning_rate = 0.01
    
    # Forward pass
    Z1 = X @ W1 + b1
    A1 = 1 / (1 + np.exp(-np.clip(Z1, -250, 250)))  # sigmoid
    Z2 = A1 @ W2 + b2
    A2 = 1 / (1 + np.exp(-np.clip(Z2, -250, 250)))  # sigmoid
    
    # Loss (MSE)
    loss = np.mean((A2 - y) ** 2)
    
    # Backward pass
    m = batch_size
    
    # Output layer
    dA2 = 2 * (A2 - y) / m
    dZ2 = dA2 * A2 * (1 - A2)  # sigmoid derivative
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)  # sigmoid derivative
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    print(f"Loss: {loss:.4f}")
    print("Gradients computed and weights updated!")

backpropagation_matrix_form()
```

---

## Gradient Descent

### What is Gradient Descent?

Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent (negative gradient).

**Mathematical Formulation:**
- `θ_new = θ_old - α * ∇J(θ)`
- Where `α` is learning rate and `∇J(θ)` is the gradient

### Variants of Gradient Descent

```python
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1

# Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """Uses all training data for each update"""
    weights = np.random.randn(X.shape[1])
    losses = []
    
    for epoch in range(epochs):
        predictions = X @ weights
        error = predictions - y
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        gradient = X.T @ error / len(X)
        weights -= learning_rate * gradient
    
    return weights, losses

# Stochastic Gradient Descent
def sgd(X, y, learning_rate=0.01, epochs=100):
    """Uses one sample at a time"""
    weights = np.random.randn(X.shape[1])
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(X))
        
        for i in indices:
            prediction = X[i] @ weights
            error = prediction - y[i]
            epoch_loss += error ** 2
            gradient = X[i] * error
            weights -= learning_rate * gradient
        
        losses.append(epoch_loss / len(X))
    
    return weights, losses

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    """Uses small batches (most common)"""
    weights = np.random.randn(X.shape[1])
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(X))
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            predictions = batch_X @ weights
            error = predictions - batch_y
            batch_loss = np.mean(error ** 2)
            epoch_loss += batch_loss
            
            gradient = batch_X.T @ error / len(batch_X)
            weights -= learning_rate * gradient
        
        losses.append(epoch_loss / (len(X) // batch_size))
    
    return weights, losses

# Compare methods
weights_batch, losses_batch = batch_gradient_descent(X, y, epochs=50)
weights_sgd, losses_sgd = sgd(X, y, epochs=50)
weights_mini, losses_mini = mini_batch_gd(X, y, batch_size=32, epochs=50)

# Plot convergence
plt.figure(figsize=(12, 5))
plt.plot(losses_batch, label='Batch GD', linewidth=2)
plt.plot(losses_sgd, label='SGD', linewidth=2, alpha=0.7)
plt.plot(losses_mini, label='Mini-batch GD', linewidth=2, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Gradient Descent Variants Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Final Losses:")
print(f"  Batch GD: {losses_batch[-1]:.6f}")
print(f"  SGD: {losses_sgd[-1]:.6f}")
print(f"  Mini-batch GD: {losses_mini[-1]:.6f}")
```

### Comparison of Variants

| Method | Batch Size | Pros | Cons | Use Case |
|--------|------------|------|------|----------|
| **Batch GD** | All data | Stable, accurate gradients | Slow, memory intensive | Small datasets |
| **SGD** | 1 | Fast, can escape local minima | Noisy, unstable | Online learning |
| **Mini-batch GD** | 32-256 | Balance of speed and stability | Need to tune batch size | Most common |

### Advanced Optimizers

```python
# Momentum: Adds velocity to gradient descent
def momentum_gd(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
    weights = np.random.randn(X.shape[1])
    velocity = np.zeros_like(weights)
    losses = []
    
    for epoch in range(epochs):
        predictions = X @ weights
        error = predictions - y
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        gradient = X.T @ error / len(X)
        velocity = momentum * velocity + learning_rate * gradient
        weights -= velocity
    
    return weights, losses

# RMSprop: Adaptive learning rate
def rmsprop_gd(X, y, learning_rate=0.01, decay=0.9, epochs=100):
    weights = np.random.randn(X.shape[1])
    cache = np.zeros_like(weights)
    losses = []
    epsilon = 1e-8
    
    for epoch in range(epochs):
        predictions = X @ weights
        error = predictions - y
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        gradient = X.T @ error / len(X)
        cache = decay * cache + (1 - decay) * gradient ** 2
        weights -= learning_rate * gradient / (np.sqrt(cache) + epsilon)
    
    return weights, losses

# Compare optimizers
weights_momentum, losses_momentum = momentum_gd(X, y, epochs=50)
weights_rmsprop, losses_rmsprop = rmsprop_gd(X, y, epochs=50)

plt.figure(figsize=(12, 5))
plt.plot(losses_batch, label='Batch GD', linewidth=2)
plt.plot(losses_momentum, label='Momentum', linewidth=2)
plt.plot(losses_rmsprop, label='RMSprop', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Optimizer Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Loss Functions

### Common Loss Functions

**For Regression:**
- **Mean Squared Error (MSE)**: `L = (1/n) * Σ(y_pred - y_true)²`
- **Mean Absolute Error (MAE)**: `L = (1/n) * Σ|y_pred - y_true|`

**For Classification:**
- **Binary Cross-Entropy**: `L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]`
- **Categorical Cross-Entropy**: `L = -Σ y*log(ŷ)`

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-15):
    """Binary Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-15):
    """Categorical Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage
y_true_binary = np.array([1, 0, 1, 0])
y_pred_binary = np.array([0.9, 0.1, 0.8, 0.2])
print(f"Binary CE Loss: {binary_crossentropy_loss(y_true_binary, y_pred_binary):.4f}")

y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_cat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
print(f"Categorical CE Loss: {categorical_crossentropy_loss(y_true_cat, y_pred_cat):.4f}")
```

## Weight Initialization

### Why Initialization Matters

Poor initialization can lead to:
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Symmetry breaking (all weights same)

### Common Initialization Methods

```python
def xavier_init(fan_in, fan_out):
    """Xavier/Glorot initialization (for tanh/sigmoid)"""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_init(fan_in, fan_out):
    """He initialization (for ReLU)"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def random_init(fan_in, fan_out, scale=0.1):
    """Simple random initialization"""
    return np.random.randn(fan_in, fan_out) * scale

# Compare initializations
fan_in, fan_out = 100, 50

weights_xavier = xavier_init(fan_in, fan_out)
weights_he = he_init(fan_in, fan_out)
weights_random = random_init(fan_in, fan_out)

print("Weight Initialization Statistics:")
print(f"Xavier: mean={weights_xavier.mean():.4f}, std={weights_xavier.std():.4f}")
print(f"He: mean={weights_he.mean():.4f}, std={weights_he.std():.4f}")
print(f"Random: mean={weights_random.mean():.4f}, std={weights_random.std():.4f}")
```

## Practice Exercises

### Exercise 1: Build Perceptron for AND Gate

**Task:** Implement perceptron for AND gate and visualize decision boundary.

**Solution:**
```python
# See Perceptron class above
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_and, y_and)
predictions = perceptron.predict(X_and)

print("AND Gate Results:")
for x, y_true, y_pred in zip(X_and, y_and, predictions):
    print(f"  Input: {x}, Target: {y_true}, Prediction: {y_pred}")

perceptron.plot_decision_boundary(X_and, y_and)
```

### Exercise 2: Build MLP for XOR Problem

**Task:** Implement MLP to solve XOR problem (which perceptron cannot solve).

**Solution:**
```python
# XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # Reshape for MLP

# MLP with hidden layer
mlp = MLP(layers=[2, 4, 1], learning_rate=0.5)  # 2 inputs, 4 hidden, 1 output
mlp.fit(X_xor, y_xor, epochs=5000)

predictions = mlp.predict(X_xor)
print("\nXOR Gate Results (MLP):")
for x, y_true, y_pred in zip(X_xor, y_xor, predictions):
    print(f"  Input: {x}, Target: {y_true[0]}, Prediction: {y_pred[0]:.3f}")
```

### Exercise 3: Compare Activation Functions

**Task:** Compare different activation functions on same network.

**Solution:**
```python
# Modify MLP to use different activations
# Test sigmoid, tanh, and ReLU
# Compare training speed and final accuracy
```

---

## Key Takeaways

1. **Perceptron**: Basic building block, can only solve linearly separable problems
2. **MLP**: Multiple layers enable non-linear learning, can solve XOR and complex problems
3. **Activation Functions**: Introduce non-linearity (ReLU most common for hidden layers)
4. **Backpropagation**: Chain rule enables efficient gradient computation
5. **Gradient Descent**: Optimization algorithm (mini-batch most common)
6. **Loss Functions**: MSE for regression, Cross-Entropy for classification
7. **Weight Initialization**: Critical for training (Xavier for tanh/sigmoid, He for ReLU)
8. **Learning Rate**: Critical hyperparameter - too high (divergence), too low (slow convergence)

---

## Best Practices

### Training Neural Networks

1. **Data Preprocessing**:
   - Normalize/standardize inputs
   - Handle missing values
   - Shuffle data

2. **Architecture**:
   - Start simple, add complexity gradually
   - Use appropriate activation functions
   - Initialize weights properly

3. **Training**:
   - Use mini-batch gradient descent
   - Monitor training and validation loss
   - Use early stopping to prevent overfitting
   - Tune learning rate carefully

4. **Regularization**:
   - L1/L2 regularization
   - Dropout (covered in deep learning)
   - Batch normalization (covered in deep learning)

5. **Debugging**:
   - Check gradients (should not vanish/explode)
   - Monitor loss (should decrease)
   - Visualize activations
   - Start with small network to verify code

---

## Next Steps

- Practice building networks from scratch
- Experiment with different architectures
- Learn about regularization techniques
- Move to [10-deep-learning-frameworks](../10-deep-learning-frameworks/README.md) for frameworks

**Remember**: Understanding fundamentals helps when using frameworks! Master the basics before moving to high-level APIs.

