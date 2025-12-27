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

Neural networks are computing systems inspired by biological neural networks. They learn patterns from data.

**Key Components:**
- **Neurons**: Basic processing units
- **Layers**: Groups of neurons
- **Weights**: Learned parameters
- **Biases**: Offset parameters

---

## Perceptron

### Single Perceptron

Simplest neural network - single neuron.

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
                y_predicted = self.activation(linear_output)
                
                # Update weights
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [self.activation(x) for x in linear_output]

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron()
perceptron.fit(X, y)
predictions = perceptron.predict(X)
print("Predictions:", predictions)
```

**Limitation**: Can only solve linearly separable problems (cannot solve XOR).

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

### Common Activations

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# Tanh
tanh = np.tanh(x)

# ReLU
relu = np.maximum(0, x)

# Leaky ReLU
leaky_relu = np.maximum(0.01 * x, x)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, sigmoid)
axes[0, 0].set_title('Sigmoid')
axes[0, 1].plot(x, tanh)
axes[0, 1].set_title('Tanh')
axes[1, 0].plot(x, relu)
axes[1, 0].set_title('ReLU')
axes[1, 1].plot(x, leaky_relu)
axes[1, 1].set_title('Leaky ReLU')
plt.tight_layout()
plt.show()
```

### When to Use

- **Sigmoid**: Output layer for binary classification
- **Tanh**: Hidden layers (better than sigmoid)
- **ReLU**: Most common for hidden layers
- **Softmax**: Output layer for multiclass classification

---

## Backpropagation

### Understanding Backpropagation

Algorithm for training neural networks using chain rule.

**Process:**
1. Forward pass: Compute predictions
2. Calculate loss
3. Backward pass: Compute gradients
4. Update weights

```python
# Simplified backpropagation example
def backpropagation_example():
    # Forward
    x = 2
    w1 = 0.5
    w2 = 0.3
    b = 0.1
    
    z1 = x * w1 + b
    a1 = sigmoid(z1)
    z2 = a1 * w2
    y_pred = sigmoid(z2)
    
    # Loss (assuming target = 1)
    y_true = 1
    loss = (y_pred - y_true) ** 2
    
    # Backward (chain rule)
    dloss_dypred = 2 * (y_pred - y_true)
    dypred_dz2 = sigmoid_derivative(y_pred)
    dz2_dw2 = a1
    
    dloss_dw2 = dloss_dypred * dypred_dz2 * dz2_dw2
    
    print(f"Gradient for w2: {dloss_dw2}")
```

---

## Gradient Descent

### Variants

```python
# Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    weights = np.random.randn(X.shape[1])
    for epoch in range(epochs):
        predictions = X @ weights
        error = predictions - y
        gradient = X.T @ error / len(X)
        weights -= learning_rate * gradient
    return weights

# Stochastic Gradient Descent
def sgd(X, y, learning_rate=0.01, epochs=100):
    weights = np.random.randn(X.shape[1])
    for epoch in range(epochs):
        for i in range(len(X)):
            prediction = X[i] @ weights
            error = prediction - y[i]
            gradient = X[i] * error
            weights -= learning_rate * gradient
    return weights

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    weights = np.random.randn(X.shape[1])
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            predictions = batch_X @ weights
            error = predictions - batch_y
            gradient = batch_X.T @ error / len(batch_X)
            weights -= learning_rate * gradient
    return weights
```

---

## Practice Exercises

### Exercise 1: Build Perceptron

**Task:** Implement perceptron for AND gate.

**Solution:**
```python
# See Perceptron class above
perceptron = Perceptron()
perceptron.fit(X, y)
predictions = perceptron.predict(X)
print("AND Gate Predictions:", predictions)
```

---

## Key Takeaways

1. **Perceptron**: Basic building block
2. **MLP**: Multiple layers enable non-linear learning
3. **Activation Functions**: Introduce non-linearity
4. **Backpropagation**: How networks learn
5. **Gradient Descent**: Optimization algorithm

---

## Next Steps

- Practice building networks from scratch
- Move to [10-deep-learning-frameworks](../10-deep-learning-frameworks/README.md) for frameworks

**Remember**: Understanding fundamentals helps when using frameworks!

