# Prerequisites Project Tutorial: Build a Neural Network from Scratch

Complete step-by-step tutorial building a neural network from scratch using only NumPy, combining Python, mathematics, and data manipulation skills.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Understanding the Math](#step-1-understanding-the-math)
- [Step 2: Implementing the Neural Network](#step-2-implementing-the-neural-network)
- [Step 3: Training the Network](#step-3-training-the-network)
- [Step 4: Testing and Evaluation](#step-4-testing-and-evaluation)
- [Step 5: Improvements and Extensions](#step-5-improvements-and-extensions)

---

## Project Overview

**Goal**: Build a fully functional neural network from scratch using only NumPy

**What You'll Learn**:
- Linear algebra operations (matrix multiplication, dot products)
- Calculus (gradients, chain rule, backpropagation)
- Statistics (data normalization, evaluation metrics)
- Python programming (classes, NumPy operations)
- Optimization (gradient descent)

**Dataset**: We'll use a simple binary classification problem (XOR problem)

**Time**: 2-3 hours

---

## Step 1: Understanding the Math

### Neural Network Basics

A neural network consists of:
1. **Input Layer**: Receives input data
2. **Hidden Layers**: Process the data
3. **Output Layer**: Produces predictions

**Forward Pass**:
```
z = W * x + b
a = activation(z)
```

**Backward Pass (Backpropagation)**:
```
Compute gradients using chain rule
Update weights using gradient descent
```

### Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

# Visualize activation functions
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Step 2: Implementing the Neural Network

### Neural Network Class

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        """
        Initialize neural network
        
        Args:
            layers: List of layer sizes, e.g., [2, 4, 1] for 2 inputs, 4 hidden, 1 output
            learning_rate: Learning rate for gradient descent
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (n_samples, n_features)
        
        Returns:
            activations: List of activations for each layer
        """
        activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            # Activation (sigmoid for hidden, sigmoid for output)
            if i < len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self.sigmoid(z)  # Output layer
            activations.append(a)
            current_input = a
        
        return activations
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def backward(self, activations, y):
        """
        Backward propagation
        
        Args:
            activations: List of activations from forward pass
            y: True labels
        
        Returns:
            dW: List of weight gradients
            dB: List of bias gradients
        """
        m = y.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        output = activations[-1]
        error = output - y
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient of activation
            if i == len(self.weights) - 1:
                # Output layer
                delta = error * self.sigmoid_derivative(activations[i+1])
            else:
                # Hidden layer
                delta = np.dot(delta, self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            
            # Compute gradients
            dW[i] = np.dot(activations[i].T, delta) / m
            dB[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, dB
    
    def update_weights(self, dW, dB):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * dB[i]
    
    def predict(self, X):
        """Make predictions"""
        activations = self.forward(X)
        return activations[-1]
    
    def compute_loss(self, y_pred, y_true):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss
```

---

## Step 3: Training the Network

### Prepare Data

```python
# XOR problem (non-linearly separable)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("Input data:")
print(X)
print("\nTarget labels:")
print(y)
```

### Training Function

```python
def train(nn, X, y, epochs=10000, print_every=1000):
    """
    Train the neural network
    
    Args:
        nn: NeuralNetwork instance
        X: Input data
        y: Target labels
        epochs: Number of training iterations
        print_every: Print loss every N epochs
    """
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        activations = nn.forward(X)
        y_pred = activations[-1]
        
        # Compute loss
        loss = nn.compute_loss(y_pred, y)
        losses.append(loss)
        
        # Backward pass
        dW, dB = nn.backward(activations, y)
        
        # Update weights
        nn.update_weights(dW, dB)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    return losses

# Create and train network
nn = NeuralNetwork(layers=[2, 4, 1], learning_rate=0.5)
losses = train(nn, X, y, epochs=10000, print_every=1000)

# Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()
```

---

## Step 4: Testing and Evaluation

### Make Predictions

```python
# Make predictions
predictions = nn.predict(X)
print("\nPredictions:")
print(predictions)

# Convert to binary predictions
binary_predictions = (predictions > 0.5).astype(int)
print("\nBinary Predictions:")
print(binary_predictions)

# Compare with true labels
print("\nTrue Labels:")
print(y)
print("\nAccuracy:", np.mean(binary_predictions == y))
```

### Test on New Data

```python
# Test on new data points
test_X = np.array([[0.5, 0.5],
                   [0.2, 0.8],
                   [0.8, 0.2]])

test_predictions = nn.predict(test_X)
print("\nTest Predictions:")
print(test_predictions)
```

### Visualize Decision Boundary

```python
def plot_decision_boundary(nn, X, y):
    """Plot decision boundary"""
    # Create a mesh
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=100, cmap='RdYlBu', edgecolors='black')
    plt.colorbar()
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(nn, X, y)
```

---

## Step 5: Improvements and Extensions

### Add More Activation Functions

```python
class ImprovedNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, learning_rate=0.1, activation='sigmoid'):
        super().__init__(layers, learning_rate)
        self.activation = activation
    
    def activate(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return self.sigmoid(x)
    
    def activate_derivative(self, x):
        """Apply activation derivative"""
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            return self.sigmoid_derivative(x)
```

### Add Momentum

```python
class NeuralNetworkWithMomentum(NeuralNetwork):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers, learning_rate)
        self.momentum = momentum
        self.velocity_W = [np.zeros_like(w) for w in self.weights]
        self.velocity_B = [np.zeros_like(b) for b in self.biases]
    
    def update_weights(self, dW, dB):
        """Update weights with momentum"""
        for i in range(len(self.weights)):
            # Update velocity
            self.velocity_W[i] = self.momentum * self.velocity_W[i] - self.learning_rate * dW[i]
            self.velocity_B[i] = self.momentum * self.velocity_B[i] - self.learning_rate * dB[i]
            
            # Update weights
            self.weights[i] += self.velocity_W[i]
            self.biases[i] += self.velocity_B[i]
```

### Test on Real Dataset

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          random_state=42)
y = y.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train network
nn = NeuralNetwork(layers=[2, 8, 4, 1], learning_rate=0.1)
losses = train(nn, X_train_scaled, y_train, epochs=5000, print_every=500)

# Evaluate
train_pred = (nn.predict(X_train_scaled) > 0.5).astype(int)
test_pred = (nn.predict(X_test_scaled) > 0.5).astype(int)

print(f"\nTrain Accuracy: {np.mean(train_pred == y_train):.4f}")
print(f"Test Accuracy: {np.mean(test_pred == y_test):.4f}")
```

---

## Complete Code

```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
            current_input = a
        
        return activations
    
    def backward(self, activations, y):
        m = y.shape[0]
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]
        
        output = activations[-1]
        error = output - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = error * self.sigmoid_derivative(activations[i+1])
            else:
                delta = np.dot(delta, self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            
            dW[i] = np.dot(activations[i].T, delta) / m
            dB[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, dB
    
    def update_weights(self, dW, dB):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * dB[i]
    
    def predict(self, X):
        return self.forward(X)[-1]
    
    def compute_loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train
nn = NeuralNetwork(layers=[2, 4, 1], learning_rate=0.5)

for epoch in range(10000):
    activations = nn.forward(X)
    loss = nn.compute_loss(activations[-1], y)
    dW, dB = nn.backward(activations, y)
    nn.update_weights(dW, dB)
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Predict
predictions = nn.predict(X)
print("\nPredictions:", predictions)
print("Accuracy:", np.mean((predictions > 0.5).astype(int) == y))
```

---

## Key Takeaways

1. **Math in Action**: You've applied linear algebra, calculus, and statistics
2. **Python Skills**: Used classes, NumPy, and object-oriented programming
3. **Understanding**: You now understand how neural networks work internally
4. **Foundation**: This knowledge will help you understand deep learning frameworks

---

**Congratulations!** You've built a neural network from scratch using only NumPy!

