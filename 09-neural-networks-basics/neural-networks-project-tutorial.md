# Complete Neural Network Project Tutorial

Step-by-step walkthrough of building a neural network from scratch for classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Preparation](#step-1-data-preparation)
- [Step 2: Build Neural Network from Scratch](#step-2-build-neural-network-from-scratch)
- [Step 3: Implement Forward Propagation](#step-3-implement-forward-propagation)
- [Step 4: Implement Backpropagation](#step-4-implement-backpropagation)
- [Step 5: Train the Network](#step-5-train-the-network)
- [Step 6: Evaluate and Visualize](#step-6-evaluate-and-visualize)
- [Step 7: Improve the Model](#step-7-improve-the-model)

---

## Project Overview

**Project**: Build Neural Network from Scratch for Classification

**Dataset**: Synthetic classification dataset (can be replaced with real data)

**Goals**:
1. Implement neural network from scratch (no frameworks)
2. Understand forward and backward propagation
3. Train network to classify data
4. Visualize learning process
5. Improve model performance

**Type**: Classification with Neural Networks

**Difficulty**: Intermediate

**Time**: 2-3 hours

---

## Step 1: Data Preparation

### Generate/Load Data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Alternative: Use moons dataset (non-linear)
# X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Visualize data
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.6, s=50, label='Class 0', edgecolors='black')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.6, s=50, label='Class 1', edgecolors='black')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Classification Dataset', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")
```

### Preprocess Data

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape y for neural network (column vector)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
```

---

## Step 2: Build Neural Network from Scratch

### Define Neural Network Class

```python
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, activation='sigmoid'):
        """
        Initialize neural network
        
        Parameters:
        - layers: list of layer sizes, e.g., [2, 4, 1] for 2 inputs, 4 hidden, 1 output
        - learning_rate: learning rate for gradient descent
        - activation: activation function ('sigmoid' or 'relu')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.history = {'loss': [], 'accuracy': []}
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization for sigmoid, He for ReLU
            if activation == 'relu':
                std = np.sqrt(2.0 / layers[i])
            else:
                std = np.sqrt(1.0 / layers[i])
            
            w = np.random.randn(layers[i], layers[i+1]) * std
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def activate(self, x):
        """Apply activation function"""
        if self.activation == 'relu':
            return self.relu(x)
        else:
            return self.sigmoid(x)
    
    def activate_derivative(self, x):
        """Apply activation derivative"""
        if self.activation == 'relu':
            return self.relu_derivative(x)
        else:
            return self.sigmoid_derivative(x)
```

---

## Step 3: Implement Forward Propagation

### Forward Pass

```python
    def forward(self, X):
        """
        Forward propagation
        
        Returns:
        - activations: list of activations for each layer
        """
        activations = [X]  # Input layer
        
        for i in range(len(self.weights)):
            # Compute weighted sum
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            # Apply activation function
            a = self.activate(z)
            activations.append(a)
        
        return activations
    
    def predict(self, X):
        """Make predictions"""
        activations = self.forward(X)
        predictions = activations[-1]
        # For binary classification, return class labels
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return probability predictions"""
        activations = self.forward(X)
        return activations[-1]
```

---

## Step 4: Implement Backpropagation

### Backward Pass

```python
    def backward(self, activations, y):
        """
        Backward propagation
        
        Returns:
        - gradients_w: gradients for weights
        - gradients_b: gradients for biases
        """
        m = y.shape[0]  # Number of samples
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        error = activations[-1] - y
        delta = error * self.activate_derivative(activations[-1])
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * self.activate_derivative(activations[i])
        
        return gradients_w, gradients_b
```

---

## Step 5: Train the Network

### Training Loop

```python
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy"""
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y_true)
        return accuracy
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True, X_val=None, y_val=None):
        """
        Train the neural network
        
        Parameters:
        - X: training features
        - y: training labels
        - epochs: number of training epochs
        - batch_size: batch size for mini-batch gradient descent
        - verbose: whether to print progress
        - X_val: validation features (optional)
        - y_val: validation labels (optional)
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Forward pass
                activations = self.forward(batch_X)
                
                # Compute loss
                batch_loss = self.compute_loss(batch_y, activations[-1])
                epoch_loss += batch_loss
                
                # Backward pass
                grad_w, grad_b = self.backward(activations, batch_y)
                
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * grad_w[j]
                    self.biases[j] -= self.learning_rate * grad_b[j]
            
            # Average loss for epoch
            avg_loss = epoch_loss / (n_samples // batch_size)
            
            # Compute accuracy
            train_pred = self.predict_proba(X)
            train_acc = self.compute_accuracy(y, train_pred)
            
            # Store history
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = self.compute_accuracy(y_val, val_pred)
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}, "
                          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            else:
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.4f}")

# Create and train network
nn = NeuralNetwork(layers=[2, 8, 4, 1], learning_rate=0.01, activation='sigmoid')

# Split validation set
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train
nn.fit(X_train_split, y_train_split, epochs=1000, batch_size=32, 
       X_val=X_val_split, y_val=y_val_split)
```

---

## Step 6: Evaluate and Visualize

### Evaluate Model

```python
# Evaluate on test set
test_pred = nn.predict(X_test_scaled)
test_proba = nn.predict_proba(X_test_scaled)
test_acc = nn.compute_accuracy(y_test, test_proba)

print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(nn.history['loss'], linewidth=2, label='Training Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(nn.history['accuracy'], linewidth=2, label='Training Accuracy', color='green')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Visualize Decision Boundary

```python
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot decision boundary"""
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], 
               c='red', s=50, alpha=0.8, label='Class 0', edgecolors='black')
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], 
               c='blue', s=50, alpha=0.8, label='Class 1', edgecolors='black')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot decision boundary
plot_decision_boundary(nn, X_test_scaled, y_test, "Neural Network Decision Boundary")
```

---

## Step 7: Improve the Model

### Try Different Architectures

```python
# Experiment with different architectures
architectures = [
    ([2, 4, 1], "Small Network"),
    ([2, 8, 4, 1], "Medium Network"),
    ([2, 16, 8, 4, 1], "Deep Network")
]

results = []

for layers, name in architectures:
    print(f"\nTraining {name}...")
    nn = NeuralNetwork(layers=layers, learning_rate=0.01, activation='sigmoid')
    nn.fit(X_train_split, y_train_split, epochs=500, batch_size=32, verbose=False)
    
    test_pred = nn.predict_proba(X_test_scaled)
    test_acc = nn.compute_accuracy(y_test, test_pred)
    
    results.append((name, test_acc))
    print(f"{name} Test Accuracy: {test_acc:.4f}")

print("\nArchitecture Comparison:")
for name, acc in results:
    print(f"  {name}: {acc:.4f}")
```

### Try Different Activation Functions

```python
# Compare activation functions
activations = ['sigmoid', 'relu']

for activation in activations:
    print(f"\nTraining with {activation} activation...")
    nn = NeuralNetwork(layers=[2, 8, 4, 1], learning_rate=0.01, activation=activation)
    nn.fit(X_train_split, y_train_split, epochs=500, batch_size=32, verbose=False)
    
    test_pred = nn.predict_proba(X_test_scaled)
    test_acc = nn.compute_accuracy(y_test, test_pred)
    
    print(f"{activation.capitalize()} Test Accuracy: {test_acc:.4f}")
```

### Tune Learning Rate

```python
# Try different learning rates
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}...")
    nn = NeuralNetwork(layers=[2, 8, 4, 1], learning_rate=lr, activation='sigmoid')
    nn.fit(X_train_split, y_train_split, epochs=500, batch_size=32, verbose=False)
    
    test_pred = nn.predict_proba(X_test_scaled)
    test_acc = nn.compute_accuracy(y_test, test_pred)
    
    print(f"LR={lr} Test Accuracy: {test_acc:.4f}")
```

### Final Model

```python
# Train final best model
print("\nTraining final model...")
final_nn = NeuralNetwork(layers=[2, 16, 8, 1], learning_rate=0.01, activation='sigmoid')
final_nn.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, 
             X_val=X_test_scaled, y_val=y_test)

# Final evaluation
final_pred = final_nn.predict(X_test_scaled)
final_proba = final_nn.predict_proba(X_test_scaled)
final_acc = final_nn.compute_accuracy(y_test, final_proba)

print(f"\nFinal Test Accuracy: {final_acc:.4f}")

# Plot final decision boundary
plot_decision_boundary(final_nn, X_test_scaled, y_test, "Final Model Decision Boundary")
```

---

## Key Takeaways

1. **Data Preprocessing**: Always scale features before training
2. **Weight Initialization**: Use proper initialization (Xavier/He)
3. **Forward Propagation**: Compute predictions layer by layer
4. **Backpropagation**: Use chain rule to compute gradients
5. **Training**: Use mini-batch gradient descent for efficiency
6. **Evaluation**: Monitor both loss and accuracy
7. **Visualization**: Plot decision boundaries to understand model
8. **Experimentation**: Try different architectures, activations, learning rates

---

**Congratulations!** You've built a neural network from scratch and trained it successfully!

