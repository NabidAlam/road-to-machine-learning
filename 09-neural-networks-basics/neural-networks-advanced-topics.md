# Advanced Neural Networks Topics

Comprehensive guide to advanced neural network concepts and techniques.

## Table of Contents

- [Advanced Architectures](#advanced-architectures)
- [Advanced Optimization](#advanced-optimization)
- [Regularization Techniques](#regularization-techniques)
- [Batch Normalization](#batch-normalization)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Debugging Neural Networks](#debugging-neural-networks)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Architectures

### Deep Neural Networks

Networks with many hidden layers.

```python
class DeepMLP:
    def __init__(self, layers, learning_rate=0.01, activation='relu'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights with He initialization for ReLU
        for i in range(len(layers) - 1):
            if activation == 'relu':
                std = np.sqrt(2.0 / layers[i])
            else:
                std = np.sqrt(1.0 / layers[i])
            w = np.random.randn(layers[i], layers[i+1]) * std
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if self.activation == 'relu':
                a = self.relu(z)
            else:
                a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            activations.append(a)
        return activations
    
    def backward(self, activations, y):
        m = y.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer
        error = activations[-1] - y
        if self.activation == 'relu':
            delta = error * self.relu_derivative(activations[-1])
        else:
            delta = error * activations[-1] * (1 - activations[-1])
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                if self.activation == 'relu':
                    delta = delta * self.relu_derivative(activations[i])
                else:
                    delta = delta * activations[i] * (1 - activations[i])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        n_samples = len(X)
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                activations = self.forward(batch_X)
                grad_w, grad_b = self.backward(activations, batch_y)
                
                # Update weights
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * grad_w[j]
                    self.biases[j] -= self.learning_rate * grad_b[j]
            
            if epoch % 100 == 0:
                activations = self.forward(X)
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Residual Connections

Skip connections that help with deep networks.

```python
class ResidualBlock:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, X):
        # Main path
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        
        # Residual connection (if dimensions match)
        if X.shape[1] == z2.shape[1]:
            return z2 + X  # Skip connection
        else:
            return z2
```

---

## Advanced Optimization

### Adam Optimizer

Adaptive Moment Estimation - combines momentum and RMSprop.

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, weights, gradients, key):
        """Update weights using Adam"""
        if key not in self.m:
            self.m[key] = np.zeros_like(weights)
            self.v[key] = np.zeros_like(weights)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        
        # Update weights
        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return weights

# Usage example
adam = AdamOptimizer(learning_rate=0.001)
weights = np.random.randn(10, 5)
gradients = np.random.randn(10, 5)
weights = adam.update(weights, gradients, 'layer1')
```

### Learning Rate Scheduling

Adjust learning rate during training.

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_type='exponential', decay_rate=0.95):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.epoch = 0
    
    def get_lr(self):
        """Get current learning rate"""
        if self.decay_type == 'exponential':
            return self.initial_lr * (self.decay_rate ** self.epoch)
        elif self.decay_type == 'step':
            return self.initial_lr * (self.decay_rate ** (self.epoch // 10))
        elif self.decay_type == 'polynomial':
            return self.initial_lr / (1 + self.epoch * self.decay_rate)
        else:
            return self.initial_lr
    
    def step(self):
        """Increment epoch"""
        self.epoch += 1

# Usage
scheduler = LearningRateScheduler(initial_lr=0.01, decay_type='exponential')
for epoch in range(100):
    current_lr = scheduler.get_lr()
    # Use current_lr for training
    scheduler.step()
```

---

## Regularization Techniques

### L1 and L2 Regularization

Prevent overfitting by penalizing large weights.

```python
def l2_regularization(weights, lambda_reg=0.01):
    """L2 regularization (Ridge)"""
    return lambda_reg * np.sum(weights ** 2)

def l1_regularization(weights, lambda_reg=0.01):
    """L1 regularization (Lasso)"""
    return lambda_reg * np.sum(np.abs(weights))

def elastic_net(weights, lambda_l1=0.01, lambda_l2=0.01):
    """Elastic Net (L1 + L2)"""
    return lambda_l1 * np.sum(np.abs(weights)) + lambda_l2 * np.sum(weights ** 2)

# Add to loss function
def compute_loss_with_regularization(y_true, y_pred, weights, lambda_reg=0.01):
    mse_loss = np.mean((y_true - y_pred) ** 2)
    reg_loss = l2_regularization(weights, lambda_reg)
    total_loss = mse_loss + reg_loss
    return total_loss

# Gradient includes regularization term
def compute_gradient_with_regularization(X, y, weights, lambda_reg=0.01):
    predictions = X @ weights
    error = predictions - y
    gradient = X.T @ error / len(X)
    gradient += lambda_reg * weights  # L2 regularization gradient
    return gradient
```

### Dropout

Randomly set some neurons to zero during training.

```python
def dropout(X, dropout_rate=0.5, training=True):
    """Dropout regularization"""
    if not training:
        return X
    
    # Create dropout mask
    mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape) / (1 - dropout_rate)
    return X * mask

# Usage in forward pass
def forward_with_dropout(X, weights, biases, dropout_rate=0.5, training=True):
    z = X @ weights + biases
    a = np.maximum(0, z)  # ReLU
    a = dropout(a, dropout_rate, training)
    return a
```

---

## Batch Normalization

Normalize activations to stabilize training.

```python
class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.gamma = 1.0  # Scale parameter
        self.beta = 0.0   # Shift parameter
    
    def forward(self, X, training=True):
        """Forward pass"""
        if training:
            # Compute batch statistics
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            
            # Update running statistics
            if self.running_mean is None:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # Use running statistics during inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * X_norm + self.beta
        
        return out

# Usage
bn = BatchNormalization()
X_normalized = bn.forward(X, training=True)
```

---

## Hyperparameter Tuning

### Grid Search

```python
def grid_search_hyperparameters(X_train, y_train, X_val, y_val):
    """Grid search for hyperparameters"""
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [32, 64, 128]
    batch_sizes = [16, 32, 64]
    
    best_score = float('inf')
    best_params = None
    
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                # Train model
                model = MLP(layers=[X_train.shape[1], hidden_size, 1], learning_rate=lr)
                model.fit(X_train, y_train, epochs=100, batch_size=batch_size)
                
                # Evaluate
                predictions = model.predict(X_val)
                score = np.mean((predictions - y_val) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_params = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size}
    
    return best_params, best_score
```

### Random Search

```python
def random_search_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20):
    """Random search for hyperparameters"""
    best_score = float('inf')
    best_params = None
    
    for _ in range(n_trials):
        lr = np.random.uniform(0.001, 0.1)
        hidden_size = np.random.choice([32, 64, 128, 256])
        batch_size = np.random.choice([16, 32, 64])
        
        # Train and evaluate
        model = MLP(layers=[X_train.shape[1], hidden_size, 1], learning_rate=lr)
        model.fit(X_train, y_train, epochs=100, batch_size=batch_size)
        
        predictions = model.predict(X_val)
        score = np.mean((predictions - y_val) ** 2)
        
        if score < best_score:
            best_score = score
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size}
    
    return best_params, best_score
```

---

## Debugging Neural Networks

### Gradient Checking

Verify backpropagation implementation.

```python
def gradient_check(model, X, y, epsilon=1e-7):
    """Numerical gradient checking"""
    # Compute analytical gradients
    activations = model.forward(X)
    grad_w, grad_b = model.backward(activations, y)
    
    # Compute numerical gradients
    numerical_grad_w = []
    for i, w in enumerate(model.weights):
        num_grad = np.zeros_like(w)
        for j in range(w.size):
            # Perturb weight
            w_flat = w.flatten()
            w_flat[j] += epsilon
            w_perturbed = w_flat.reshape(w.shape)
            
            # Compute loss with perturbed weight
            model.weights[i] = w_perturbed
            activations_pert = model.forward(X)
            loss_pert = np.mean((activations_pert[-1] - y) ** 2)
            
            # Reset weight
            w_flat[j] -= 2 * epsilon
            w_perturbed = w_flat.reshape(w.shape)
            model.weights[i] = w_perturbed
            activations_pert = model.forward(X)
            loss_pert2 = np.mean((activations_pert[-1] - y) ** 2)
            
            # Numerical gradient
            num_grad.flat[j] = (loss_pert - loss_pert2) / (2 * epsilon)
            
            # Reset weight
            w_flat[j] += epsilon
            model.weights[i] = w_flat.reshape(w.shape)
        
        numerical_grad_w.append(num_grad)
        
        # Compare
        diff = np.abs(grad_w[i] - num_grad)
        print(f"Layer {i} gradient difference: {np.max(diff):.2e}")
    
    return numerical_grad_w
```

### Monitoring Training

```python
def monitor_training(model, X_train, y_train, X_val, y_val, epochs=1000):
    """Monitor training progress"""
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        activations = model.forward(X_train)
        grad_w, grad_b = model.backward(activations, y_train)
        
        # Update weights
        for i in range(len(model.weights)):
            model.weights[i] -= model.learning_rate * grad_w[i]
            model.biases[i] -= model.learning_rate * grad_b[i]
        
        # Compute losses
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val) ** 2)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return train_losses, val_losses
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Vanishing Gradients

**Problem**: Gradients become very small in deep networks

**Solution**:
- Use ReLU instead of sigmoid/tanh
- Use proper weight initialization (He/Xavier)
- Use batch normalization
- Use residual connections

### Pitfall 2: Exploding Gradients

**Problem**: Gradients become very large

**Solution**:
- Gradient clipping
- Lower learning rate
- Proper weight initialization
- Batch normalization

### Pitfall 3: Overfitting

**Problem**: Model memorizes training data

**Solution**:
- Add regularization (L1/L2)
- Use dropout
- Early stopping
- More training data
- Reduce model complexity

### Pitfall 4: Learning Rate Too High/Low

**Problem**: Training diverges or converges too slowly

**Solution**:
- Start with learning rate around 0.001-0.01
- Use learning rate scheduling
- Try adaptive optimizers (Adam)
- Monitor loss curves

### Pitfall 5: Poor Weight Initialization

**Problem**: Network doesn't learn or learns slowly

**Solution**:
- Use He initialization for ReLU
- Use Xavier initialization for tanh/sigmoid
- Don't initialize all weights to zero
- Don't initialize weights too large

---

## Key Takeaways

1. **Deep Networks**: More layers can learn complex patterns but harder to train
2. **Advanced Optimizers**: Adam combines benefits of momentum and RMSprop
3. **Regularization**: L1/L2, dropout prevent overfitting
4. **Batch Normalization**: Stabilizes training in deep networks
5. **Hyperparameter Tuning**: Grid search, random search, or Bayesian optimization
6. **Debugging**: Gradient checking, monitoring training, visualizing activations
7. **Common Issues**: Vanishing/exploding gradients, overfitting, poor initialization

---

## Next Steps

- Practice implementing advanced techniques
- Experiment with different architectures
- Learn about convolutional and recurrent networks
- Move to deep learning frameworks module

**Remember**: Understanding these advanced concepts is crucial for building effective neural networks!

