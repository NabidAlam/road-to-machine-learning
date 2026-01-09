# Calculus for Machine Learning

Comprehensive guide to calculus concepts essential for understanding optimization, gradient descent, and training machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Derivatives](#derivatives)
- [Partial Derivatives](#partial-derivatives)
- [Gradients](#gradients)
- [Gradient Descent](#gradient-descent)
- [Chain Rule and Backpropagation](#chain-rule-and-backpropagation)
- [Optimization](#optimization)
- [Applications in ML](#applications-in-ml)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Calculus Matters in ML

Calculus is the **mathematics of change** and is essential for:

- **Optimization**: Finding best model parameters (gradient descent)
- **Training Neural Networks**: Backpropagation uses chain rule
- **Loss Functions**: Derivatives show how to minimize error
- **Hyperparameter Tuning**: Understanding learning rates
- **Feature Engineering**: Understanding feature importance

### What You'll Learn

- Derivatives and their geometric meaning
- Partial derivatives for multivariable functions
- Gradients and their direction
- Gradient descent algorithm
- Chain rule for backpropagation
- Optimization techniques

---

## Derivatives

### What is a Derivative?

The **derivative** measures the **rate of change** of a function.

**Geometric Interpretation**: Slope of the tangent line at a point.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Function: f(x) = x²
def f(x):
    return x**2

# Derivative: f'(x) = 2x
def df(x):
    return 2*x

# Evaluate at x = 3
x = 3
print(f"f(3) = {f(3)}")      # 9
print(f"f'(3) = {df(3)}")    # 6 (slope at x=3)

# Visualize
x_vals = np.linspace(-5, 5, 100)
y_vals = f(x_vals)
tangent_slope = df(3)
tangent_line = f(3) + tangent_slope * (x_vals - 3)

plt.plot(x_vals, y_vals, label='f(x) = x²')
plt.plot(x_vals, tangent_line, 'r--', label=f"Tangent at x=3 (slope={tangent_slope})")
plt.scatter([3], [f(3)], color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

### Common Derivatives

```python
# Power rule: d/dx(xⁿ) = n*xⁿ⁻¹
# f(x) = x³ → f'(x) = 3x²

# Product rule: d/dx(f*g) = f'*g + f*g'
# f(x) = x * sin(x) → f'(x) = 1*sin(x) + x*cos(x)

# Quotient rule: d/dx(f/g) = (f'*g - f*g')/g²

# Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
# f(x) = (x² + 1)³ → f'(x) = 3(x² + 1)² * 2x

# Common functions
# d/dx(e^x) = e^x
# d/dx(ln(x)) = 1/x
# d/dx(sin(x)) = cos(x)
# d/dx(cos(x)) = -sin(x)
```

### Numerical Derivatives

```python
# Approximate derivative using finite differences
def numerical_derivative(f, x, h=1e-5):
    """
    Approximate derivative using central difference
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

# Example
def f(x):
    return x**2

x = 3
exact = df(3)  # 6
approx = numerical_derivative(f, x)
print(f"Exact derivative: {exact}")
print(f"Numerical approximation: {approx:.6f}")
print(f"Error: {abs(exact - approx):.2e}")
```

### Second Derivatives

```python
# Second derivative: rate of change of the first derivative
# f(x) = x³
# f'(x) = 3x²
# f''(x) = 6x

def f(x):
    return x**3

def df(x):
    return 3*x**2

def d2f(x):
    return 6*x

# Second derivative indicates concavity
# f''(x) > 0: Concave up (minimum)
# f''(x) < 0: Concave down (maximum)
```

---

## Partial Derivatives

### What are Partial Derivatives?

For functions of multiple variables, **partial derivatives** measure change with respect to one variable while keeping others constant.

```python
# Function: f(x, y) = x² + y²
# Partial with respect to x: ∂f/∂x = 2x
# Partial with respect to y: ∂f/∂y = 2y

def f(x, y):
    return x**2 + y**2

def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return 2*y

# At point (3, 4)
x, y = 3, 4
print(f"f(3, 4) = {f(x, y)}")           # 25
print(f"∂f/∂x at (3,4) = {df_dx(x, y)}")  # 6
print(f"∂f/∂y at (3,4) = {df_dy(x, y)}")  # 8
```

### Numerical Partial Derivatives

```python
def numerical_partial_derivative(f, x, y, var='x', h=1e-5):
    """
    Approximate partial derivative
    """
    if var == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    else:  # var == 'y'
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

# Example
approx_dx = numerical_partial_derivative(f, 3, 4, 'x')
approx_dy = numerical_partial_derivative(f, 3, 4, 'y')
print(f"∂f/∂x ≈ {approx_dx:.6f}")  # Should be close to 6
print(f"∂f/∂y ≈ {approx_dy:.6f}")  # Should be close to 8
```

---

## Gradients

### What is a Gradient?

The **gradient** is a vector of all partial derivatives. It points in the direction of **steepest ascent**.

```python
def gradient(f, x, y, h=1e-5):
    """
    Calculate gradient vector [∂f/∂x, ∂f/∂y]
    """
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# For f(x, y) = x² + y²
# Gradient: ∇f = [2x, 2y]

def f(x, y):
    return x**2 + y**2

def gradient_exact(x, y):
    return np.array([2*x, 2*y])

# At point (3, 4)
x, y = 3, 4
grad = gradient_exact(x, y)
print(f"Gradient at (3,4): {grad}")  # [6, 8]

# Gradient magnitude
grad_magnitude = np.linalg.norm(grad)
print(f"Gradient magnitude: {grad_magnitude:.2f}")  # 10.0
```

### Gradient Properties

```python
# 1. Gradient points in direction of steepest ascent
# 2. Negative gradient points in direction of steepest descent
# 3. Gradient is perpendicular to level curves

# Visualize gradient field
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Function values
Z = X**2 + Y**2

# Gradient components
U = 2*X  # ∂f/∂x
V = 2*Y  # ∂f/∂y

plt.contour(X, Y, Z, levels=20)
plt.quiver(X, Y, U, V, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Field of f(x,y) = x² + y²')
plt.axis('equal')
plt.show()
```

---

## Gradient Descent

### The Algorithm

**Gradient Descent** finds the minimum of a function by moving in the direction opposite to the gradient.

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, iterations=100, tolerance=1e-6):
    """
    Gradient descent optimization
    
    Parameters:
    - f: Function to minimize
    - grad_f: Gradient function
    - x0: Initial point
    - learning_rate: Step size
    - iterations: Maximum iterations
    - tolerance: Convergence tolerance
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(iterations):
        gradient = grad_f(x[0], x[1])
        
        # Update: move opposite to gradient
        x_new = x - learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        x = x_new
        history.append(x.copy())
    
    return x, history

# Example: Minimize f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# Start from (5, 5)
x0 = np.array([5.0, 5.0])
minimum, history = gradient_descent(f, grad_f, x0, learning_rate=0.1, iterations=100)

print(f"Minimum found at: {minimum}")
print(f"Function value: {f(minimum[0], minimum[1])}")  # Should be close to 0
```

### Learning Rate

```python
# Learning rate is crucial!

# Too small: Slow convergence
x_small_lr, _ = gradient_descent(f, grad_f, [5, 5], learning_rate=0.01, iterations=50)
print(f"Small LR result: {x_small_lr}")

# Too large: May diverge or oscillate
x_large_lr, _ = gradient_descent(f, grad_f, [5, 5], learning_rate=1.0, iterations=50)
print(f"Large LR result: {x_large_lr}")  # May not converge!

# Good learning rate
x_good_lr, _ = gradient_descent(f, grad_f, [5, 5], learning_rate=0.1, iterations=50)
print(f"Good LR result: {x_good_lr}")
```

### Visualizing Gradient Descent

```python
# 2D visualization
x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=20)
history_array = np.array(history)
plt.plot(history_array[:, 0], history_array[:, 1], 'ro-', markersize=5)
plt.scatter([0], [0], color='green', s=200, marker='*', label='True Minimum')
plt.scatter([history_array[0, 0]], [history_array[0, 1]], 
           color='blue', s=200, marker='o', label='Start')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Path')
plt.legend()
plt.axis('equal')
plt.show()
```

---

## Chain Rule and Backpropagation

### Chain Rule

The **chain rule** is fundamental for computing derivatives of composite functions.

```
If z = f(y) and y = g(x), then:
dz/dx = (dz/dy) * (dy/dx)
```

```python
# Example: z = (x² + 1)³
# Let y = x² + 1, then z = y³
# dz/dx = dz/dy * dy/dx = 3y² * 2x = 3(x² + 1)² * 2x

def z(x):
    return (x**2 + 1)**3

def dz_dx(x):
    return 3 * (x**2 + 1)**2 * 2*x

# Verify
x = 2
exact = dz_dx(x)
approx = numerical_derivative(z, x)
print(f"Exact: {exact}")
print(f"Approximate: {approx:.6f}")
```

### Backpropagation in Neural Networks

```python
# Simplified backpropagation example
# Network: input → hidden → output

# Forward pass
def forward_pass(x, w1, b1, w2, b2):
    z1 = w1 @ x + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = w2 @ a1 + b2
    return z2, a1, z1

# Loss function: MSE
def loss(y_pred, y_true):
    return 0.5 * np.sum((y_pred - y_true)**2)

# Backward pass (simplified)
def backward_pass(x, y_true, w1, b1, w2, b2):
    # Forward
    z2, a1, z1 = forward_pass(x, w1, b1, w2, b2)
    
    # Output layer gradient
    dL_dz2 = z2 - y_true
    
    # Backpropagate through layers (chain rule)
    dL_dw2 = dL_dz2 @ a1.T
    dL_db2 = dL_dz2
    
    dL_da1 = w2.T @ dL_dz2
    dL_dz1 = dL_da1 * (z1 > 0)  # ReLU derivative
    dL_dw1 = dL_dz1 @ x.T
    dL_db1 = dL_dz1
    
    return dL_dw1, dL_db1, dL_dw2, dL_db2

# Example usage
x = np.array([1, 2])
y_true = np.array([0.5])
w1 = np.random.randn(3, 2)
b1 = np.random.randn(3)
w2 = np.random.randn(1, 3)
b2 = np.random.randn(1)

grads = backward_pass(x, y_true, w1, b1, w2, b2)
print("Gradients computed using chain rule (backpropagation)")
```

---

## Optimization

### Local vs Global Minima

```python
# Some functions have multiple minima
def f_multiple_minima(x):
    return x**4 - 4*x**2 + x

x_vals = np.linspace(-3, 3, 100)
y_vals = f_multiple_minima(x_vals)

plt.plot(x_vals, y_vals)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function with Multiple Minima')
plt.grid(True)
plt.show()

# Gradient descent may find local minimum, not global
# Solution: Multiple random starts, or use global optimization
```

### Momentum

```python
def gradient_descent_with_momentum(f, grad_f, x0, learning_rate=0.1, 
                                   momentum=0.9, iterations=100):
    """
    Gradient descent with momentum
    Helps escape local minima and speeds convergence
    """
    x = np.array(x0, dtype=float)
    velocity = np.zeros_like(x)
    
    for i in range(iterations):
        gradient = grad_f(x[0], x[1])
        
        # Update velocity (exponentially weighted average)
        velocity = momentum * velocity - learning_rate * gradient
        
        # Update position
        x = x + velocity
    
    return x

# Momentum helps in narrow valleys
x_momentum = gradient_descent_with_momentum(f, grad_f, [5, 5], 
                                           learning_rate=0.1, momentum=0.9)
print(f"With momentum: {x_momentum}")
```

---

## Applications in ML

### Linear Regression

```python
# Loss function: L = (1/2n) * Σ(y_pred - y_true)²
# y_pred = X @ w + b
# Gradient: ∂L/∂w = (1/n) * X^T @ (y_pred - y_true)

def linear_regression_gradient(X, y, w, b):
    n = len(y)
    y_pred = X @ w + b
    error = y_pred - y
    
    dw = (1/n) * X.T @ error
    db = (1/n) * np.sum(error)
    
    return dw, db

# Gradient descent for linear regression
def train_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    n_features = X.shape[1]
    w = np.random.randn(n_features)
    b = 0
    
    for i in range(iterations):
        dw, db = linear_regression_gradient(X, y, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    return w, b
```

### Neural Network Training

```python
# The entire training process uses gradient descent
# 1. Forward pass: Compute predictions
# 2. Compute loss
# 3. Backward pass: Compute gradients (chain rule)
# 4. Update weights: w = w - lr * gradient

# This is exactly what happens in:
# - model.fit() in Keras
# - Training loop in PyTorch
```

## Computational Calculus in Practice

### Why Computational Calculus Matters

Calculus formulas become intuitive when implemented in code. This section shows how derivatives, gradients, and optimization translate directly to Python code used in machine learning.

### Example 1: Gradient Descent Visualization

**Mathematical Concept**: Gradient points in direction of steepest ascent. Gradient descent: x_new = x_old - learning_rate × gradient

**In Code**:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to minimize: f(x, y) = x² + 2y²
def f(x, y):
    return x**2 + 2*y**2

def gradient(x, y):
    """Compute gradient: [∂f/∂x, ∂f/∂y]"""
    return np.array([2*x, 4*y])

# Gradient descent
x, y = 3.0, 4.0  # Starting point
learning_rate = 0.1
steps = 20

path = [(x, y, f(x, y))]
for i in range(steps):
    grad = gradient(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    path.append((x, y, f(x, y)))

# Visualize
x_path, y_path, z_path = zip(*path)
x_grid = np.linspace(-4, 4, 50)
y_grid = np.linspace(-4, 4, 50)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = f(X_grid, Y_grid)

fig = plt.figure(figsize=(12, 5))

# 2D contour plot
ax1 = fig.add_subplot(121)
ax1.contour(X_grid, Y_grid, Z_grid, levels=20)
ax1.plot(x_path, y_path, 'ro-', label='Gradient Descent Path', markersize=8)
ax1.plot(0, 0, 'g*', markersize=20, label='Minimum (0, 0)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gradient Descent: f(x,y) = x² + 2y²')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3D surface plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.6, cmap='viridis')
ax2.plot(x_path, y_path, z_path, 'r-o', label='Path', markersize=8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')
ax2.set_title('3D View')

plt.tight_layout()
plt.show()

print(f"Final point: ({x:.4f}, {y:.4f})")
print(f"Final value: {f(x, y):.4f}")
print(f"Distance from minimum: {np.sqrt(x**2 + y**2):.4f}")
```

### Example 2: Learning Rate Impact

**Mathematical Concept**: Learning rate controls step size in gradient descent. Too small = slow convergence, too large = overshooting.

**In Code**:
```python
def gradient_descent_with_lr(lr, max_steps=50):
    """Gradient descent with different learning rates"""
    x, y = 3.0, 4.0
    path = [(x, y)]
    
    for i in range(max_steps):
        grad = gradient(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
        path.append((x, y))
        
        # Stop if converged
        if np.sqrt(x**2 + y**2) < 0.01:
            break
    
    return path, len(path)

# Test different learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    path, steps = gradient_descent_with_lr(lr)
    x_path, y_path = zip(*path)
    
    axes[idx].contour(X_grid, Y_grid, Z_grid, levels=20, alpha=0.3)
    axes[idx].plot(x_path, y_path, 'ro-', markersize=6)
    axes[idx].plot(0, 0, 'g*', markersize=15)
    axes[idx].set_title(f'Learning Rate = {lr} ({steps} steps)')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Results show:
# - lr=0.01: Converges slowly (many steps)
# - lr=0.1: Converges well (optimal)
# - lr=0.5: Oscillates but converges
# - lr=1.0: Diverges (overshoots minimum)
```

### Example 3: Chain Rule in Backpropagation

**Mathematical Concept**: Chain rule: ∂L/∂w = (∂L/∂y) × (∂y/∂z) × (∂z/∂w)

**In Code**:
```python
# Simple neural network layer with chain rule
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Forward pass
def forward(X, W, b):
    z = X @ W + b
    a = relu(z)
    return z, a

# Backward pass (chain rule)
def backward(X, W, z, a, dL_da):
    """
    Compute gradients using chain rule:
    dL/dW = dL/da × da/dz × dz/dW
    """
    # da/dz (derivative of ReLU)
    da_dz = relu_derivative(z)
    
    # dL/dz = dL/da × da/dz (chain rule)
    dL_dz = dL_da * da_dz
    
    # dz/dW = X (from z = X @ W + b)
    # dL/dW = dL/dz × dz/dW = dL/dz × X
    dL_dW = X.T @ dL_dz
    
    # dz/db = 1
    # dL/db = dL/dz × dz/db = dL/dz
    dL_db = np.sum(dL_dz, axis=0)
    
    return dL_dW, dL_db

# Example usage
X = np.array([[1, 2], [3, 4]])  # Input (2 samples, 2 features)
W = np.array([[0.5, 0.3], [0.2, 0.8]])  # Weights (2 features -> 2 neurons)
b = np.array([0.1, 0.2])  # Bias

# Forward pass
z, a = forward(X, W, b)

# Loss gradient (from next layer, example)
dL_da = np.array([[0.1, -0.2], [0.3, 0.1]])

# Backward pass (chain rule)
dL_dW, dL_db = backward(X, W, z, a, dL_da)

print("Gradients computed using chain rule:")
print(f"dL/dW:\n{dL_dW}")
print(f"dL/db: {dL_db}")
```

### Example 4: Second Derivatives (Hessian) for Optimization

**Mathematical Concept**: Second derivative (Hessian matrix) tells us about curvature. Helps choose better learning rates.

**In Code**:
```python
from scipy.optimize import minimize
import numpy as np

# Function with known minimum
def f(x):
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

def gradient(x):
    return np.array([2*x[0] + x[1], 4*x[1] + x[0]])

def hessian(x):
    """Hessian matrix (second derivatives)"""
    return np.array([[2, 1], [1, 4]])

# Newton's method (uses Hessian for better convergence)
x0 = np.array([3.0, 4.0])
result = minimize(f, x0, method='Newton-CG', jac=gradient, hess=hessian)

print(f"Minimum found at: {result.x}")
print(f"Function value: {result.fun}")
print(f"Converged in {result.nit} iterations")

# Compare with gradient descent (slower)
x_gd = x0.copy()
for i in range(100):
    grad = gradient(x_gd)
    x_gd -= 0.1 * grad
    if np.linalg.norm(grad) < 1e-6:
        break

print(f"\nGradient descent: {x_gd} (in {i+1} iterations)")
print(f"Newton's method is faster because it uses curvature information")
```

### Key Takeaway

**Theory → Code → Intuition**:
1. Understand the calculus concept (derivative, gradient)
2. Implement it in Python
3. Visualize the results
4. See how it's used in ML algorithms

This builds deeper understanding than formulas alone.

---

## Practice Exercises

### Exercise 1: Compute Derivatives

Find derivatives of:
1. f(x) = x³ + 2x² + x
2. f(x) = e^x * sin(x)
3. f(x) = ln(x² + 1)

### Exercise 2: Gradient Descent

Implement gradient descent to minimize:
- f(x, y) = x² + 2y²
- f(x, y) = (x-1)² + (y-2)²

### Exercise 3: Chain Rule

Compute derivative of f(x) = sin(x² + 1) using chain rule.

---

## Resources

### Books

- **"Calculus"** by Michael Spivak
- **"Calculus Made Easy"** by Silvanus Thompson

### Online Courses

- [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - **Highly recommended!**
- [Khan Academy - Calculus](https://www.khanacademy.org/math/calculus-1)
- [MIT Single Variable Calculus](https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/)

---

## Key Takeaways

1. **Derivatives** measure rate of change
2. **Gradients** point in direction of steepest ascent
3. **Gradient descent** minimizes functions by following negative gradient
4. **Chain rule** enables backpropagation in neural networks
5. **Learning rate** is crucial for convergence

---

**Remember**: Calculus is the tool that makes ML optimization possible. Understanding gradients is key to understanding how models learn!
