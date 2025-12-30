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
