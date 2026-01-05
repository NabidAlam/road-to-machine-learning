# Prerequisites Quick Reference

Quick lookup guide for Python, mathematics, and NumPy essentials.

## Table of Contents

- [Python Syntax Quick Reference](#python-syntax-quick-reference)
- [NumPy Operations Cheat Sheet](#numpy-operations-cheat-sheet)
- [Math Formulas Quick Lookup](#math-formulas-quick-lookup)
- [Common Patterns and Idioms](#common-patterns-and-idioms)

---

## Python Syntax Quick Reference

### Data Types

```python
# Numbers
x = 10          # int
y = 3.14        # float
z = 3 + 4j      # complex

# Strings
s = "Hello"
s = 'World'
s = """Multi-line"""

# Boolean
b = True
b = False

# Collections
lst = [1, 2, 3]           # List (mutable)
tup = (1, 2, 3)           # Tuple (immutable)
dct = {'a': 1, 'b': 2}    # Dictionary
st = {1, 2, 3}            # Set
```

### Control Flow

```python
# If-else
if condition:
    do_something()
elif other_condition:
    do_other()
else:
    do_default()

# For loop
for item in iterable:
    process(item)

# While loop
while condition:
    do_something()

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

### Functions

```python
# Basic function
def function_name(param1, param2):
    return result

# Default arguments
def func(x, y=10):
    return x + y

# Variable arguments
def func(*args, **kwargs):
    pass

# Lambda
square = lambda x: x**2
```

### Classes

```python
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def method(self):
        return self.value
```

---

## NumPy Operations Cheat Sheet

### Array Creation

```python
import numpy as np

# From lists
arr = np.array([1, 2, 3])

# Special arrays
zeros = np.zeros((3, 4))
ones = np.ones((3, 4))
full = np.full((3, 4), 5)
eye = np.eye(3)
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 10)
random = np.random.rand(3, 4)
randint = np.random.randint(0, 10, (3, 4))
```

### Array Properties

```python
arr.shape      # Dimensions
arr.size       # Total elements
arr.dtype      # Data type
arr.ndim       # Number of dimensions
arr.itemsize   # Bytes per element
arr.nbytes     # Total bytes
```

### Indexing and Slicing

```python
# Basic indexing
arr[0]           # First element
arr[-1]          # Last element
arr[0, 0]        # 2D indexing

# Slicing
arr[1:5]         # Slice
arr[1:5:2]       # Slice with step
arr[:, 0]        # All rows, first column
arr[0, :]        # First row, all columns

# Boolean indexing
mask = arr > 5
filtered = arr[mask]

# Fancy indexing
indices = [0, 2, 4]
selected = arr[indices]
```

### Array Operations

```python
# Arithmetic
arr + 1          # Add scalar
arr * 2          # Multiply scalar
arr1 + arr2      # Element-wise addition
arr1 * arr2      # Element-wise multiplication
arr1 @ arr2      # Matrix multiplication
np.dot(arr1, arr2)  # Dot product

# Mathematical functions
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.sin(arr)
np.cos(arr)
np.abs(arr)
```

### Array Manipulation

```python
# Reshaping
arr.reshape(3, 4)
arr.flatten()
arr.ravel()

# Stacking
np.vstack([arr1, arr2])    # Vertical
np.hstack([arr1, arr2])    # Horizontal
np.concatenate([arr1, arr2], axis=0)

# Splitting
np.split(arr, 3)
np.vsplit(arr, 3)
np.hsplit(arr, 3)
```

### Reductions

```python
np.sum(arr)              # Sum all
np.sum(arr, axis=0)      # Sum along axis
np.mean(arr)             # Mean
np.std(arr)              # Standard deviation
np.min(arr)              # Minimum
np.max(arr)              # Maximum
np.argmin(arr)           # Index of minimum
np.argmax(arr)           # Index of maximum
np.cumsum(arr)           # Cumulative sum
np.cumprod(arr)          # Cumulative product
```

### Linear Algebra

```python
np.dot(A, B)             # Matrix multiplication
np.transpose(A)          # Transpose
A.T                      # Transpose (shorthand)
np.linalg.inv(A)         # Inverse
np.linalg.det(A)         # Determinant
np.linalg.eig(A)         # Eigenvalues/eigenvectors
np.linalg.svd(A)         # SVD
np.linalg.norm(A)        # Norm
np.linalg.solve(A, b)    # Solve linear system
```

---

## Math Formulas Quick Lookup

### Linear Algebra

**Vector Operations:**
- Dot product: `a · b = Σ(a_i * b_i)`
- Norm: `||a|| = √(Σ(a_i²))`
- Cosine similarity: `cos(θ) = (a · b) / (||a|| * ||b||)`

**Matrix Operations:**
- Matrix multiplication: `C = A @ B` where `C_ij = Σ(A_ik * B_kj)`
- Transpose: `(A^T)_ij = A_ji`
- Inverse: `A^(-1) * A = I`

**Eigenvalues/Eigenvectors:**
- `Av = λv` where λ is eigenvalue, v is eigenvector

### Statistics

**Descriptive Statistics:**
- Mean: `μ = (1/n) * Σ(x_i)`
- Variance: `σ² = (1/n) * Σ(x_i - μ)²`
- Standard deviation: `σ = √(σ²)`
- Covariance: `Cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]`
- Correlation: `ρ = Cov(X,Y) / (σ_X * σ_Y)`

**Probability:**
- Conditional: `P(A|B) = P(A∩B) / P(B)`
- Bayes' theorem: `P(A|B) = P(B|A) * P(A) / P(B)`

### Calculus

**Derivatives:**
- Power rule: `d/dx(x^n) = n*x^(n-1)`
- Product rule: `d/dx(f*g) = f'*g + f*g'`
- Chain rule: `d/dx(f(g(x))) = f'(g(x)) * g'(x)`

**Gradient:**
- `∇f = [∂f/∂x, ∂f/∂y, ...]`

**Gradient Descent:**
- `θ_new = θ_old - α * ∇J(θ)`
- where α is learning rate, J is cost function

---

## Common Patterns and Idioms

### NumPy Patterns

```python
# Normalize array
normalized = (arr - arr.min()) / (arr.max() - arr.min())
# or
normalized = (arr - arr.mean()) / arr.std()

# One-hot encoding
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

# Batch processing
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    process(batch)

# Broadcasting example
arr = np.random.rand(100, 10)
mean = np.mean(arr, axis=0)
centered = arr - mean  # Broadcasting
```

### Python Patterns

```python
# List operations
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]

# Dictionary operations
dct = {k: v for k, v in zip(keys, values)}
inverted = {v: k for k, v in dct.items()}

# Error handling
try:
    result = risky_operation()
except ValueError as e:
    handle_error(e)
finally:
    cleanup()

# Context manager
with open('file.txt', 'r') as f:
    content = f.read()
```

### Performance Patterns

```python
# Vectorization (fast)
result = np.sum(arr**2)

# vs Loop (slow)
result = 0
for x in arr:
    result += x**2

# Pre-allocate arrays
result = np.zeros(1000)
for i in range(1000):
    result[i] = compute(i)

# Use views not copies
view = arr[::2]      # View (fast)
copy = arr[::2].copy()  # Copy (slow)
```

---

## Quick Tips

### NumPy Tips

1. **Always use vectorization** - Avoid Python loops
2. **Use appropriate dtypes** - int32 vs int64, float32 vs float64
3. **Pre-allocate arrays** - Don't append in loops
4. **Use views when possible** - Avoid unnecessary copies
5. **Broadcasting is powerful** - Learn to use it effectively

### Python Tips

1. **List comprehensions** - More Pythonic than loops
2. **Use generators** - For memory-efficient iteration
3. **Context managers** - For resource management
4. **Type hints** - For better code documentation
5. **Docstrings** - Document your functions

### Math Tips

1. **Understand the intuition** - Not just formulas
2. **Visualize** - Use plots to understand concepts
3. **Practice with code** - Implement formulas yourself
4. **Connect to ML** - See how math applies to ML

---

## Common Errors and Solutions

### NumPy Errors

```python
# Error: shapes not aligned
# Solution: Check dimensions
arr1.shape  # Check shape
arr2.shape  # Check shape
result = arr1 @ arr2  # Matrix multiplication

# Error: broadcasting failed
# Solution: Reshape arrays
arr1 = arr1.reshape(-1, 1)  # Add dimension
result = arr1 + arr2

# Error: memory error
# Solution: Use generators, process in chunks
for chunk in process_in_chunks(data, chunk_size=1000):
    process(chunk)
```

### Python Errors

```python
# Error: IndexError
# Solution: Check bounds
if 0 <= index < len(arr):
    value = arr[index]

# Error: KeyError
# Solution: Check key exists
if key in dictionary:
    value = dictionary[key]

# Error: AttributeError
# Solution: Check object type
if hasattr(obj, 'method'):
    obj.method()
```

---

**Remember**: This is a quick reference. For detailed explanations, see the main guides!

