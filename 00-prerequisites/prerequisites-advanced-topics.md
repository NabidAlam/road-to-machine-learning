# Advanced Prerequisites Topics

Comprehensive guide to advanced Python, mathematics, and optimization techniques needed for advanced machine learning.

## Table of Contents

- [Advanced Python Concepts](#advanced-python-concepts)
- [Advanced NumPy Operations](#advanced-numpy-operations)
- [Advanced Mathematical Concepts](#advanced-mathematical-concepts)
- [Performance Optimization](#performance-optimization)
- [Memory Management](#memory-management)
- [Advanced Statistics](#advanced-statistics)

---

## Advanced Python Concepts

### Decorators

Decorators allow you to modify or extend functions without changing their code.

```python
# Basic decorator
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"

# Class decorator
def add_method(cls):
    def new_method(self):
        return "New method added"
    cls.new_method = new_method
    return cls

@add_method
class MyClass:
    pass
```

### Context Managers

Context managers ensure proper resource management.

```python
# Using context managers
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed

# Custom context manager
class Timer:
    def __init__(self):
        self.start = None
    
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        print(f"Elapsed time: {time.time() - self.start:.4f} seconds")

with Timer():
    # Your code here
    time.sleep(1)
```

### Generators and Generator Expressions

Memory-efficient iteration for large datasets.

```python
# Generator function
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Generator expression
squares = (x**2 for x in range(1000000))  # Memory efficient
# vs
squares_list = [x**2 for x in range(1000000)]  # Uses memory

# Using generators
for num in fibonacci_generator(10):
    print(num)
```

### Metaclasses (Advanced)

Metaclasses allow you to customize class creation.

```python
# Simple metaclass example
class Meta(type):
    def __new__(cls, name, bases, dct):
        # Add a class attribute
        dct['created_by_meta'] = True
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

print(MyClass.created_by_meta)  # True
```

### Functional Programming

Python supports functional programming patterns.

```python
from functools import reduce, partial

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
sum_all = reduce(lambda x, y: x + y, numbers)

# Partial functions
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # 10
```

---

## Advanced NumPy Operations

### Advanced Array Manipulation

```python
import numpy as np

# Reshaping and stacking
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
stacked = np.vstack([arr, arr])  # Vertical stack
hstacked = np.hstack([arr, arr])  # Horizontal stack

# Advanced indexing
arr = np.arange(12).reshape(3, 4)
# Boolean indexing
mask = arr > 5
filtered = arr[mask]

# Fancy indexing
indices = [0, 2]
selected = arr[indices]

# Multi-dimensional indexing
arr[0:2, 1:3]  # Slice multiple dimensions
```

### Broadcasting Advanced

```python
# Broadcasting rules
arr1 = np.arange(12).reshape(3, 4)
arr2 = np.arange(4)
result = arr1 + arr2  # Broadcasting

# Outer product
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5])
outer = np.outer(arr1, arr2)

# Einsum for complex operations
arr1 = np.random.rand(3, 4)
arr2 = np.random.rand(4, 5)
result = np.einsum('ij,jk->ik', arr1, arr2)  # Matrix multiplication
```

### Advanced Linear Algebra

```python
# Matrix decompositions
A = np.random.rand(5, 5)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, s, Vt = np.linalg.svd(A)

# QR decomposition
Q, R = np.linalg.qr(A)

# Cholesky decomposition (for positive definite)
A_pos = A @ A.T  # Make positive definite
L = np.linalg.cholesky(A_pos)
```

### Advanced Array Operations

```python
# Advanced reductions
arr = np.random.rand(100, 50)

# Along specific axis
mean_axis0 = np.mean(arr, axis=0)  # Mean along rows
mean_axis1 = np.mean(arr, axis=1)  # Mean along columns

# Cumulative operations
cumsum = np.cumsum(arr, axis=0)
cumprod = np.cumprod(arr, axis=1)

# Advanced sorting
arr = np.random.rand(10)
sorted_indices = np.argsort(arr)
sorted_arr = np.sort(arr)
```

---

## Advanced Mathematical Concepts

### Tensors

Tensors are multi-dimensional arrays, fundamental in deep learning.

```python
import numpy as np

# Scalars (0D tensor)
scalar = np.array(5)

# Vectors (1D tensor)
vector = np.array([1, 2, 3])

# Matrices (2D tensor)
matrix = np.array([[1, 2], [3, 4]])

# 3D tensor
tensor_3d = np.random.rand(3, 4, 5)

# Tensor operations
# Element-wise operations
result = tensor_3d + 1

# Reduction operations
sum_all = np.sum(tensor_3d)
sum_axis = np.sum(tensor_3d, axis=0)
```

### Advanced Linear Algebra

```python
# Matrix norms
A = np.random.rand(5, 5)
frobenius_norm = np.linalg.norm(A, 'fro')
spectral_norm = np.linalg.norm(A, 2)

# Matrix rank
rank = np.linalg.matrix_rank(A)

# Condition number
cond = np.linalg.cond(A)

# Pseudo-inverse
A_inv = np.linalg.pinv(A)

# Solving linear systems
b = np.random.rand(5)
x = np.linalg.solve(A, b)
```

### Advanced Calculus

```python
# Numerical differentiation
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Numerical integration
from scipy import integrate

def f(x):
    return x**2

result = integrate.quad(f, 0, 1)

# Gradient computation
def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (f(x_plus) - f(x)) / h
    return grad
```

---

## Performance Optimization

### Vectorization

```python
import numpy as np
import time

# Slow: Python loops
def slow_sum(arr):
    result = 0
    for x in arr:
        result += x
    return result

# Fast: NumPy vectorization
def fast_sum(arr):
    return np.sum(arr)

# Benchmark
arr = np.random.rand(1000000)
start = time.time()
slow_sum(arr)
print(f"Slow: {time.time() - start:.4f}s")

start = time.time()
fast_sum(arr)
print(f"Fast: {time.time() - start:.4f}s")
```

### NumPy Optimization

```python
# Use in-place operations
arr = np.random.rand(1000)
arr += 1  # In-place (faster)
# vs
arr = arr + 1  # Creates new array (slower)

# Pre-allocate arrays
result = np.zeros(1000)  # Pre-allocate
for i in range(1000):
    result[i] = i**2

# Use views instead of copies
arr = np.random.rand(1000, 1000)
view = arr[::2, ::2]  # View (no copy)
copy = arr[::2, ::2].copy()  # Copy (slower)
```

### Parallel Processing

```python
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    return np.sum(chunk**2)

# Parallel processing
def parallel_sum_squares(arr, n_processes=4):
    chunks = np.array_split(arr, n_processes)
    with Pool(n_processes) as pool:
        results = pool.map(process_chunk, chunks)
    return sum(results)

# Usage
arr = np.random.rand(1000000)
result = parallel_sum_squares(arr)
```

---

## Memory Management

### Understanding Memory Usage

```python
import sys
import numpy as np

# Check memory usage
arr = np.random.rand(1000, 1000)
print(f"Memory: {arr.nbytes / 1024**2:.2f} MB")

# Different dtypes use different memory
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
arr_int64 = np.array([1, 2, 3], dtype=np.int64)
print(f"int32: {arr_int32.nbytes} bytes")
print(f"int64: {arr_int64.nbytes} bytes")
```

### Memory-Efficient Operations

```python
# Use generators for large datasets
def large_data_generator(n):
    for i in range(n):
        yield np.random.rand(1000)

# Process in chunks
def process_large_file(filename, chunk_size=10000):
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process chunk
        yield process_chunk(chunk)

# Delete large objects
large_array = np.random.rand(100000, 1000)
# ... use it ...
del large_array  # Free memory
import gc
gc.collect()  # Force garbage collection
```

---

## Advanced Statistics

### Bayesian Inference

```python
from scipy import stats

# Bayesian updating
prior = stats.beta(2, 2)  # Prior distribution
# After observing data
posterior = stats.beta(2 + 10, 2 + 5)  # Updated distribution

# Sampling from posterior
samples = posterior.rvs(1000)
```

### Advanced Distributions

```python
from scipy import stats

# Multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
mvn = stats.multivariate_normal(mean, cov)
samples = mvn.rvs(1000)

# Student's t-distribution
t_dist = stats.t(df=10)
samples = t_dist.rvs(1000)

# Chi-square distribution
chi2 = stats.chi2(df=5)
samples = chi2.rvs(1000)
```

### Hypothesis Testing Advanced

```python
from scipy import stats

# ANOVA (Analysis of Variance)
group1 = np.random.normal(5, 1, 30)
group2 = np.random.normal(6, 1, 30)
group3 = np.random.normal(7, 1, 30)

f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Chi-square test
observed = np.array([10, 20, 30])
expected = np.array([20, 20, 20])
chi2, p = stats.chisquare(observed, expected)

# Kolmogorov-Smirnov test
data = np.random.normal(0, 1, 100)
ks_stat, p_value = stats.kstest(data, 'norm')
```

### Monte Carlo Methods

```python
import numpy as np

# Monte Carlo integration
def monte_carlo_integration(f, a, b, n_samples=10000):
    x = np.random.uniform(a, b, n_samples)
    y = f(x)
    return (b - a) * np.mean(y)

# Example: integrate x^2 from 0 to 1
result = monte_carlo_integration(lambda x: x**2, 0, 1)
print(f"Monte Carlo: {result:.4f}")
print(f"Exact: {1/3:.4f}")

# Monte Carlo simulation
def estimate_pi(n_samples=1000000):
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside = np.sum(x**2 + y**2 <= 1)
    return 4 * inside / n_samples

pi_estimate = estimate_pi()
print(f"Pi estimate: {pi_estimate:.4f}")
```

---

## Key Takeaways

1. **Advanced Python**: Decorators, context managers, generators make code more efficient and elegant
2. **Advanced NumPy**: Master broadcasting, advanced indexing, and linear algebra operations
3. **Performance**: Vectorization and parallel processing are crucial for large datasets
4. **Memory**: Understand memory usage and use efficient data structures
5. **Advanced Math**: Tensors, advanced linear algebra, and statistical methods are essential for ML

---

**Remember**: These advanced topics will be used throughout your ML journey. Master them now to make everything easier later!

