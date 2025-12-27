# NumPy Complete Guide

Comprehensive guide to NumPy - the foundation of numerical computing in Python and essential for machine learning.

## Table of Contents

- [Introduction](#introduction)
- [Creating Arrays](#creating-arrays)
- [Array Operations](#array-operations)
- [Indexing and Slicing](#indexing-and-slicing)
- [Broadcasting](#broadcasting)
- [Mathematical Operations](#mathematical-operations)
- [Linear Algebra Operations](#linear-algebra-operations)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is NumPy?

NumPy (Numerical Python) is a library for numerical computing. It provides:
- **N-dimensional arrays** (ndarray) - faster than Python lists
- **Mathematical functions** - optimized for arrays
- **Linear algebra operations** - essential for ML

### Why NumPy?

- **Speed**: 10-100x faster than Python lists
- **Memory efficient**: Less memory than Python lists
- **Foundation**: Most ML libraries (Pandas, Scikit-learn, TensorFlow) built on NumPy
- **Vectorization**: Perform operations on entire arrays at once

### Installation

```python
pip install numpy
```

```python
import numpy as np
print(np.__version__)  # Check version
```

---

## Creating Arrays

### From Lists

```python
import numpy as np

# 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)
# Output: [1 2 3 4 5]

# 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Output:
# [[1 2 3]
#  [4 5 6]]

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3d.shape)  # Output: (2, 2, 2)
```

### Built-in Array Creation Functions

```python
# Zeros
zeros = np.zeros((3, 4))
print(zeros)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Ones
ones = np.ones((2, 3))
print(ones)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Full (fill with specific value)
full = np.full((2, 2), 7)
print(full)
# Output:
# [[7 7]
#  [7 7]]

# Identity matrix
identity = np.eye(3)
print(identity)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Range
range_arr = np.arange(0, 10, 2)
print(range_arr)  # Output: [0 2 4 6 8]

# Linspace (evenly spaced)
linspace_arr = np.linspace(0, 1, 5)
print(linspace_arr)  # Output: [0.   0.25 0.5  0.75 1.  ]

# Random
random_arr = np.random.rand(3, 3)  # Uniform [0, 1)
print(random_arr)

random_int = np.random.randint(0, 10, (3, 3))  # Random integers
print(random_int)

# Normal distribution
normal = np.random.normal(0, 1, (3, 3))  # Mean=0, Std=1
print(normal)
```

### Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)      # Output: (2, 3) - dimensions
print(arr.size)       # Output: 6 - total elements
print(arr.ndim)       # Output: 2 - number of dimensions
print(arr.dtype)      # Output: int64 - data type
print(arr.itemsize)   # Output: 8 - bytes per element
```

---

## Array Operations

### Arithmetic Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(a + b)   # Output: [ 6  8 10 12]
print(a - b)   # Output: [-4 -4 -4 -4]
print(a * b)   # Output: [ 5 12 21 32]
print(a / b)   # Output: [0.2 0.333... 0.428... 0.5]
print(a ** 2)  # Output: [ 1  4  9 16] - squared

# Scalar operations
print(a + 10)  # Output: [11 12 13 14]
print(a * 2)   # Output: [2 4 6 8]
```

### Comparison Operations

```python
a = np.array([1, 2, 3, 4, 5])

print(a > 3)        # Output: [False False False  True  True]
print(a == 3)       # Output: [False False  True False False]
print(a != 3)       # Output: [ True  True False  True  True]

# Boolean indexing
print(a[a > 3])     # Output: [4 5]
print(a[(a > 2) & (a < 5)])  # Output: [3 4]
```

### Aggregate Functions

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr.sum())      # Output: 15
print(arr.mean())     # Output: 3.0
print(arr.std())      # Output: 1.414... (standard deviation)
print(arr.var())      # Output: 2.0 (variance)
print(arr.min())      # Output: 1
print(arr.max())      # Output: 5
print(arr.argmin())   # Output: 0 (index of minimum)
print(arr.argmax())   # Output: 4 (index of maximum)

# For 2D arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d.sum(axis=0))  # Output: [5 7 9] (sum along columns)
print(arr2d.sum(axis=1))  # Output: [6 15] (sum along rows)
```

---

## Indexing and Slicing

### 1D Arrays

```python
arr = np.array([10, 20, 30, 40, 50])

# Indexing
print(arr[0])    # Output: 10 (first element)
print(arr[-1])   # Output: 50 (last element)

# Slicing [start:stop:step]
print(arr[1:4])      # Output: [20 30 40]
print(arr[:3])       # Output: [10 20 30] (first 3)
print(arr[2:])       # Output: [30 40 50] (from index 2)
print(arr[::2])      # Output: [10 30 50] (every 2nd element)
print(arr[::-1])     # Output: [50 40 30 20 10] (reverse)
```

### 2D Arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
print(arr[0, 0])     # Output: 1 (row 0, col 0)
print(arr[1, 2])     # Output: 6 (row 1, col 2)

# Slicing
print(arr[0, :])     # Output: [1 2 3] (first row, all columns)
print(arr[:, 1])     # Output: [2 5 8] (all rows, column 1)
print(arr[0:2, 1:3]) # Output: [[2 3] [5 6]] (submatrix)

# Fancy indexing
print(arr[[0, 2]])   # Output: [[1 2 3] [7 8 9]] (rows 0 and 2)
print(arr[:, [0, 2]]) # Output: [[1 3] [4 6] [7 9]] (columns 0 and 2)
```

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Select elements based on condition
mask = arr > 5
print(mask)          # Output: [False False False False False  True  True  True  True  True]
print(arr[mask])     # Output: [ 6  7  8  9 10]

# Multiple conditions
mask = (arr > 3) & (arr < 8)
print(arr[mask])     # Output: [4 5 6 7]
```

---

## Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes.

### Rules

1. Arrays are aligned from the right
2. Dimensions must match or be 1
3. Missing dimensions are treated as 1

### Examples

```python
# Scalar broadcasting
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10
print(result)
# Output:
# [[11 12 13]
#  [14 15 16]]

# Row vector broadcasting
arr = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
result = arr + row
print(result)
# Output:
# [[11 22 33]
#  [14 25 36]]

# Column vector broadcasting
arr = np.array([[1, 2, 3], [4, 5, 6]])
col = np.array([[10], [20]])
result = arr + col
print(result)
# Output:
# [[11 12 13]
#  [24 25 26]]
```

---

## Mathematical Operations

### Universal Functions (ufuncs)

```python
arr = np.array([1, 2, 3, 4])

# Trigonometric
print(np.sin(arr))    # Sine
print(np.cos(arr))    # Cosine
print(np.tan(arr))    # Tangent

# Exponential and logarithmic
print(np.exp(arr))    # e^x
print(np.log(arr))    # Natural log
print(np.log10(arr))  # Base 10 log
print(np.power(arr, 2))  # x^2

# Rounding
arr_float = np.array([1.7, 2.3, 3.8, 4.1])
print(np.round(arr_float))    # Output: [2. 2. 4. 4.]
print(np.floor(arr_float))    # Output: [1. 2. 3. 4.]
print(np.ceil(arr_float))     # Output: [2. 3. 4. 5.]

# Absolute value
arr_neg = np.array([-1, -2, 3, -4])
print(np.abs(arr_neg))        # Output: [1 2 3 4]
```

### Statistical Functions

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(np.mean(arr))       # Mean: 5.5
print(np.median(arr))     # Median: 5.5
print(np.std(arr))        # Standard deviation
print(np.var(arr))        # Variance
print(np.percentile(arr, 50))  # 50th percentile (median)
print(np.percentile(arr, [25, 50, 75]))  # Quartiles
```

---

## Linear Algebra Operations

### Matrix Operations

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)  # or A @ B
print(C)
# Output:
# [[19 22]
#  [43 50]]

# Matrix transpose
A_T = A.T
print(A_T)
# Output:
# [[1 3]
#  [2 4]]

# Matrix inverse
A_inv = np.linalg.inv(A)
print(A_inv)
# Output:
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Determinant
det = np.linalg.det(A)
print(det)  # Output: -2.0
```

### Eigenvalues and Eigenvectors

```python
A = np.array([[1, 2], [2, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(x)  # Output: [2. 3.]
# Verification: A @ x should equal b
print(A @ x)  # Output: [9. 8.]
```

### Vector Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(a, b)
print(dot_product)  # Output: 32 (1*4 + 2*5 + 3*6)

# Cross product (3D only)
cross_product = np.cross(a, b)
print(cross_product)  # Output: [-3  6 -3]

# Vector norm
norm = np.linalg.norm(a)
print(norm)  # Output: 3.741... (√(1² + 2² + 3²))
```

---

## Practice Exercises

### Exercise 1: Array Creation and Manipulation

**Task:** Create a 5x5 array filled with random integers between 1 and 100, then find the mean, max, and min.

**Solution:**
```python
arr = np.random.randint(1, 101, (5, 5))
print("Array:\n", arr)
print("Mean:", arr.mean())
print("Max:", arr.max())
print("Min:", arr.min())
```

### Exercise 2: Boolean Indexing

**Task:** Create an array of numbers 1-20, and extract all even numbers.

**Solution:**
```python
arr = np.arange(1, 21)
evens = arr[arr % 2 == 0]
print(evens)  # Output: [ 2  4  6  8 10 12 14 16 18 20]
```

### Exercise 3: Matrix Operations

**Task:** Create two 3x3 matrices and compute their product, then find the determinant of the result.

**Solution:**
```python
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)

C = A @ B
det = np.linalg.det(C)
print("Determinant:", det)
```

### Exercise 4: Statistical Analysis

**Task:** Generate 1000 random numbers from a normal distribution (mean=0, std=1) and calculate statistics.

**Solution:**
```python
data = np.random.normal(0, 1, 1000)
print("Mean:", np.mean(data))
print("Std:", np.std(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("25th percentile:", np.percentile(data, 25))
print("75th percentile:", np.percentile(data, 75))
```

### Exercise 5: Reshaping and Broadcasting

**Task:** Create a 1D array of numbers 1-12, reshape it to 3x4, then add a row vector [10, 20, 30, 40] to each row.

**Solution:**
```python
arr = np.arange(1, 13).reshape(3, 4)
row = np.array([10, 20, 30, 40])
result = arr + row
print(result)
```

---

## Key Takeaways

1. **NumPy arrays are fast** - Use instead of Python lists for numerical data
2. **Vectorization** - Operations on entire arrays at once
3. **Broadcasting** - Operations on arrays of different shapes
4. **Linear algebra** - Built-in functions for matrix operations
5. **Indexing** - Powerful slicing and boolean indexing

---

## Common Patterns

### Pattern 1: Data Normalization

```python
# Normalize to [0, 1]
data = np.array([10, 20, 30, 40, 50])
normalized = (data - data.min()) / (data.max() - data.min())
print(normalized)  # Output: [0.   0.25 0.5  0.75 1.  ]

# Standardize (mean=0, std=1)
standardized = (data - data.mean()) / data.std()
print(standardized)
```

### Pattern 2: Finding Indices

```python
arr = np.array([1, 5, 3, 9, 2, 7])

# Find index of maximum
max_idx = np.argmax(arr)
print(f"Max value {arr[max_idx]} at index {max_idx}")

# Find indices where condition is true
indices = np.where(arr > 5)
print(indices)  # Output: (array([3, 5]),)
```

### Pattern 3: Reshaping

```python
arr = np.arange(12)
print(arr)  # Output: [ 0  1  2  3  4  5  6  7  8  9 10 11]

reshaped = arr.reshape(3, 4)
print(reshaped)
# Output:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Flatten
flattened = reshaped.flatten()
print(flattened)  # Back to 1D
```

---

## Next Steps

- Practice with NumPy arrays daily
- Work through the exercises
- Move to [02-pandas.md](02-pandas.md) to learn data manipulation

**Remember**: NumPy is the foundation - master it well!

