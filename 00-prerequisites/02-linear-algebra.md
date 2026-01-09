# Linear Algebra for Machine Learning - Deep Dive

Comprehensive guide to linear algebra concepts essential for understanding machine learning algorithms. This is the mathematical foundation of ML.

## Table of Contents

- [Introduction](#introduction)
- [Tensors](#tensors)
- [Vectors](#vectors)
- [Matrices](#matrices)
- [Matrix Operations](#matrix-operations)
- [Solving Systems of Linear Equations](#solving-systems-of-linear-equations)
  - [Gaussian Elimination (Row Reduction)](#gaussian-elimination-row-reduction)
  - [Solving Ax = 0 (Homogeneous System)](#solving-ax--0-homogeneous-system)
  - [Solving Ax = b (Non-homogeneous System)](#solving-ax--b-non-homogeneous-system)
- [Column Space and Nullspace](#column-space-and-nullspace)
  - [Column Space (Range)](#column-space-range)
  - [Nullspace (Kernel)](#nullspace-kernel)
- [The Four Fundamental Subspaces](#the-four-fundamental-subspaces)
- [Linear Transformations](#linear-transformations)
- [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
- [Determinants](#determinants)
  - [What is a Determinant?](#what-is-a-determinant)
  - [Computing Determinants](#computing-determinants)
  - [Properties of Determinants](#properties-of-determinants)
  - [Cofactor Expansion](#cofactor-expansion)
  - [Cramer's Rule](#cramers-rule)
- [Diagonalization](#diagonalization)
  - [What is Diagonalization?](#what-is-diagonalization)
  - [Diagonalization Process](#diagonalization-process)
  - [Powers of A](#powers-of-a)
- [Matrix Exponentials](#matrix-exponentials)
  - [What is exp(At)?](#what-is-expat)
  - [Computing Matrix Exponential](#computing-matrix-exponential)
  - [Solving Differential Equations](#solving-differential-equations)
- [Matrix Decompositions](#matrix-decompositions)
  - [Eigendecomposition](#1-eigendecomposition)
  - [Singular Value Decomposition (SVD)](#2-singular-value-decomposition-svd)
  - [QR Decomposition](#3-qr-decomposition)
- [Pseudoinverse (Moore-Penrose Inverse)](#pseudoinverse-moore-penrose-inverse)
  - [What is a Pseudoinverse?](#what-is-a-pseudoinverse)
  - [Computing Pseudoinverse](#computing-pseudoinverse)
  - [Left and Right Inverses](#left-and-right-inverses)
- [Positive Definite Matrices](#positive-definite-matrices)
  - [What are Positive Definite Matrices?](#what-are-positive-definite-matrices)
  - [Properties and Applications](#properties-and-applications)
  - [Positive Semidefinite Matrices](#positive-semidefinite-matrices)
  - [Cholesky Decomposition](#cholesky-decomposition)
- [Complex Matrices and Fast Fourier Transform](#complex-matrices-and-fast-fourier-transform)
  - [Complex Matrices](#complex-matrices)
  - [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
  - [FFT in Machine Learning](#fft-in-machine-learning)
- [Computational Math in Practice](#computational-math-in-practice)
- [Applications in ML](#applications-in-ml)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Linear Algebra is Critical for ML

Linear algebra is the **language of machine learning**. Almost every ML algorithm uses linear algebra:

- **Neural Networks**: Matrix multiplications for forward/backward propagation
- **PCA**: Dimensionality reduction using eigenvectors
- **SVM**: Support vectors and hyperplanes
- **Linear Regression**: Solving systems of equations
- **Recommendation Systems**: Matrix factorization
- **Image Processing**: Convolutions as matrix operations

### What You'll Learn

- Vector operations and properties
- Matrix operations and their meanings
- Linear transformations and their geometric intuition
- Eigenvalues/eigenvectors and their importance
- Matrix decompositions (SVD, eigendecomposition)
- How these concepts apply to ML algorithms

---

## Tensors

### What are Tensors?

A **tensor** is a generalization of scalars, vectors, and matrices to higher dimensions. In machine learning, tensors are fundamental data structures used in deep learning frameworks like TensorFlow and PyTorch.

### Tensor Dimensions

**0D Tensor (Scalar)**
- A single number
- Shape: `()`
- Example: `5`, `3.14`

```python
import numpy as np

# 0D tensor (scalar)
scalar = np.array(5)
print(f"Scalar: {scalar}")
print(f"Shape: {scalar.shape}")  # ()
print(f"Dimensions: {scalar.ndim}")  # 0
```

**1D Tensor (Vector)**
- An array of numbers
- Shape: `(n,)`
- Example: `[1, 2, 3, 4]`

```python
# 1D tensor (vector)
vector = np.array([1, 2, 3, 4])
print(f"Vector: {vector}")
print(f"Shape: {vector.shape}")  # (4,)
print(f"Dimensions: {vector.ndim}")  # 1
```

**2D Tensor (Matrix)**
- A 2D array of numbers
- Shape: `(m, n)`
- Example: `[[1, 2], [3, 4]]`

```python
# 2D tensor (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (2, 3)
print(f"Dimensions: {matrix.ndim}")  # 2
```

**3D Tensor**
- A 3D array
- Shape: `(d, m, n)`
- Example: Batch of images, RGB images

```python
# 3D tensor (e.g., batch of 2D images)
tensor_3d = np.array([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])
print(f"3D Tensor:\n{tensor_3d}")
print(f"Shape: {tensor_3d.shape}")  # (2, 2, 2)
print(f"Dimensions: {tensor_3d.ndim}")  # 3
```

**4D Tensor**
- Common in deep learning for batches of images
- Shape: `(batch, height, width, channels)`
- Example: Batch of RGB images: `(32, 224, 224, 3)`

```python
# 4D tensor (e.g., batch of RGB images)
# Shape: (batch_size, height, width, channels)
batch_images = np.random.rand(32, 224, 224, 3)
print(f"4D Tensor Shape: {batch_images.shape}")  # (32, 224, 224, 3)
print(f"Dimensions: {batch_images.ndim}")  # 4
print(f"Total elements: {batch_images.size}")  # 32 * 224 * 224 * 3
```

**5D Tensor**
- Used for video data or 3D medical images
- Shape: `(batch, time, height, width, channels)`

```python
# 5D tensor (e.g., video frames)
# Shape: (batch, frames, height, width, channels)
video = np.random.rand(8, 16, 224, 224, 3)
print(f"5D Tensor Shape: {video.shape}")  # (8, 16, 224, 224, 3)
print(f"Dimensions: {video.ndim}")  # 5
```

### Tensor Properties

**Rank (Order)**
- The number of dimensions (axes) of a tensor
- Scalar: rank 0, Vector: rank 1, Matrix: rank 2, etc.

```python
tensors = [
    np.array(5),                    # Rank 0
    np.array([1, 2, 3]),            # Rank 1
    np.array([[1, 2], [3, 4]]),    # Rank 2
    np.random.rand(2, 3, 4)         # Rank 3
]

for i, t in enumerate(tensors):
    print(f"Tensor {i}: Rank = {t.ndim}, Shape = {t.shape}")
```

**Axes (Dimensions)**
- Each dimension is an axis
- Axis 0: first dimension, Axis 1: second dimension, etc.

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Sum along axis 0 (rows)
print(f"Sum along axis 0 (columns): {matrix.sum(axis=0)}")  # [5 7 9]

# Sum along axis 1 (columns)
print(f"Sum along axis 1 (rows): {matrix.sum(axis=1)}")  # [6 15]
```

**Shape**
- Tuple describing the size of each dimension
- `(batch_size, height, width, channels)` for images

```python
# Common tensor shapes in ML
examples = {
    'scalar': np.array(5).shape,                    # ()
    'vector': np.array([1, 2, 3]).shape,            # (3,)
    'matrix': np.array([[1, 2], [3, 4]]).shape,     # (2, 2)
    'image': np.random.rand(224, 224, 3).shape,    # (224, 224, 3)
    'batch_images': np.random.rand(32, 224, 224, 3).shape  # (32, 224, 224, 3)
}

for name, shape in examples.items():
    print(f"{name}: {shape}")
```

### Tensor Operations

**Reshaping**
```python
# Reshape tensor
vector = np.array([1, 2, 3, 4, 5, 6])
matrix = vector.reshape(2, 3)
print(f"Reshaped to matrix:\n{matrix}")

# Flatten
flattened = matrix.flatten()
print(f"Flattened: {flattened}")
```

**Broadcasting**
```python
# Broadcasting: operations between tensors of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])

# Broadcasting: vector is added to each row
result = matrix + vector
print(f"Broadcasting result:\n{result}")
```

### Tensors in Machine Learning

**Neural Networks**
- Input: `(batch_size, features)` - 2D tensor
- Weights: `(input_features, output_features)` - 2D tensor
- Output: `(batch_size, output_features)` - 2D tensor

**Convolutional Neural Networks**
- Input images: `(batch, height, width, channels)` - 4D tensor
- Convolution filters: `(filter_height, filter_width, in_channels, out_channels)` - 4D tensor

**Recurrent Neural Networks**
- Sequences: `(batch, time_steps, features)` - 3D tensor

```python
# Example: Neural network forward pass
batch_size = 32
input_features = 784
hidden_units = 128
output_units = 10

# Input tensor: (32, 784)
X = np.random.rand(batch_size, input_features)

# Weight tensor: (784, 128)
W1 = np.random.rand(input_features, hidden_units)

# Output tensor: (32, 128)
hidden = X @ W1  # Matrix multiplication

print(f"Input shape: {X.shape}")
print(f"Weight shape: {W1.shape}")
print(f"Output shape: {hidden.shape}")
```

### Key Takeaways

1. **Scalars** (0D): Single numbers
2. **Vectors** (1D): Arrays of numbers
3. **Matrices** (2D): 2D arrays
4. **Higher-order tensors** (3D+): Multi-dimensional arrays
5. **Rank**: Number of dimensions
6. **Shape**: Size of each dimension
7. **Axes**: Each dimension is an axis
8. **Essential in Deep Learning**: All data in neural networks are tensors

---

## Vectors

### What is a Vector?

A **vector** is an ordered list of numbers. In ML:
- **Feature vector**: One data point (e.g., [age, height, weight])
- **Weight vector**: Model parameters
- **Gradient vector**: Direction of steepest ascent

### Vector Representation

```python
import numpy as np
import matplotlib.pyplot as plt

# Column vector (default in NumPy)
v = np.array([3, 4])
print(f"Vector v: {v}")
print(f"Shape: {v.shape}")  # (2,)
print(f"Dimension: {v.ndim}")  # 1

# Row vector
v_row = v.reshape(1, -1)
print(f"Row vector shape: {v_row.shape}")  # (1, 2)
```

### Vector Operations

#### 1. Vector Addition

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Element-wise addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")  # [5 7 9]

# Geometric interpretation: Parallelogram rule
# Visual representation
fig, ax = plt.subplots()
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
ax.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1+v2')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.legend()
plt.grid(True)
plt.show()
```

#### 2. Scalar Multiplication

```python
v = np.array([1, 2, 3])
scalar = 2

# Multiply each component
v_scaled = scalar * v
print(f"2 * v = {v_scaled}")  # [2 4 6]

# Geometric interpretation: Scaling (stretching/compressing)
```

#### 3. Dot Product (Inner Product)

The dot product measures **similarity** and **projection**.

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")  # 32

# Formula: v1 · v2 = Σ(v1_i * v2_i) = 1*4 + 2*5 + 3*6 = 32

# Alternative notation
dot_product = v1 @ v2  # Python 3.5+
dot_product = np.sum(v1 * v2)  # Element-wise then sum
```

**Geometric Interpretation:**
- **Projection**: How much of v1 points in direction of v2
- **Similarity**: Higher dot product = more similar directions
- **Angle**: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)

```python
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

# Example: Similar vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # v2 = 2 * v1 (same direction)
similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity: {similarity}")  # 1.0 (identical direction)

# Orthogonal vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])
similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity (orthogonal): {similarity}")  # 0.0
```

#### 4. Vector Norm (Length/Magnitude)

```python
v = np.array([3, 4])

# L2 norm (Euclidean distance)
norm_l2 = np.linalg.norm(v)
print(f"||v||₂ = {norm_l2}")  # 5.0 (√(3² + 4²) = √25 = 5)

# L1 norm (Manhattan distance)
norm_l1 = np.sum(np.abs(v))
print(f"||v||₁ = {norm_l1}")  # 7 (|3| + |4|)

# L∞ norm (Max norm)
norm_inf = np.max(np.abs(v))
print(f"||v||∞ = {norm_inf}")  # 4 (max(|3|, |4|))

# Normalize vector (unit vector)
v_normalized = v / np.linalg.norm(v)
print(f"Normalized v: {v_normalized}")  # [0.6, 0.8]
print(f"Norm of normalized: {np.linalg.norm(v_normalized)}")  # 1.0
```

**Why norms matter in ML:**
- **Regularization**: L1 (Lasso), L2 (Ridge) regularization
- **Distance metrics**: K-means, KNN use norms
- **Normalization**: Feature scaling

#### 5. Cross Product (3D only)

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Cross product (perpendicular to both vectors)
cross_product = np.cross(v1, v2)
print(f"v1 × v2 = {cross_product}")  # [-3  6 -3]

# Used in: Computer graphics, physics simulations
```

---

## Matrices

### What is a Matrix?

A **matrix** is a 2D array of numbers. In ML:
- **Data matrix**: Rows = samples, Columns = features
- **Weight matrix**: Neural network parameters
- **Transformation matrix**: Linear transformations

### Matrix Representation

```python
# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")  # (3, 3) - 3 rows, 3 columns
print(f"Size: {A.size}")   # 9 elements
print(f"Rank: {A.ndim}")   # 2 dimensions

# Special matrices
# Zero matrix
zeros = np.zeros((3, 3))
print(f"Zero matrix:\n{zeros}")

# Identity matrix
I = np.eye(3)
print(f"Identity matrix:\n{I}")

# Ones matrix
ones = np.ones((2, 3))
print(f"Ones matrix:\n{ones}")
```

### Matrix Operations

#### 1. Matrix Addition and Subtraction

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Element-wise addition
C = A + B
print(f"A + B =\n{C}")
# [[ 6  8]
#  [10 12]]

# Element-wise subtraction
D = A - B
print(f"A - B =\n{D}")
# [[-4 -4]
#  [-4 -4]]
```

#### 2. Scalar Multiplication

```python
A = np.array([[1, 2],
              [3, 4]])

# Multiply each element
B = 2 * A
print(f"2 * A =\n{B}")
# [[2 4]
#  [6 8]]
```

#### 3. Matrix Multiplication

**Critical for ML!** This is how neural networks work.

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication (NOT element-wise!)
C = np.dot(A, B)
# Alternative: C = A @ B

print(f"A @ B =\n{C}")
# [[19 22]
#  [43 50]]

# How it works:
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
# C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
# C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
# C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
```

**Important Rules:**
- **Commutative?** NO! A @ B ≠ B @ A (in general)
- **Associative?** YES! (A @ B) @ C = A @ (B @ C)
- **Distributive?** YES! A @ (B + C) = A @ B + A @ C

```python
# Demonstrate non-commutativity
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

AB = A @ B
BA = B @ A

print(f"A @ B =\n{AB}")
print(f"\nB @ A =\n{BA}")
print(f"\nAre they equal? {np.array_equal(AB, BA)}")  # False
```

#### 4. Matrix-Vector Multiplication

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
v = np.array([1, 2, 3])

# Matrix-vector multiplication
result = A @ v
print(f"A @ v = {result}")  # [14 32]

# How it works:
# result[0] = A[0,0]*v[0] + A[0,1]*v[1] + A[0,2]*v[2] = 1*1 + 2*2 + 3*3 = 14
# result[1] = A[1,0]*v[0] + A[1,1]*v[1] + A[1,2]*v[2] = 4*1 + 5*2 + 6*3 = 32
```

**In ML:**
- Neural network forward pass: `output = W @ input + b`
- Linear regression: `predictions = X @ weights`

#### 5. Transpose

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Transpose (swap rows and columns)
A_T = A.T
print(f"A =\n{A}")
print(f"\nA^T =\n{A_T}")
# A:    A^T:
# [[1 2 3]   [[1 4]
#  [4 5 6]]   [2 5]
#             [3 6]]

# Properties:
# (A^T)^T = A
# (A + B)^T = A^T + B^T
# (AB)^T = B^T @ A^T
```

**In ML:**
- Gradient calculations: `gradient = X^T @ error`
- Normal equations: `weights = (X^T @ X)^(-1) @ X^T @ y`

#### 6. Matrix Inverse

```python
A = np.array([[1, 2],
              [3, 4]])

# Inverse (only for square, non-singular matrices)
A_inv = np.linalg.inv(A)
print(f"A^(-1) =\n{A_inv}")
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify: A @ A^(-1) = I
I = A @ A_inv
print(f"A @ A^(-1) =\n{I}")
# Should be identity matrix (approximately)

# Properties:
# (A^(-1))^(-1) = A
# (AB)^(-1) = B^(-1) @ A^(-1)
```

**In ML:**
- Normal equations for linear regression
- Solving systems of linear equations

---

## Solving Systems of Linear Equations

### Gaussian Elimination (Row Reduction)

**Gaussian elimination** is the fundamental algorithm for solving systems of linear equations. It transforms a matrix into **row echelon form (REF)** or **reduced row echelon form (RREF)**.

```python
import numpy as np
from scipy.linalg import lu

# Example system: Ax = b
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([[8],
              [-11],
              [-3]])

# Augmented matrix [A | b]
augmented = np.hstack([A, b])
print("Augmented matrix [A | b]:")
print(augmented)

# Row reduction using NumPy
# P: permutation matrix, L: lower triangular, U: upper triangular
P, L, U = lu(A)

print("\nUpper triangular (U) - row echelon form:")
print(U)

# Solve using back substitution
from scipy.linalg import solve
x = solve(A, b)
print(f"\nSolution x:\n{x}")
```

**Row Echelon Form (REF):**
- All zero rows at the bottom
- First nonzero entry (pivot) in each row is 1
- Each pivot is to the right of the pivot above it

**Reduced Row Echelon Form (RREF):**
- REF + each pivot is the only nonzero entry in its column

```python
# Manual row reduction example
def row_reduce(A):
    """Convert matrix to row echelon form"""
    A = A.copy().astype(float)
    m, n = A.shape
    
    pivot_row = 0
    for col in range(n):
        # Find pivot
        pivot_idx = None
        for row in range(pivot_row, m):
            if abs(A[row, col]) > 1e-10:
                pivot_idx = row
                break
        
        if pivot_idx is None:
            continue
        
        # Swap rows
        A[[pivot_row, pivot_idx]] = A[[pivot_idx, pivot_row]]
        
        # Normalize pivot
        pivot_val = A[pivot_row, col]
        A[pivot_row] /= pivot_val
        
        # Eliminate below
        for row in range(pivot_row + 1, m):
            factor = A[row, col]
            A[row] -= factor * A[pivot_row]
        
        pivot_row += 1
        if pivot_row >= m:
            break
    
    return A

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
A_ref = row_reduce(A)
print("Row echelon form:")
print(A_ref)
```

### Solving Ax = 0 (Homogeneous System)

When solving **Ax = 0**, we find the **nullspace** (kernel) of A.

```python
# Example: Find nullspace of A
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]])

# Find nullspace using SVD
U, S, Vt = np.linalg.svd(A)

# Nullspace vectors are columns of Vt corresponding to zero singular values
# In practice, we look for very small singular values
tolerance = 1e-10
nullspace_basis = Vt[S < tolerance]

print("Singular values:", S)
print("Nullspace basis vectors:")
print(nullspace_basis)

# Verify: A @ nullspace_vector ≈ 0
if len(nullspace_basis) > 0:
    null_vec = nullspace_basis[0]
    result = A @ null_vec
    print(f"\nVerification: A @ nullspace_vector = {result}")
    print(f"Is it approximately zero? {np.allclose(result, 0)}")
```

**Pivot Variables vs Free Variables:**
- **Pivot variables**: Correspond to pivot columns (determined by the system)
- **Free variables**: Correspond to non-pivot columns (can be set arbitrarily)

### Solving Ax = b (Non-homogeneous System)

```python
# Example: Solve Ax = b
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
b = np.array([1, 2, 3])

# Check if system has a solution
rank_A = np.linalg.matrix_rank(A)
augmented = np.hstack([A, b.reshape(-1, 1)])
rank_aug = np.linalg.matrix_rank(augmented)

print(f"Rank of A: {rank_A}")
print(f"Rank of [A|b]: {rank_aug}")

if rank_A == rank_aug:
    if rank_A == A.shape[1]:
        # Unique solution
        x = np.linalg.solve(A, b)
        print(f"Unique solution: {x}")
    else:
        # Infinite solutions
        print("Infinite solutions (underdetermined)")
        # Use least squares for one solution
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"One solution (least squares): {x}")
else:
    print("No solution (inconsistent system)")
    # Use least squares for best approximation
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"Best approximation (least squares): {x}")
```

**In ML:**
- Linear regression: Solving normal equations
- Neural networks: Backpropagation involves solving systems
- Optimization: Constraint satisfaction

---

## Column Space and Nullspace

### Column Space (Range)

The **column space** C(A) is the span of all columns of A. It contains all possible outputs of the transformation Ax.

```python
# Find column space
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Column space = span of pivot columns
# Use QR decomposition to find basis
Q, R = np.linalg.qr(A)

# Pivot columns correspond to nonzero diagonal entries of R
pivot_cols = np.abs(np.diag(R)) > 1e-10
column_space_basis = Q[:, pivot_cols]

print("Column space basis (orthonormal):")
print(column_space_basis)
print(f"\nDimension of column space (rank): {np.sum(pivot_cols)}")
```

### Nullspace (Kernel)

The **nullspace** N(A) contains all vectors x such that Ax = 0.

```python
# Find nullspace
A = np.array([[1, 2, 3],
              [2, 4, 6]])

# Nullspace using SVD
U, S, Vt = np.linalg.svd(A)

# Columns of Vt corresponding to zero singular values form nullspace basis
tolerance = 1e-10
nullspace_basis = Vt[S < tolerance].T

print("Nullspace basis:")
print(nullspace_basis)
print(f"\nDimension of nullspace (nullity): {nullspace_basis.shape[1]}")
print(f"Rank + Nullity = {np.linalg.matrix_rank(A)} + {nullspace_basis.shape[1]} = {A.shape[1]}")
```

**Fundamental Theorem:**
- **Rank-Nullity Theorem**: rank(A) + nullity(A) = n (number of columns)

---

## The Four Fundamental Subspaces

For any matrix A (m × n), there are **four fundamental subspaces**:

1. **Column Space** C(A): Subspace of R^m spanned by columns
2. **Row Space** C(A^T): Subspace of R^n spanned by rows
3. **Nullspace** N(A): Subspace of R^n of solutions to Ax = 0
4. **Left Nullspace** N(A^T): Subspace of R^m of solutions to A^T y = 0

```python
def fundamental_subspaces(A):
    """Analyze the four fundamental subspaces of matrix A"""
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)
    
    # 1. Column Space C(A) - dimension = rank
    U, S, Vt = np.linalg.svd(A)
    col_space_basis = U[:, :rank]
    
    # 2. Row Space C(A^T) - dimension = rank
    row_space_basis = Vt[:rank, :].T
    
    # 3. Nullspace N(A) - dimension = n - rank
    nullity = n - rank
    if nullity > 0:
        nullspace_basis = Vt[rank:, :].T
    else:
        nullspace_basis = np.zeros((n, 0))
    
    # 4. Left Nullspace N(A^T) - dimension = m - rank
    left_nullity = m - rank
    if left_nullity > 0:
        left_nullspace_basis = U[:, rank:].T
    else:
        left_nullspace_basis = np.zeros((0, m))
    
    return {
        'column_space': col_space_basis,
        'row_space': row_space_basis,
        'nullspace': nullspace_basis,
        'left_nullspace': left_nullspace_basis,
        'rank': rank,
        'nullity': nullity,
        'left_nullity': left_nullity
    }

# Example
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

subspaces = fundamental_subspaces(A)
print("Four Fundamental Subspaces:")
print(f"Rank: {subspaces['rank']}")
print(f"Column space dimension: {subspaces['rank']}")
print(f"Row space dimension: {subspaces['rank']}")
print(f"Nullspace dimension: {subspaces['nullity']}")
print(f"Left nullspace dimension: {subspaces['left_nullity']}")

# Orthogonality relationships
print("\nOrthogonality:")
print("Column space ⟂ Left nullspace")
print("Row space ⟂ Nullspace")
```

**Key Relationships:**
- Column space ⟂ Left nullspace
- Row space ⟂ Nullspace
- C(A) ⊕ N(A^T) = R^m
- C(A^T) ⊕ N(A) = R^n

**In ML:**
- Understanding model capacity and constraints
- Analyzing data dependencies
- Dimensionality reduction

---

## Linear Transformations

### What are Linear Transformations?

A **linear transformation** is a function that preserves vector addition and scalar multiplication:
- T(v + w) = T(v) + T(w)
- T(cv) = cT(v)

### Matrix as Linear Transformation

Every matrix represents a linear transformation!

```python
# Rotation matrix (rotate 90 degrees counterclockwise)
R = np.array([[0, -1],
              [1,  0]])

# Original vector
v = np.array([1, 0])

# Transform
v_rotated = R @ v
print(f"Original: {v}")
print(f"Rotated: {v_rotated}")  # [0, 1]

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Before
ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')
ax1.set_xlim([-1, 2])
ax1.set_ylim([-1, 2])
ax1.grid(True)
ax1.set_title('Before Rotation')

# After
ax2.quiver(0, 0, v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1, color='b')
ax2.set_xlim([-1, 2])
ax2.set_ylim([-1, 2])
ax2.grid(True)
ax2.set_title('After Rotation')
plt.show()
```

### Common Transformations

```python
# Scaling
S = np.array([[2, 0],
              [0, 3]])  # Scale x by 2, y by 3

# Reflection
Ref = np.array([[-1, 0],
                [0,  1]])  # Reflect across y-axis

# Shear
Sh = np.array([[1, 1],
               [0, 1]])  # Shear transformation
```

---

## Eigenvalues and Eigenvectors

### What are Eigenvalues and Eigenvectors?

For a matrix A, if:
```
A @ v = λ @ v
```
Then:
- **v** is an eigenvector
- **λ** (lambda) is the corresponding eigenvalue

**Intuition**: Eigenvectors are directions that don't change when transformed by the matrix (only scaled).

### Computing Eigenvalues and Eigenvectors

```python
A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
# [5. 2.]

print(f"Eigenvectors:\n{eigenvectors}")
# [[ 0.89442719 -0.70710678]
#  [ 0.4472136   0.70710678]]

# Verify: A @ v = λ @ v
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
    result = A @ eigenvector
    expected = eigenvalue * eigenvector
    print(f"\nEigenvalue {i+1}: {eigenvalue:.2f}")
    print(f"A @ v = {result}")
    print(f"λ * v = {expected}")
    print(f"Are they equal? {np.allclose(result, expected)}")  # True
```

### Applications in ML

#### 1. Principal Component Analysis (PCA)

PCA finds the directions of maximum variance using eigenvectors.

```python
# Example: PCA on 2D data
np.random.seed(42)
data = np.random.randn(100, 2)
data[:, 1] = 0.5 * data[:, 0] + 0.5 * data[:, 1]  # Create correlation

# Center the data
data_centered = data - data.mean(axis=0)

# Covariance matrix
cov_matrix = np.cov(data_centered.T)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Principal components (eigenvectors):\n{eigenvectors}")
print(f"Explained variance (eigenvalues): {eigenvalues}")

# First principal component (direction of max variance)
PC1 = eigenvectors[:, 0]
print(f"\nFirst Principal Component: {PC1}")
```

#### 2. PageRank Algorithm

Google's PageRank uses eigenvectors of the web graph matrix.

#### 3. Spectral Clustering

Uses eigenvectors of the graph Laplacian.

---

## Determinants

### What is a Determinant?

The **determinant** det(A) is a scalar value that encodes important properties of a square matrix:
- **Volume scaling factor**: How much the transformation scales volumes
- **Invertibility**: det(A) ≠ 0 ⟺ A is invertible
- **Orientation**: Sign indicates if transformation preserves or reverses orientation

### Computing Determinants

```python
# Determinant of a 2×2 matrix
A_2x2 = np.array([[a, b],
                  [c, d]])
# det(A) = ad - bc

# Determinant of a 3×3 matrix
A_3x3 = np.array([[a, b, c],
                  [d, e, f],
                  [g, h, i]])
# det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)

# Using NumPy
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")

# Check if matrix is invertible
is_invertible = abs(det_A) > 1e-10
print(f"Is A invertible? {is_invertible}")
```

### Properties of Determinants

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 1. det(AB) = det(A) * det(B)
det_AB = np.linalg.det(A @ B)
det_A_times_det_B = np.linalg.det(A) * np.linalg.det(B)
print(f"det(AB) = {det_AB}")
print(f"det(A) * det(B) = {det_A_times_det_B}")
print(f"Are they equal? {np.isclose(det_AB, det_A_times_det_B)}")

# 2. det(A^T) = det(A)
det_AT = np.linalg.det(A.T)
print(f"\ndet(A^T) = {det_AT}")
print(f"det(A) = {np.linalg.det(A)}")

# 3. det(A^(-1)) = 1 / det(A)
A_inv = np.linalg.inv(A)
det_A_inv = np.linalg.det(A_inv)
print(f"\ndet(A^(-1)) = {det_A_inv}")
print(f"1 / det(A) = {1 / np.linalg.det(A)}")

# 4. det(cA) = c^n * det(A) for n×n matrix
c = 2
det_cA = np.linalg.det(c * A)
c_to_n = c ** A.shape[0]
print(f"\ndet({c}A) = {det_cA}")
print(f"{c}^n * det(A) = {c_to_n * np.linalg.det(A)}")
```

### Cofactor Expansion

```python
def cofactor_expansion(A):
    """Compute determinant using cofactor expansion"""
    n = A.shape[0]
    
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    det = 0
    for j in range(n):
        # Minor: matrix without row 0 and column j
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = (-1) ** (0 + j) * cofactor_expansion(minor)
        det += A[0, j] * cofactor
    
    return det

# Example
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

det_cofactor = cofactor_expansion(A)
det_numpy = np.linalg.det(A)
print(f"Determinant (cofactor): {det_cofactor}")
print(f"Determinant (NumPy): {det_numpy}")
print(f"Are they equal? {np.isclose(det_cofactor, det_numpy)}")
```

### Cramer's Rule

**Cramer's Rule** solves Ax = b using determinants:

```python
def cramers_rule(A, b):
    """Solve Ax = b using Cramer's rule"""
    det_A = np.linalg.det(A)
    
    if abs(det_A) < 1e-10:
        raise ValueError("Matrix is singular, Cramer's rule cannot be applied")
    
    n = A.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        # Replace column i with b
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A
    
    return x

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x_cramer = cramers_rule(A, b)
x_solve = np.linalg.solve(A, b)

print(f"Solution (Cramer's rule): {x_cramer}")
print(f"Solution (np.linalg.solve): {x_solve}")
print(f"Are they equal? {np.allclose(x_cramer, x_solve)}")
```

**Note**: Cramer's rule is computationally expensive (O(n!) for n×n matrix) and mainly of theoretical interest. Use Gaussian elimination or `np.linalg.solve` in practice.

**In ML:**
- Checking matrix invertibility
- Volume calculations in probability distributions
- Change of variables in integrals

---

## Diagonalization

### What is Diagonalization?

A matrix A is **diagonalizable** if it can be written as:
```
A = P @ D @ P^(-1)
```
where D is a diagonal matrix and P contains the eigenvectors.

**Conditions for Diagonalization:**
- A has n linearly independent eigenvectors (n×n matrix)
- A is symmetric (always diagonalizable)
- All eigenvalues are distinct (sufficient but not necessary)

### Diagonalization Process

```python
def diagonalize(A):
    """Diagonalize matrix A = P @ D @ P^(-1)"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Check if diagonalizable (all eigenvalues are distinct or matrix is symmetric)
    if len(eigenvalues) != len(set(eigenvalues)) and not np.allclose(A, A.T):
        print("Warning: Matrix may not be diagonalizable")
    
    # D: diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    
    # P: matrix of eigenvectors (columns)
    P = eigenvectors
    
    # Verify: A = P @ D @ P^(-1)
    P_inv = np.linalg.inv(P)
    A_reconstructed = P @ D @ P_inv
    
    return P, D, P_inv, A_reconstructed

# Example: Diagonalizable matrix
A = np.array([[4, 1],
              [2, 3]])

P, D, P_inv, A_reconstructed = diagonalize(A)

print("Original A:")
print(A)
print("\nEigenvalues (diagonal of D):")
print(np.diag(D))
print("\nEigenvectors (columns of P):")
print(P)
print("\nReconstructed A = P @ D @ P^(-1):")
print(A_reconstructed)
print(f"\nIs reconstruction accurate? {np.allclose(A, A_reconstructed)}")
```

### Powers of A

Diagonalization makes computing powers of A easy:

```python
# Computing A^k using diagonalization
A = np.array([[4, 1],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# A^k = P @ D^k @ P^(-1)
# Since D is diagonal, D^k is just raising each diagonal element to k
k = 5
D_k = np.diag(eigenvalues ** k)
A_k = P @ D_k @ P_inv

# Verify with direct computation
A_k_direct = np.linalg.matrix_power(A, k)

print(f"A^{k} (using diagonalization):")
print(A_k)
print(f"\nA^{k} (direct computation):")
print(A_k_direct)
print(f"\nAre they equal? {np.allclose(A_k, A_k_direct)}")
```

**In ML:**
- Markov chains: Computing state probabilities after k steps
- Graph algorithms: Computing paths of length k
- Recurrent neural networks: Analyzing long-term dependencies

---

## Matrix Exponentials

### What is exp(At)?

The **matrix exponential** exp(At) is defined as:
```
exp(At) = I + At + (At)²/2! + (At)³/3! + ...
```

It's crucial for solving systems of linear differential equations.

### Computing Matrix Exponential

```python
from scipy.linalg import expm

# Example: exp(At) for solving dx/dt = Ax
A = np.array([[-2, 1],
              [1, -2]])

t = 1.0
exp_At = expm(A * t)

print(f"exp(A * {t}):")
print(exp_At)

# Using diagonalization: exp(At) = P @ exp(Dt) @ P^(-1)
eigenvalues, eigenvectors = np.linalg.eig(A)
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# exp(Dt) is diagonal with exp(λᵢt) on diagonal
exp_Dt = np.diag(np.exp(eigenvalues * t))
exp_At_diag = P @ exp_Dt @ P_inv

print(f"\nexp(At) using diagonalization:")
print(exp_At_diag)
print(f"\nAre they equal? {np.allclose(exp_At, exp_At_diag)}")
```

### Solving Differential Equations

```python
# System: dx/dt = Ax, x(0) = x0
# Solution: x(t) = exp(At) @ x0

A = np.array([[-1, 2],
              [0, -3]])
x0 = np.array([1, 1])

# Solution at time t
t = 2.0
exp_At = expm(A * t)
x_t = exp_At @ x0

print(f"Initial condition x(0) = {x0}")
print(f"Solution at t = {t}: x({t}) = {x_t}")

# Visualize solution over time
import matplotlib.pyplot as plt

times = np.linspace(0, 5, 100)
solutions = []

for t in times:
    exp_At = expm(A * t)
    x_t = exp_At @ x0
    solutions.append(x_t)

solutions = np.array(solutions)

plt.figure(figsize=(10, 6))
plt.plot(times, solutions[:, 0], label='x₁(t)')
plt.plot(times, solutions[:, 1], label='x₂(t)')
plt.xlabel('Time t')
plt.ylabel('x(t)')
plt.title('Solution of dx/dt = Ax')
plt.legend()
plt.grid(True)
plt.show()
```

**In ML:**
- Continuous-time neural networks
- Dynamical systems modeling
- Control theory applications

---

## Matrix Decompositions

### 1. Eigendecomposition

```python
A = np.array([[4, 2],
              [1, 3]])

# Eigendecomposition: A = Q @ Λ @ Q^(-1)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Reconstruct
Q = eigenvectors
Lambda = np.diag(eigenvalues)
Q_inv = np.linalg.inv(Q)

A_reconstructed = Q @ Lambda @ Q_inv
print(f"Original A:\n{A}")
print(f"\nReconstructed A:\n{A_reconstructed}")
print(f"\nAre they equal? {np.allclose(A, A_reconstructed)}")  # True
```

### 2. Singular Value Decomposition (SVD)

**Most important decomposition for ML!**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# SVD: A = U @ Σ @ V^T
U, S, Vt = np.linalg.svd(A)

print(f"U (left singular vectors):\n{U}")
print(f"\nS (singular values): {S}")
print(f"\nV^T (right singular vectors, transposed):\n{Vt}")

# Reconstruct
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:len(S), :len(S)] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt

print(f"\nOriginal A:\n{A}")
print(f"\nReconstructed A:\n{A_reconstructed}")
print(f"\nAre they equal? {np.allclose(A, A_reconstructed)}")  # True
```

**Applications:**
- **PCA**: SVD of centered data matrix
- **Matrix Factorization**: Recommender systems
- **Dimensionality Reduction**: Low-rank approximation
- **Image Compression**: Keep only top singular values

```python
# Low-rank approximation example
A = np.random.randn(100, 100)

# Full SVD
U, S, Vt = np.linalg.svd(A)

# Keep only top 10 singular values (rank-10 approximation)
k = 10
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]

# Reconstruct
A_approx = U_k @ np.diag(S_k) @ Vt_k

# Compression ratio
original_size = A.size
compressed_size = U_k.size + S_k.size + Vt_k.size
print(f"Original size: {original_size}")
print(f"Compressed size: {compressed_size}")
print(f"Compression ratio: {original_size / compressed_size:.2f}x")
```

### 3. QR Decomposition

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# QR decomposition: A = Q @ R
Q, R = np.linalg.qr(A)

print(f"Q (orthogonal):\n{Q}")
print(f"\nR (upper triangular):\n{R}")

# Verify
A_reconstructed = Q @ R
print(f"\nOriginal A:\n{A}")
print(f"\nReconstructed A:\n{A_reconstructed}")
```

**Applications:**
- Solving least squares problems
- Gram-Schmidt orthogonalization

---

## Pseudoinverse (Moore-Penrose Inverse)

### What is a Pseudoinverse?

The **pseudoinverse** A⁺ generalizes the matrix inverse to non-square or singular matrices. It's the unique matrix satisfying:
- A A⁺ A = A
- A⁺ A A⁺ = A⁺
- (A A⁺)^T = A A⁺
- (A⁺ A)^T = A⁺ A

### Computing Pseudoinverse

```python
# Using SVD: A⁺ = V @ Σ⁺ @ U^T
# where Σ⁺ is the pseudoinverse of Σ (reciprocal of nonzero singular values)

def pseudoinverse_svd(A):
    """Compute pseudoinverse using SVD"""
    U, S, Vt = np.linalg.svd(A)
    
    # Pseudoinverse of Σ: reciprocal of nonzero singular values
    S_plus = np.zeros((A.shape[1], A.shape[0]))
    tolerance = 1e-10
    for i in range(min(A.shape[0], A.shape[1])):
        if S[i] > tolerance:
            S_plus[i, i] = 1.0 / S[i]
    
    # A⁺ = V @ Σ⁺ @ U^T
    A_plus = Vt.T @ S_plus @ U.T
    return A_plus

# Example: Overdetermined system (more equations than unknowns)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
b = np.array([1, 2, 3])

# Using NumPy
A_plus_numpy = np.linalg.pinv(A)

# Using our SVD implementation
A_plus_svd = pseudoinverse_svd(A)

print("Pseudoinverse (NumPy):")
print(A_plus_numpy)
print("\nPseudoinverse (SVD):")
print(A_plus_svd)
print(f"\nAre they equal? {np.allclose(A_plus_numpy, A_plus_svd)}")

# Solve overdetermined system: x = A⁺ @ b
x = A_plus_numpy @ b
print(f"\nSolution x = A⁺ @ b: {x}")

# Verify: A @ x should be close to b (least squares solution)
print(f"A @ x = {A @ x}")
print(f"b = {b}")
print(f"Error: {np.linalg.norm(A @ x - b)}")
```

### Left and Right Inverses

```python
# Left inverse: A_left^(-1) @ A = I (for tall matrices, m > n)
# Right inverse: A @ A_right^(-1) = I (for wide matrices, m < n)

# Tall matrix (more rows than columns)
A_tall = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

# Left inverse: (A^T @ A)^(-1) @ A^T
A_left_inv = np.linalg.inv(A_tall.T @ A_tall) @ A_tall.T
print("Left inverse:")
print(A_left_inv)
print(f"A_left^(-1) @ A = I? {np.allclose(A_left_inv @ A_tall, np.eye(2))}")

# Wide matrix (more columns than rows)
A_wide = A_tall.T

# Right inverse: A^T @ (A @ A^T)^(-1)
A_right_inv = A_wide.T @ np.linalg.inv(A_wide @ A_wide.T)
print("\nRight inverse:")
print(A_right_inv)
print(f"A @ A_right^(-1) = I? {np.allclose(A_wide @ A_right_inv, np.eye(2))}")
```

**In ML:**
- **Least squares**: Solving overdetermined systems
- **Ridge regression**: Regularized least squares
- **Neural networks**: Backpropagation with non-square weight matrices
- **Dimensionality reduction**: Projecting onto subspaces

---

## Positive Definite Matrices

### What are Positive Definite Matrices?

A symmetric matrix A is **positive definite** if:
- x^T A x > 0 for all nonzero vectors x
- All eigenvalues are positive
- All leading principal minors are positive

### Properties and Applications

```python
def is_positive_definite(A):
    """Check if matrix is positive definite"""
    # Must be symmetric
    if not np.allclose(A, A.T):
        return False
    
    # All eigenvalues must be positive
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

# Example: Covariance matrix (always positive semidefinite, often positive definite)
np.random.seed(42)
data = np.random.randn(100, 3)
cov_matrix = np.cov(data.T)

print("Covariance matrix:")
print(cov_matrix)
print(f"\nIs positive definite? {is_positive_definite(cov_matrix)}")

# Eigenvalues
eigenvalues = np.linalg.eigvals(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive? {np.all(eigenvalues > 0)}")
```

### Positive Semidefinite Matrices

A matrix is **positive semidefinite** if x^T A x ≥ 0 for all x (eigenvalues ≥ 0).

```python
# Example: Gram matrix (always positive semidefinite)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Gram matrix: G = X @ X^T
G = X @ X.T

print("Gram matrix G = X @ X^T:")
print(G)
print(f"\nIs symmetric? {np.allclose(G, G.T)}")

eigenvalues = np.linalg.eigvals(G)
print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive semidefinite? {np.all(eigenvalues >= 0)}")
```

### Cholesky Decomposition

For positive definite matrices, we can use **Cholesky decomposition**: A = L @ L^T

```python
# Cholesky decomposition: A = L @ L^T
# Only works for positive definite matrices

# Create positive definite matrix
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

# Make it symmetric and positive definite
A = (A + A.T) / 2
A = A + 2 * np.eye(3)  # Ensure positive definiteness

try:
    L = np.linalg.cholesky(A)
    print("Cholesky factor L:")
    print(L)
    print("\nVerification: L @ L^T = A?")
    print(L @ L.T)
    print(f"Is it equal to A? {np.allclose(L @ L.T, A)}")
except np.linalg.LinAlgError:
    print("Matrix is not positive definite")
```

**In ML:**
- **Covariance matrices**: Always positive semidefinite
- **Kernel matrices**: In kernel methods (SVM, Gaussian processes)
- **Hessian matrices**: In optimization (Newton's method)
- **Regularization**: Ensuring numerical stability

---

## Complex Matrices and Fast Fourier Transform

### Complex Matrices

Complex matrices have complex entries. Many properties extend from real matrices.

```python
# Complex matrix
A_complex = np.array([[1+2j, 3-1j],
                      [2+1j, 4+3j]])

print("Complex matrix A:")
print(A_complex)

# Conjugate transpose (Hermitian transpose)
A_H = A_complex.conj().T
print("\nConjugate transpose A^H:")
print(A_H)

# Hermitian matrix: A^H = A
A_hermitian = np.array([[2, 1+1j],
                        [1-1j, 3]])
print("\nHermitian matrix (A^H = A):")
print(A_hermitian)
print(f"Is Hermitian? {np.allclose(A_hermitian, A_hermitian.conj().T)}")

# Unitary matrix: U^H @ U = I
U = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
              [1j/np.sqrt(2), -1j/np.sqrt(2)]])
print("\nUnitary matrix (U^H @ U = I):")
print(U)
print(f"U^H @ U = I? {np.allclose(U.conj().T @ U, np.eye(2))}")
```

### Fast Fourier Transform (FFT)

The **FFT** is an efficient algorithm for computing the Discrete Fourier Transform (DFT).

```python
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# Example: Signal processing
# Create a signal with multiple frequencies
t = np.linspace(0, 1, 1000, endpoint=False)
signal = (np.sin(2 * np.pi * 5 * t) + 
          0.5 * np.sin(2 * np.pi * 10 * t) + 
          0.3 * np.sin(2 * np.pi * 20 * t))

# FFT
fft_signal = fft(signal)
frequencies = fftfreq(len(signal), t[1] - t[0])

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Time domain
ax1.plot(t, signal)
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.set_title('Signal in Time Domain')
ax1.grid(True)

# Frequency domain
ax2.plot(frequencies[:len(frequencies)//2], 
         np.abs(fft_signal[:len(fft_signal)//2]))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Signal in Frequency Domain (FFT)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Inverse FFT
signal_reconstructed = ifft(fft_signal)
print(f"Reconstruction error: {np.max(np.abs(signal - signal_reconstructed.real))}")
```

### FFT in Machine Learning

```python
# Example: Convolution using FFT (much faster for large kernels)
# Convolution in time domain = multiplication in frequency domain

# Signal
x = np.random.randn(1000)

# Kernel
kernel = np.array([0.25, 0.5, 0.25])

# Convolution (time domain)
conv_time = np.convolve(x, kernel, mode='same')

# Convolution (frequency domain using FFT)
X = fft(x, n=len(x) + len(kernel) - 1)
K = fft(kernel, n=len(x) + len(kernel) - 1)
conv_freq = ifft(X * K).real
conv_freq = conv_freq[:len(x)]  # Trim to same length

print(f"Time domain convolution error: {np.max(np.abs(conv_time - conv_freq))}")
```

**In ML:**
- **Signal processing**: Audio, image processing
- **Convolutional Neural Networks**: FFT-based convolution
- **Time series analysis**: Frequency domain features
- **Data compression**: JPEG, MP3 use FFT variants

---

## Computational Math in Practice

### Why Computational Math Matters

Understanding math concepts is important, but seeing them in code helps build intuition. This section shows how mathematical concepts translate directly to Python code used in machine learning.

### Example 1: Matrix Multiplication in Neural Networks

**Mathematical Concept**: Matrix multiplication `C = A @ B` where `C[i,j] = Σ A[i,k] * B[k,j]`

**In Code**:
```python
import numpy as np

# Input layer (3 features, batch of 4 samples)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

# Weight matrix (3 features -> 2 neurons)
W = np.array([[0.5, 0.3],
              [0.2, 0.8],
              [0.1, 0.6]])

# Forward pass: X @ W
# Shape: (4, 3) @ (3, 2) = (4, 2)
output = X @ W
print(f"Neural network output:\n{output}")
# Each row is one sample's output from the layer
```

**Visualization**:
```python
import matplotlib.pyplot as plt

# Visualize the transformation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Input space (3D projected to 2D)
ax1.scatter(X[:, 0], X[:, 1], c=range(len(X)), cmap='viridis')
ax1.set_title('Input Space (first 2 features)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Output space (2D)
ax2.scatter(output[:, 0], output[:, 1], c=range(len(output)), cmap='viridis')
ax2.set_title('Output Space (after matrix multiplication)')
ax2.set_xlabel('Neuron 1')
ax2.set_ylabel('Neuron 2')

plt.tight_layout()
plt.show()
```

### Example 2: Gradient Descent Visualization

**Mathematical Concept**: Gradient points in direction of steepest ascent. Gradient descent moves opposite to gradient.

**In Code**:
```python
# Simple 2D function: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    return np.array([2*x, 2*y])

# Gradient descent
x, y = 3.0, 4.0  # Starting point
learning_rate = 0.1
steps = 20

path = [(x, y)]
for i in range(steps):
    grad = gradient(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    path.append((x, y))

# Visualize
x_path, y_path = zip(*path)
x_grid = np.linspace(-4, 4, 100)
y_grid = np.linspace(-4, 4, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z = f(X_grid, Y_grid)

plt.contour(X_grid, Y_grid, Z, levels=20)
plt.plot(x_path, y_path, 'ro-', label='Gradient Descent Path')
plt.plot(0, 0, 'g*', markersize=20, label='Minimum')
plt.legend()
plt.title('Gradient Descent on f(x,y) = x² + y²')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### Example 3: Eigenvalues in PCA

**Mathematical Concept**: Eigenvalues represent variance explained by principal components.

**In Code**:
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
# Create correlation
X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Eigenvalues = explained variance
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

print(f"Eigenvalues: {eigenvalues}")
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"First component explains {explained_variance_ratio[0]*100:.1f}% of variance")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Original data
ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# After PCA
ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax2.set_title('After PCA (Principal Components)')
ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')

plt.tight_layout()
plt.show()
```

### Example 4: Dot Product for Similarity

**Mathematical Concept**: Dot product measures similarity between vectors. Used in recommendation systems, embeddings.

**In Code**:
```python
# Word embeddings (simplified)
words = {
    'king': np.array([1, 2, 3]),
    'queen': np.array([1, 2, 2]),
    'man': np.array([2, 1, 3]),
    'woman': np.array([2, 1, 2]),
    'apple': np.array([0, 0, 1])
}

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar words to 'king'
king = words['king']
similarities = {}
for word, embedding in words.items():
    if word != 'king':
        similarities[word] = cosine_similarity(king, embedding)

# Sort by similarity
sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print("Words most similar to 'king':")
for word, sim in sorted_words:
    print(f"  {word}: {sim:.3f}")
```

### Key Takeaway

**Math → Code → Intuition**:
1. Learn the mathematical concept
2. See it in Python code
3. Visualize it with plots
4. Apply it in ML algorithms

This approach builds deeper understanding than theory alone.

---

## Applications in ML

### 1. Neural Networks

```python
# Forward pass in a neural network layer
# output = activation(W @ input + b)

# Example: Single layer
input_size = 3
hidden_size = 4
output_size = 2

# Weight matrices
W1 = np.random.randn(hidden_size, input_size)  # First layer weights
b1 = np.random.randn(hidden_size)              # First layer bias
W2 = np.random.randn(output_size, hidden_size) # Second layer weights
b2 = np.random.randn(output_size)              # Second layer bias

# Input
x = np.array([1, 2, 3])

# Forward pass
z1 = W1 @ x + b1
a1 = np.maximum(0, z1)  # ReLU activation
z2 = W2 @ a1 + b2
output = z2

print(f"Input: {x}")
print(f"Output: {output}")
```

### 2. Linear Regression

```python
# Normal equation: w = (X^T @ X)^(-1) @ X^T @ y

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + 0.1 * np.random.randn(100)

# Solve using normal equation
weights = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"True weights: {true_weights}")
print(f"Estimated weights: {weights}")
```

### 3. Dimensionality Reduction (PCA)

```python
from sklearn.decomposition import PCA

# Generate data
np.random.seed(42)
X = np.random.randn(100, 10)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

---

## Practice Exercises

### Exercise 1: Vector Operations

Implement functions for:
1. Dot product
2. Cosine similarity
3. Euclidean distance

### Exercise 2: Matrix Operations

Implement matrix multiplication from scratch (without NumPy).

### Exercise 3: Eigenvalue Problem

Find eigenvalues/eigenvectors and verify the definition.

### Exercise 4: SVD Application

Use SVD for low-rank matrix approximation.

---

## Resources

### Books

- **"Linear Algebra Done Right"** by Sheldon Axler
- **"Introduction to Linear Algebra"** by Gilbert Strang
- **"Mathematics for Machine Learning"** (free online book)

### Online Courses

- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - **Highly recommended!**
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

### Interactive Tools

- [Matrix Calculator](https://matrixcalc.org/)
- [Eigenvalue Calculator](https://www.symbolab.com/solver/matrix-eigenvalues-calculator)

---

## Key Takeaways

1. **Vectors** represent data points and features
2. **Matrices** represent transformations and datasets
3. **Matrix multiplication** is the core operation in neural networks
4. **Eigenvalues/eigenvectors** are crucial for PCA and spectral methods
5. **SVD** is the most important decomposition for ML
6. **Practice** with NumPy to build intuition

---

**Remember**: Linear algebra is the foundation. Master these concepts, and ML algorithms will make much more sense!

