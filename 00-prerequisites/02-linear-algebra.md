# Linear Algebra for Machine Learning - Deep Dive

Comprehensive guide to linear algebra concepts essential for understanding machine learning algorithms. This is the mathematical foundation of ML.

## Table of Contents

- [Introduction](#introduction)
- [Tensors](#tensors)
- [Vectors](#vectors)
- [Matrices](#matrices)
- [Matrix Operations](#matrix-operations)
- [Linear Transformations](#linear-transformations)
- [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
- [Matrix Decompositions](#matrix-decompositions)
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

