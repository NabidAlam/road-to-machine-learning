# Mathematics Basics for Machine Learning

Essential mathematical concepts needed to understand machine learning algorithms. Focus on practical understanding rather than deep theory.

## Table of Contents

- [Linear Algebra](#linear-algebra)
- [Statistics](#statistics)
- [Probability](#probability)
- [Calculus](#calculus)
- [Practice Exercises](#practice-exercises)

---

## Linear Algebra

### Why Linear Algebra?

Linear algebra is the foundation of machine learning. Most ML algorithms use vectors and matrices for computations.

### Vectors

**What is a Vector?**
A vector is an ordered list of numbers. In ML, vectors represent data points or features.

```python
import numpy as np

# Creating vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

print(vector1)  # Output: [1 2 3]
print(vector2)  # Output: [4 5 6]
```

**Vector Operations:**
```python
# Addition
result = vector1 + vector2
print(result)  # Output: [5 7 9]

# Scalar multiplication
result = 2 * vector1
print(result)  # Output: [2 4 6]

# Dot product
dot_product = np.dot(vector1, vector2)
print(dot_product)  # Output: 32 (1*4 + 2*5 + 3*6)

# Vector norm (length)
norm = np.linalg.norm(vector1)
print(norm)  # Output: 3.741657... (√(1² + 2² + 3²))
```

**Why it matters in ML:**
- Feature vectors represent data points
- Dot products calculate similarity
- Vector norms measure distances

### Matrices

**What is a Matrix?**
A matrix is a 2D array of numbers. In ML, matrices represent datasets or transformations.

```python
# Creating matrices
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

print(matrix.shape)  # Output: (3, 3) - 3 rows, 3 columns
```

**Matrix Operations:**
```python
# Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

C = np.dot(A, B)
print(C)
# Output:
# [[19 22]
#  [43 50]]

# Transpose
A_T = A.T
print(A_T)
# Output:
# [[1 3]
#  [2 4]]

# Matrix-vector multiplication
vector = np.array([1, 2])
result = np.dot(A, vector)
print(result)  # Output: [5 11]
```

**Why it matters in ML:**
- Datasets are matrices (rows = samples, columns = features)
- Neural networks use matrix multiplication
- Transformations are matrix operations

### Key Concepts

**1. Dot Product (Inner Product)**
- Measures similarity between vectors
- Used in neural networks, similarity calculations
- Formula: `a · b = Σ(a_i * b_i)`

**2. Matrix Multiplication**
- Combines transformations
- Core operation in neural networks
- Each element: `C[i,j] = Σ(A[i,k] * B[k,j])`

**3. Transpose**
- Flips rows and columns
- Used in gradient calculations
- Notation: A^T

**4. Identity Matrix**
```python
I = np.eye(3)  # 3x3 identity matrix
print(I)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

---

## Statistics

### Why Statistics?

Statistics help us understand data, make predictions, and evaluate models.

### Descriptive Statistics

**Mean (Average)**
```python
data = [10, 20, 30, 40, 50]
mean = np.mean(data)
print(mean)  # Output: 30.0

# Formula: mean = (1/n) * Σx_i
```

**Median**
```python
median = np.median(data)
print(median)  # Output: 30.0

# Middle value when sorted
# For [10, 20, 30, 40, 50], median = 30
```

**Mode**
```python
from scipy import stats
data = [1, 2, 2, 3, 3, 3, 4]
mode = stats.mode(data)
print(mode.mode)  # Output: [3] (most frequent)
```

**Variance and Standard Deviation**
```python
variance = np.var(data)
std_dev = np.std(data)
print(f"Variance: {variance}")    # Output: Variance: 200.0
print(f"Std Dev: {std_dev}")      # Output: Std Dev: 14.142...

# Variance measures spread
# Standard deviation = √variance
```

**Why it matters:**
- Understand data distribution
- Detect outliers
- Normalize features

### Correlation

**What is Correlation?**
Measures how two variables change together.

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

correlation = np.corrcoef(x, y)[0, 1]
print(correlation)  # Output: 1.0 (perfect positive correlation)

# Range: -1 to 1
# 1 = perfect positive correlation
# -1 = perfect negative correlation
# 0 = no correlation
```

**Why it matters:**
- Feature selection (remove highly correlated features)
- Understand relationships in data

---

## Probability

### Why Probability?

Probability is the foundation of machine learning. Models make probabilistic predictions.

### Basic Concepts

**Probability of Event**
```
P(A) = Number of favorable outcomes / Total outcomes
```

**Example:**
```python
# Probability of rolling a 6 on a die
P_6 = 1 / 6
print(f"P(rolling 6) = {P_6}")  # Output: P(rolling 6) = 0.166...

# Probability of rolling even number
P_even = 3 / 6  # {2, 4, 6} out of {1,2,3,4,5,6}
print(f"P(even) = {P_even}")  # Output: P(even) = 0.5
```

### Conditional Probability

**Formula:**
```
P(A|B) = P(A and B) / P(B)
```

**Example:**
```python
# Probability of rain given clouds
# P(Rain|Clouds) = P(Rain and Clouds) / P(Clouds)
P_rain_and_clouds = 0.3
P_clouds = 0.5
P_rain_given_clouds = P_rain_and_clouds / P_clouds
print(f"P(Rain|Clouds) = {P_rain_given_clouds}")  # Output: 0.6
```

### Bayes' Theorem

**Formula:**
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**Why it matters:**
- Foundation of Naive Bayes classifier
- Used in many ML algorithms

### Probability Distributions

**Normal Distribution (Gaussian)**
```python
from scipy import stats
import matplotlib.pyplot as plt

# Generate normal distribution
mean = 0
std = 1
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, mean, std)

# Most common distribution in ML
# Many ML assumptions rely on normal distribution
```

**Why it matters:**
- Many ML algorithms assume normal distribution
- Used in feature normalization
- Central Limit Theorem

---

## Calculus

### Why Calculus?

Calculus is used in optimization (finding best model parameters) through gradient descent.

### Derivatives

**What is a Derivative?**
Measures rate of change. In ML, we use derivatives to find optimal parameters.

**Simple Example:**
```python
# Function: f(x) = x²
# Derivative: f'(x) = 2x

# At x = 3:
# f(3) = 9
# f'(3) = 6 (slope at x=3)
```

**Why it matters:**
- Gradient descent uses derivatives
- Find minimum of loss functions
- Optimize model parameters

### Gradients

**What is a Gradient?**
Gradient is the vector of partial derivatives. Points in direction of steepest ascent.

```python
# For function f(x, y) = x² + y²
# Gradient: ∇f = [2x, 2y]

# At point (3, 4):
# Gradient = [6, 8]
# Direction of steepest increase
```

**Gradient Descent:**
```python
# Simplified gradient descent
def gradient_descent(f, df, x0, learning_rate=0.01, iterations=100):
    x = x0
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient  # Move opposite to gradient
    return x

# Find minimum of f(x) = x²
# f'(x) = 2x
def f(x):
    return x**2

def df(x):
    return 2*x

minimum = gradient_descent(f, df, x0=5)
print(f"Minimum at x = {minimum}")  # Should be close to 0
```

**Why it matters:**
- Core of training neural networks
- Optimizes all ML models
- Finds best parameters

### Chain Rule

**What is Chain Rule?**
Used in backpropagation (training neural networks).

```
If z = f(y) and y = g(x), then:
dz/dx = (dz/dy) * (dy/dx)
```

**Why it matters:**
- Backpropagation in neural networks
- Training deep learning models

---

## Practice Exercises

### Exercise 1: Vector Operations

**Task:** Calculate the cosine similarity between two vectors.

**Solution:**
```python
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity: {similarity}")
```

### Exercise 2: Statistics

**Task:** Calculate mean, variance, and standard deviation of a dataset.

**Solution:**
```python
def calculate_stats(data):
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    return {"mean": mean, "variance": variance, "std_dev": std_dev}

data = [10, 20, 30, 40, 50]
stats = calculate_stats(data)
print(stats)
# Output: {'mean': 30.0, 'variance': 200.0, 'std_dev': 14.142...}
```

### Exercise 3: Probability

**Task:** Calculate probability of getting at least one head in 3 coin flips.

**Solution:**
```python
# P(at least one head) = 1 - P(no heads)
# P(no heads) = P(all tails) = (1/2)³ = 1/8
P_no_heads = (1/2) ** 3
P_at_least_one_head = 1 - P_no_heads
print(f"P(at least one head) = {P_at_least_one_head}")  # Output: 0.875
```

### Exercise 4: Gradient Calculation

**Task:** Implement gradient descent to find minimum of f(x) = x² + 2x + 1.

**Solution:**
```python
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

def gradient_descent(df, x0=5, learning_rate=0.1, iterations=100):
    x = x0
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
    return x

minimum = gradient_descent(df, x0=5)
print(f"Minimum at x = {minimum}")  # Should be close to -1
print(f"f({minimum}) = {f(minimum)}")  # Should be close to 0
```

---

## Key Takeaways

1. **Linear Algebra**: Vectors and matrices are the language of ML
2. **Statistics**: Help understand and evaluate data
3. **Probability**: Foundation of ML predictions
4. **Calculus**: Used in optimization (gradient descent)

**You don't need to be a math expert!** Understanding these concepts at a practical level is enough to start ML.

---

## Resources for Deeper Learning

- **3Blue1Brown**: Visual explanations of linear algebra and calculus
- **Khan Academy**: Free comprehensive courses
- **Mathematics for Machine Learning (Coursera)**: Applied math for ML

---

## Next Steps

- Practice with NumPy (Python's linear algebra library)
- Work through the exercises
- Move to [03-environment-setup.md](03-environment-setup.md) when ready

**Remember**: Focus on understanding concepts, not memorizing formulas. You'll learn more as you apply these in ML projects!

