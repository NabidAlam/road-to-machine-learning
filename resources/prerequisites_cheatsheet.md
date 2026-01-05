# Prerequisites Cheatsheet

Quick reference for Python, Mathematics, and Statistics fundamentals needed for Machine Learning.

## Table of Contents

- [Python Basics](#python-basics)
- [Linear Algebra](#linear-algebra)
- [Statistics & Probability](#statistics--probability)
- [Calculus](#calculus)
- [Time Complexity](#time-complexity)

---

## Python Basics

### Variables & Data Types

```python
# Numbers
x = 10          # int
y = 3.14        # float
z = 3 + 4j      # complex

# Strings
name = "Alice"
text = f"Hello {name}"  # f-string

# Booleans
is_true = True
is_false = False

# Type conversion
int("123")      # 123
float("3.14")   # 3.14
str(123)        # "123"
```

### Data Structures

```python
# Lists
my_list = [1, 2, 3]
my_list.append(4)           # [1, 2, 3, 4]
my_list[0]                  # 1
my_list[-1]                 # 4 (last element)
my_list[1:3]                # [2, 3] (slicing)

# Dictionaries
my_dict = {"name": "Alice", "age": 25}
my_dict["name"]             # "Alice"
my_dict.get("age", 0)       # 25 (with default)
my_dict.keys()              # dict_keys(['name', 'age'])

# Tuples (immutable)
my_tuple = (1, 2, 3)
my_tuple[0]                 # 1

# Sets (unique elements)
my_set = {1, 2, 3}
my_set.add(4)               # {1, 2, 3, 4}
```

### Control Flow

```python
# If/Else
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")

# Loops
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for item in my_list:
    print(item)

while x > 0:
    x -= 1

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

### Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default arguments
def power(x, n=2):
    return x ** n

# Lambda functions
square = lambda x: x**2
square(5)  # 25

# *args and **kwargs
def func(*args, **kwargs):
    print(args)      # tuple
    print(kwargs)    # dict
```

### Object-Oriented Programming

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, I'm {self.name}"

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
```

### File Operations

```python
# Reading files
with open("file.txt", "r") as f:
    content = f.read()

# Writing files
with open("file.txt", "w") as f:
    f.write("Hello, World!")

# CSV
import csv
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

### Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Always executed")
```

---

## Linear Algebra

### Vectors

```python
import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector operations
v1 + v2                    # [5, 7, 9] (addition)
v1 * 2                     # [2, 4, 6] (scalar multiplication)
np.dot(v1, v2)             # 32 (dot product)
np.linalg.norm(v1)         # 3.74 (magnitude)
```

### Matrices

```python
# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
A + B                      # Element-wise addition
A * B                      # Element-wise multiplication
A @ B                      # Matrix multiplication
np.dot(A, B)               # Matrix multiplication
A.T                        # Transpose
np.linalg.inv(A)           # Inverse
np.linalg.det(A)           # Determinant
np.linalg.eig(A)           # Eigenvalues & eigenvectors
```

### Common Operations

```python
# Identity matrix
I = np.eye(3)              # 3x3 identity

# Zeros and ones
zeros = np.zeros((2, 3))   # 2x3 zeros
ones = np.ones((2, 3))     # 2x3 ones

# Random matrix
random = np.random.rand(3, 3)  # 3x3 random [0, 1)
normal = np.random.randn(3, 3)  # 3x3 normal distribution

# Matrix properties
A.shape                    # (2, 2)
A.size                     # 4
A.ndim                     # 2 (dimensions)
```

---

## Statistics & Probability

### Descriptive Statistics

```python
import numpy as np
from scipy import stats

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Central tendency
np.mean(data)              # 5.5 (mean)
np.median(data)            # 5.5 (median)
stats.mode(data)           # Mode

# Dispersion
np.var(data)               # 8.25 (variance)
np.std(data)               # 2.87 (standard deviation)
np.percentile(data, 25)    # 3.25 (Q1)
np.percentile(data, 75)    # 7.75 (Q3)
stats.iqr(data)            # 4.5 (interquartile range)

# Correlation
np.corrcoef(x, y)          # Correlation matrix
np.cov(x, y)               # Covariance matrix
```

### Probability Distributions

```python
from scipy import stats

# Normal distribution
normal = stats.norm(loc=0, scale=1)  # μ=0, σ=1
normal.pdf(0)              # Probability density at 0
normal.cdf(0)              # Cumulative probability at 0
normal.rvs(100)            # 100 random samples

# Binomial distribution
binom = stats.binom(n=10, p=0.5)
binom.pmf(5)               # P(X=5)
binom.cdf(5)               # P(X≤5)

# Uniform distribution
uniform = stats.uniform(loc=0, scale=1)
```

### Hypothesis Testing

```python
from scipy import stats

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, popmean=5)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Chi-square test
chi2, p_value = stats.chisquare(observed, expected)

# ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

---

## Calculus

### Derivatives

```python
from scipy.misc import derivative

# Numerical derivative
def f(x):
    return x**2

derivative(f, 1.0, dx=1e-6)  # Derivative at x=1

# Gradient (for multivariate)
from scipy.optimize import approx_fprime
gradient = approx_fprime(x0, f, epsilon=1e-6)
```

### Common Derivatives

| Function | Derivative |
|----------|------------|
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = \frac{1}{x}$ |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ |
| $f(x) = \cos(x)$ | $f'(x) = -\sin(x)$ |

### Chain Rule

$$(f(g(x)))' = f'(g(x)) \cdot g'(x)$$

### Product Rule

$$(f(x) \cdot g(x))' = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

### Quotient Rule

$$\left(\frac{f(x)}{g(x)}\right)' = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{g(x)^2}$$

---

## Time Complexity

### Big O Notation

| Notation | Name | Example |
|----------|------|---------|
| $O(1)$ | Constant | Array access |
| $O(\log n)$ | Logarithmic | Binary search |
| $O(n)$ | Linear | Single loop |
| $O(n \log n)$ | Linearithmic | Merge sort |
| $O(n^2)$ | Quadratic | Nested loops |
| $O(2^n)$ | Exponential | Recursive Fibonacci |
| $O(n!)$ | Factorial | Permutations |

### Common Operations

```python
# O(1) - Constant
arr[0]                      # Array access
dict["key"]                 # Dictionary lookup

# O(log n) - Logarithmic
# Binary search on sorted array

# O(n) - Linear
for item in list:           # Single loop
    process(item)

# O(n log n) - Linearithmic
sorted(list)                # Sorting

# O(n²) - Quadratic
for i in range(n):
    for j in range(n):
        process(i, j)
```

### Space Complexity

```python
# O(1) - Constant space
x = 5

# O(n) - Linear space
arr = [0] * n

# O(n²) - Quadratic space
matrix = [[0] * n for _ in range(n)]
```

---

## Quick Reference

### Python Built-in Functions

```python
len(obj)                    # Length
range(start, stop, step)    # Range generator
enumerate(iterable)         # Index and value
zip(*iterables)             # Combine iterables
map(func, iterable)         # Apply function
filter(func, iterable)      # Filter elements
sorted(iterable)            # Sort
sum(iterable)               # Sum
max(iterable)               # Maximum
min(iterable)               # Minimum
```

### NumPy Essentials

```python
import numpy as np

np.array([1, 2, 3])         # Create array
np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)        # [0, 0.25, 0.5, 0.75, 1]
np.zeros((3, 3))            # 3x3 zeros
np.ones((3, 3))             # 3x3 ones
np.random.rand(3, 3)        # Random [0, 1)
np.random.randn(3, 3)       # Normal distribution
```

---

**Remember**: Master these fundamentals before diving into ML! They form the foundation for understanding algorithms and models.

