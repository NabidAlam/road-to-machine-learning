# Essential Math Formulas for Machine Learning

Quick reference guide for mathematical formulas commonly used in machine learning and data science.

## Table of Contents

- [Statistics](#statistics)
- [Probability](#probability)
- [Linear Algebra](#linear-algebra)
- [Calculus](#calculus)
- [Information Theory](#information-theory)
- [ML-Specific Formulas](#ml-specific-formulas)

---

## Statistics

### Descriptive Statistics

**Mean (Average)**
```
μ = (1/n) * Σx_i
```

**Median**
- Middle value when data is sorted
- For even n: average of two middle values

**Mode**
- Most frequently occurring value

**Variance**
```
σ² = (1/n) * Σ(x_i - μ)²
```

**Standard Deviation**
```
σ = √σ² = √[(1/n) * Σ(x_i - μ)²]
```

**Covariance**
```
Cov(X,Y) = (1/n) * Σ(x_i - μ_x)(y_i - μ_y)
```

**Correlation Coefficient**
```
r = Cov(X,Y) / (σ_x * σ_y)
Range: -1 to 1
```

### Sampling

**Sample Mean**
```
x̄ = (1/n) * Σx_i
```

**Sample Variance**
```
s² = (1/(n-1)) * Σ(x_i - x̄)²
```

**Standard Error**
```
SE = σ / √n
```

---

## Probability

### Basic Probability

**Probability of Event A**
```
P(A) = Number of favorable outcomes / Total outcomes
```

**Conditional Probability**
```
P(A|B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem**
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**Independence**
```
P(A ∩ B) = P(A) * P(B)  (if independent)
```

### Probability Distributions

**Normal Distribution PDF**
```
f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
```

**Binomial Distribution**
```
P(k; n, p) = C(n,k) * p^k * (1-p)^(n-k)
where C(n,k) = n! / (k!(n-k)!)
```

**Poisson Distribution**
```
P(k; λ) = (λ^k * e^(-λ)) / k!
```

---

## Linear Algebra

### Vectors

**Dot Product**
```
a · b = Σ(a_i * b_i) = |a||b|cos(θ)
```

**Vector Norm (L2)**
```
||x||₂ = √(Σx_i²)
```

**Vector Norm (L1)**
```
||x||₁ = Σ|x_i|
```

**Cosine Similarity**
```
cos(θ) = (a · b) / (||a|| * ||b||)
```

### Matrices

**Matrix Multiplication**
```
C = AB where C_ij = Σ(A_ik * B_kj)
```

**Matrix Transpose**
```
(A^T)_ij = A_ji
```

**Matrix Inverse**
```
A^(-1) such that A * A^(-1) = I
```

**Determinant (2x2)**
```
det(A) = ad - bc for A = [[a,b],[c,d]]
```

**Eigenvalues and Eigenvectors**
```
Av = λv
where λ is eigenvalue, v is eigenvector
```

**Trace**
```
tr(A) = ΣA_ii (sum of diagonal elements)
```

---

## Calculus

### Derivatives

**Power Rule**
```
d/dx(x^n) = nx^(n-1)
```

**Product Rule**
```
d/dx(fg) = f'g + fg'
```

**Chain Rule**
```
d/dx(f(g(x))) = f'(g(x)) * g'(x)
```

**Common Derivatives**
```
d/dx(e^x) = e^x
d/dx(ln(x)) = 1/x
d/dx(sin(x)) = cos(x)
d/dx(cos(x)) = -sin(x)
```

### Gradients

**Gradient (Multivariate)**
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_n]
```

**Gradient Descent Update**
```
θ_new = θ_old - α * ∇J(θ)
where α is learning rate
```

### Partial Derivatives

**Chain Rule for Partial Derivatives**
```
∂f/∂x = (∂f/∂u)(∂u/∂x) + (∂f/∂v)(∂v/∂x)
```

---

## Information Theory

### Entropy

**Shannon Entropy**
```
H(X) = -Σ P(x_i) * log₂(P(x_i))
```

**Cross-Entropy**
```
H(P,Q) = -Σ P(x_i) * log(Q(x_i))
```

**KL Divergence**
```
D_KL(P||Q) = Σ P(x_i) * log(P(x_i)/Q(x_i))
```

**Mutual Information**
```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

---

## ML-Specific Formulas

### Regression Metrics

**Mean Squared Error (MSE)**
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**Root Mean Squared Error (RMSE)**
```
RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]
```

**Mean Absolute Error (MAE)**
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**R² (Coefficient of Determination)**
```
R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(y_true - y_pred)²
      SS_tot = Σ(y_true - y_mean)²
```

### Classification Metrics

**Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)**
```
Recall = TP / (TP + FN)
```

**Specificity**
```
Specificity = TN / (TN + FP)
```

**F1-Score**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**F-Beta Score**
```
F_β = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```

### Loss Functions

**Binary Cross-Entropy**
```
L = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

**Categorical Cross-Entropy**
```
L = -(1/n) * Σ Σ y_i * log(ŷ_i)
```

**Hinge Loss (SVM)**
```
L = max(0, 1 - y * (w·x + b))
```

### Regularization

**L1 Regularization (Lasso)**
```
L = Loss + λ * Σ|w_i|
```

**L2 Regularization (Ridge)**
```
L = Loss + λ * Σw_i²
```

**Elastic Net**
```
L = Loss + λ₁ * Σ|w_i| + λ₂ * Σw_i²
```

### Linear Regression

**Normal Equation**
```
θ = (X^T * X)^(-1) * X^T * y
```

**Gradient for Linear Regression**
```
∇J(θ) = (1/m) * X^T * (Xθ - y)
```

### Logistic Regression

**Sigmoid Function**
```
σ(z) = 1 / (1 + e^(-z))
```

**Logistic Regression Prediction**
```
P(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

**Log Loss**
```
L = -(1/n) * Σ[y*log(σ(z)) + (1-y)*log(1-σ(z))]
```

### Neural Networks

**Forward Propagation**
```
a^(l) = σ(W^(l) * a^(l-1) + b^(l))
```

**Backpropagation (Output Layer)**
```
δ^(L) = ∇_a C ⊙ σ'(z^(L))
```

**Backpropagation (Hidden Layers)**
```
δ^(l) = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^(l))
```

**Weight Update**
```
W^(l) = W^(l) - α * (1/m) * δ^(l) * (a^(l-1))^T
```

### Decision Trees

**Gini Impurity**
```
Gini = 1 - Σ(p_i)²
```

**Entropy (Information Gain)**
```
Entropy = -Σ p_i * log₂(p_i)
```

**Information Gain**
```
IG = Entropy(parent) - Σ (n_i/n) * Entropy(child_i)
```

### Clustering

**Within-Cluster Sum of Squares (WCSS)**
```
WCSS = Σ Σ ||x_i - μ_j||²
```

**Silhouette Score**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
where a(i) = average distance to points in same cluster
      b(i) = average distance to points in nearest other cluster
```

### Dimensionality Reduction

**PCA - Eigenvalue Decomposition**
```
C = (1/n) * X^T * X
C * v = λ * v
```

**PCA - Projection**
```
Y = X * W
where W contains eigenvectors
```

### Time Series

**Autocorrelation**
```
r_k = Σ[(x_t - x̄)(x_(t-k) - x̄)] / Σ(x_t - x̄)²
```

**Moving Average**
```
MA(n) = (1/n) * Σ(x_(t-i)) for i=0 to n-1
```

---

## Quick Reference

### Common Constants
- e ≈ 2.71828
- π ≈ 3.14159
- log₂(2) = 1
- ln(e) = 1

### Useful Identities
- log(ab) = log(a) + log(b)
- log(a/b) = log(a) - log(b)
- log(a^b) = b*log(a)
- e^(ln(x)) = x
- ln(e^x) = x

### Matrix Properties
- (AB)^T = B^T * A^T
- (A^T)^T = A
- (A^(-1))^(-1) = A
- det(AB) = det(A) * det(B)
- tr(AB) = tr(BA)

---

**Note**: This is a reference guide. Understanding the concepts behind these formulas is more important than memorizing them. Practice applying these formulas in context!

