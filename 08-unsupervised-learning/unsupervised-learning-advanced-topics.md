# Advanced Unsupervised Learning Topics

Comprehensive guide to advanced unsupervised learning techniques and applications.

## Table of Contents

- [Advanced Clustering Methods](#advanced-clustering-methods)
- [Advanced Dimensionality Reduction](#advanced-dimensionality-reduction)
- [Advanced Anomaly Detection](#advanced-anomaly-detection)
- [Association Rules and Market Basket Analysis](#association-rules-and-market-basket-analysis)
- [Evaluation Without Labels](#evaluation-without-labels)
- [Semi-Supervised Learning](#semi-supervised-learning)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Clustering Methods

### Gaussian Mixture Models (GMM)

Probabilistic clustering that assumes data comes from a mixture of Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture

# GMM clustering
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
clusters = gmm.fit_predict(X_scaled)

# Get probabilities
probabilities = gmm.predict_proba(X_scaled)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('GMM Clustering', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Show uncertainty (probability of belonging to cluster)
plt.subplot(1, 2, 2)
max_probs = np.max(probabilities, axis=1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=max_probs, cmap='coolwarm', alpha=0.6)
plt.colorbar(label='Max Probability')
plt.title('Cluster Assignment Confidence', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Model selection using AIC/BIC
n_components_range = range(1, 11)
aics = []
bics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    aics.append(gmm.aic(X_scaled))
    bics.append(gmm.bic(X_scaled))

optimal_n_aic = n_components_range[np.argmin(aics)]
optimal_n_bic = n_components_range[np.argmin(bics)]

print(f"Optimal components (AIC): {optimal_n_aic}")
print(f"Optimal components (BIC): {optimal_n_bic}")
```

### Spectral Clustering

Uses eigenvalues of similarity matrix. Good for non-convex clusters.

```python
from sklearn.cluster import SpectralClustering

# Spectral clustering
spectral = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42)
clusters = spectral.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Spectral Clustering', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Mean Shift

Non-parametric clustering that finds modes in density.

```python
from sklearn.cluster import MeanShift

# Mean Shift
meanshift = MeanShift(bandwidth=2.0)
clusters = meanshift.fit_predict(X_scaled)

n_clusters = len(np.unique(clusters))
print(f"Number of clusters found: {n_clusters}")

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Mean Shift Clustering', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Clustering Comparison

```python
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN,
                            GaussianMixture, SpectralClustering)

clustering_methods = {
    'K-Means': KMeans(n_clusters=3, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'GMM': GaussianMixture(n_components=3, random_state=42),
    'Spectral': SpectralClustering(n_clusters=3, random_state=42)
}

results = {}

for name, method in clustering_methods.items():
    if name == 'GMM':
        clusters = method.fit_predict(X_scaled)
    else:
        clusters = method.fit_predict(X_scaled)
    
    if len(set(clusters)) > 1:  # Valid clustering
        silhouette = silhouette_score(X_scaled, clusters)
        results[name] = {
            'clusters': clusters,
            'silhouette': silhouette,
            'n_clusters': len(set(clusters))
        }
        print(f"{name:15s}: Silhouette = {silhouette:.3f}, Clusters = {results[name]['n_clusters']}")
```

---

## Advanced Dimensionality Reduction

### Singular Value Decomposition (SVD)

SVD is a fundamental matrix decomposition technique that underlies PCA and has many applications in machine learning.

#### What is SVD?

SVD decomposes any matrix A (m×n) into three matrices:
- **U**: Left singular vectors (m×m, orthogonal)
- **Σ (Sigma)**: Singular values (diagonal matrix, m×n)
- **V^T**: Right singular vectors (n×n, orthogonal)

**Mathematical Formulation:**
```
A = U × Σ × V^T
```

#### Intuition of SVD

**Geometric Interpretation:**
- SVD finds the best low-rank approximation of a matrix
- U: Rotation/reflection in input space
- Σ: Scaling along principal axes
- V^T: Rotation/reflection in output space

**For Non-Square Matrices:**
- Rectangular matrices don't have eigenvalues
- SVD works for any matrix (square or rectangular)
- Provides similar insights to eigendecomposition

#### Relationship with Eigen Decomposition

**For Square Symmetric Matrices:**
- SVD and eigendecomposition are related
- If A is symmetric: A = U × Σ × U^T (eigendecomposition)
- Singular values = absolute values of eigenvalues

**For PCA:**
- PCA can be computed using SVD
- SVD of centered data matrix = PCA
- More numerically stable than eigendecomposition

#### How to Calculate SVD

**Step-by-Step Process:**

```python
import numpy as np
from scipy.linalg import svd

# Example matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute SVD
U, s, Vt = svd(A, full_matrices=False)

# s is 1D array of singular values
# Convert to diagonal matrix
Sigma = np.diag(s)

print("Original Matrix A:")
print(A)
print(f"\nShape: {A.shape}")

print("\nU (Left singular vectors):")
print(U)
print(f"Shape: {U.shape}")

print("\nSingular Values (s):")
print(s)

print("\nSigma (Diagonal matrix):")
print(Sigma)
print(f"Shape: {Sigma.shape}")

print("\nV^T (Right singular vectors, transposed):")
print(Vt)
print(f"Shape: {Vt.shape}")

# Reconstruct A
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed A:")
print(A_reconstructed)
print(f"\nReconstruction error: {np.abs(A - A_reconstructed).max():.2e}")
```

#### SVD in PCA

**Connection:**
- PCA can be computed via SVD of the centered data matrix
- More numerically stable than computing covariance matrix

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.random.randn(100, 10)

# Center the data
X_centered = X - X.mean(axis=0)

# Method 1: PCA (uses SVD internally)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_centered)

# Method 2: Direct SVD
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Principal components from SVD
# Vt contains the principal components (eigenvectors of covariance)
# s contains singular values (related to eigenvalues)
principal_components_svd = Vt[:5].T  # First 5 components

# Project data using SVD
X_svd = X_centered @ principal_components_svd

print("PCA components shape:", pca.components_.shape)
print("SVD components shape:", principal_components_svd.shape)
print("\nComponents match:", np.allclose(pca.components_.T, principal_components_svd, atol=1e-10))
```

#### Low-Rank Approximation

**Key Application:** SVD provides the best low-rank approximation of a matrix.

```python
# Create a matrix with some structure
np.random.seed(42)
A = np.random.randn(50, 30)

# Add some structure (rank-5 matrix + noise)
true_rank = 5
U_true = np.random.randn(50, true_rank)
V_true = np.random.randn(true_rank, 30)
A_structured = U_true @ V_true + 0.1 * np.random.randn(50, 30)

# Compute SVD
U, s, Vt = np.linalg.svd(A_structured, full_matrices=False)

# Low-rank approximations
ranks = [1, 3, 5, 10, 20]
approximation_errors = []

for rank in ranks:
    # Truncate SVD
    U_k = U[:, :rank]
    s_k = s[:rank]
    Vt_k = Vt[:rank, :]
    
    # Reconstruct
    A_approx = U_k @ np.diag(s_k) @ Vt_k
    
    # Error
    error = np.linalg.norm(A_structured - A_approx, 'fro')
    approximation_errors.append(error)
    
    print(f"Rank {rank:2d} approximation error: {error:.4f}")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(ranks, approximation_errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Approximation Error (Frobenius norm)', fontsize=12)
plt.title('SVD Low-Rank Approximation', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Applications of SVD

**1. Image Compression:**
```python
from PIL import Image
import numpy as np

# Load image
img = Image.open('image.jpg')
img_array = np.array(img.convert('L'))  # Convert to grayscale

# SVD
U, s, Vt = np.linalg.svd(img_array, full_matrices=False)

# Compress with different ranks
ranks = [10, 50, 100, 200]
fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(15, 3))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original', fontsize=10)
axes[0].axis('off')

for idx, rank in enumerate(ranks, 1):
    # Reconstruct with rank-k approximation
    U_k = U[:, :rank]
    s_k = s[:rank]
    Vt_k = Vt[:rank, :]
    img_compressed = U_k @ np.diag(s_k) @ Vt_k
    
    axes[idx].imshow(img_compressed, cmap='gray')
    axes[idx].set_title(f'Rank {rank}\nCompression: {rank/img_array.shape[1]*100:.1f}%', fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

**2. Recommender Systems:**
```python
# User-item matrix (ratings)
ratings = np.array([
    [5, 4, 0, 0, 1],
    [4, 5, 3, 0, 2],
    [0, 0, 5, 4, 3],
    [0, 0, 4, 5, 4],
    [1, 2, 3, 4, 5]
])

# SVD for matrix factorization
U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

# Low-rank approximation (latent factors)
k = 2
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Reconstructed ratings (predictions)
ratings_pred = U_k @ np.diag(s_k) @ Vt_k

print("Original ratings:")
print(ratings)
print("\nPredicted ratings (SVD approximation):")
print(ratings_pred.round(2))
```

**3. Noise Reduction:**
```python
# Signal with noise
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
noise = 0.3 * np.random.randn(1000)
noisy_signal = signal + noise

# Create matrix from signal (sliding window)
window_size = 50
matrix = np.array([noisy_signal[i:i+window_size] 
                   for i in range(len(noisy_signal) - window_size + 1)])

# SVD
U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

# Keep only top singular values (remove noise)
k = 5  # Keep top 5 components
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Reconstruct
matrix_denoised = U_k @ np.diag(s_k) @ Vt_k

# Extract denoised signal
denoised_signal = matrix_denoised[:, window_size // 2]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t[:len(signal)], signal, 'b-', label='Original Signal', linewidth=2)
plt.plot(t[:len(noisy_signal)], noisy_signal, 'r-', alpha=0.5, label='Noisy Signal')
plt.plot(t[:len(denoised_signal)], denoised_signal, 'g-', label='Denoised (SVD)', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title('Signal Denoising with SVD', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### SVD vs PCA

| Aspect | SVD | PCA |
|--------|-----|-----|
| **Input** | Any matrix | Data matrix |
| **Method** | Matrix decomposition | Covariance eigendecomposition |
| **Stability** | More numerically stable | Can be unstable |
| **Computation** | Direct | Via SVD internally |
| **Applications** | General matrix operations | Dimensionality reduction |

**Key Insight:** PCA is essentially SVD applied to the centered data matrix.

#### Properties of SVD

**1. Uniqueness:**
- Singular values are unique (up to ordering)
- Singular vectors are unique up to sign

**2. Optimality:**
- Best low-rank approximation (Eckart-Young theorem)
- Minimizes Frobenius norm error

**3. Rank:**
- Rank of matrix = number of non-zero singular values
- Numerical rank = number of "significant" singular values

```python
# Find numerical rank
A = np.random.randn(100, 100)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# True rank (all singular values)
true_rank = np.sum(s > 1e-10)

# Numerical rank (significant singular values)
tolerance = s[0] * 1e-6  # Relative tolerance
numerical_rank = np.sum(s > tolerance)

print(f"True rank: {true_rank}")
print(f"Numerical rank: {numerical_rank}")
print(f"First 10 singular values: {s[:10]}")
```

#### Key Takeaways

1. **SVD decomposes any matrix** into U, Σ, V^T
2. **PCA uses SVD internally** - more stable than eigendecomposition
3. **Low-rank approximation** - SVD provides best approximation
4. **Applications**: Image compression, recommender systems, noise reduction
5. **Numerical stability** - SVD is preferred for large matrices

---

### LDA (Linear Discriminant Analysis)

Supervised dimensionality reduction that maximizes class separation.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA (requires labels - supervised!)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Visualize
plt.figure(figsize=(10, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], 
                label=f'Class {label}', alpha=0.6)
plt.xlabel('First LDA Component', fontsize=12)
plt.ylabel('Second LDA Component', fontsize=12)
plt.title('LDA Visualization', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Explained variance ratio
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")

# Compare with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Note: LDA is supervised (needs labels), unlike PCA
# LDA maximizes class separation, PCA maximizes variance
```

### Kernel PCA

Non-linear PCA using kernel trick.

```python
from sklearn.decomposition import KernelPCA

# Kernel PCA with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, kernel in enumerate(kernels):
    kpca = KernelPCA(n_components=2, kernel=kernel, random_state=42)
    X_kpca = kpca.fit_transform(X_scaled)
    
    axes[idx].scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6, s=50)
    axes[idx].set_title(f'Kernel PCA ({kernel})', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Independent Component Analysis (ICA)

Finds statistically independent components.

```python
from sklearn.decomposition import FastICA

# ICA
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_ica[:, 0], X_ica[:, 1], alpha=0.6, s=50)
plt.xlabel('Independent Component 1', fontsize=12)
plt.ylabel('Independent Component 2', fontsize=12)
plt.title('Independent Component Analysis', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Non-negative Matrix Factorization (NMF)

Factorizes matrix into non-negative components.

```python
from sklearn.decomposition import NMF

# NMF (requires non-negative data)
X_non_neg = X_scaled - X_scaled.min() + 1  # Make non-negative

nmf = NMF(n_components=2, random_state=42, max_iter=1000)
X_nmf = nmf.fit_transform(X_non_neg)

plt.figure(figsize=(10, 6))
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], alpha=0.6, s=50)
plt.xlabel('NMF Component 1', fontsize=12)
plt.ylabel('NMF Component 2', fontsize=12)
plt.title('Non-negative Matrix Factorization', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Advanced Anomaly Detection

### Autoencoders for Anomaly Detection

Neural network-based anomaly detection.

```python
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Build autoencoder
    input_dim = X_scaled.shape[1]
    encoding_dim = 2
    
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoder = layers.Dense(input_dim, activation='sigmoid')(encoder)
    
    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
    
    # Reconstruct
    X_reconstructed = autoencoder.predict(X_scaled)
    
    # Anomaly score = reconstruction error
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    
    # Threshold (e.g., 95th percentile)
    threshold = np.percentile(reconstruction_error, 95)
    outliers = reconstruction_error > threshold
    
    print(f"Autoencoder detected {outliers.sum()} outliers")
    
except ImportError:
    print("Install TensorFlow: pip install tensorflow")
```

### Elliptic Envelope

Assumes normal distribution and finds outliers.

```python
from sklearn.covariance import EllipticEnvelope

# Elliptic Envelope
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
outliers = elliptic.fit_predict(X_scaled)

n_outliers = (outliers == -1).sum()
print(f"Elliptic Envelope detected {n_outliers} outliers")
```

### Ensemble Anomaly Detection

Combine multiple methods.

```python
from sklearn.ensemble import VotingClassifier

# Multiple anomaly detection methods
methods = {
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'LOF': LocalOutlierFactor(contamination=0.1, n_neighbors=20),
    'One-Class SVM': OneClassSVM(nu=0.1, kernel='rbf')
}

# Get predictions
predictions = {}
for name, method in methods.items():
    pred = method.fit_predict(X_scaled)
    predictions[name] = (pred == -1).astype(int)  # 1 = outlier, 0 = normal

# Ensemble: majority vote
ensemble_pred = np.sum(list(predictions.values()), axis=0)
ensemble_outliers = (ensemble_pred >= 2).astype(int)  # At least 2 methods agree

print(f"Ensemble detected {ensemble_outliers.sum()} outliers")
print(f"Agreement: {np.mean([np.mean(predictions[k] == ensemble_outliers) for k in predictions]):.2%}")
```

---

## Association Rules and Market Basket Analysis

### Apriori Algorithm

Find frequent itemsets and association rules.

```python
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    
    # Example: Market basket data
    transactions = [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'eggs'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola']
    ]
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_transactions, min_support=0.4, use_colnames=True)
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
except ImportError:
    print("Install mlxtend: pip install mlxtend")
```

### FP-Growth Algorithm

Faster alternative to Apriori.

```python
try:
    from mlxtend.frequent_patterns import fpgrowth
    
    # FP-Growth
    frequent_itemsets_fp = fpgrowth(df_transactions, min_support=0.4, use_colnames=True)
    print("FP-Growth Frequent Itemsets:")
    print(frequent_itemsets_fp)
    
except ImportError:
    print("Install mlxtend: pip install mlxtend")
```

### Rule Evaluation Metrics

```python
# Support: P(A and B)
# Confidence: P(B|A) = P(A and B) / P(A)
# Lift: P(B|A) / P(B) = Confidence / P(B)

# High lift (> 1): Positive association
# Low lift (< 1): Negative association
# Lift = 1: Independent

rules_filtered = rules[
    (rules['lift'] > 1.0) & 
    (rules['confidence'] > 0.6)
].sort_values('lift', ascending=False)

print("Strong Association Rules:")
print(rules_filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

## Evaluation Without Labels

### Internal Validation Metrics

```python
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score, adjusted_rand_score)

def evaluate_clustering(X, labels):
    """Comprehensive clustering evaluation"""
    metrics = {}
    
    # Silhouette Score
    metrics['silhouette'] = silhouette_score(X, labels)
    
    # Calinski-Harabasz Score
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    
    # Davies-Bouldin Score
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    
    return metrics

# Evaluate different clusterings
for name, clusters in clustering_results.items():
    metrics = evaluate_clustering(X_scaled, clusters)
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
```

### Stability Analysis

```python
def stability_analysis(X, n_runs=10, n_clusters=3):
    """Analyze clustering stability"""
    all_labels = []
    
    for _ in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=None)
        labels = kmeans.fit_predict(X)
        all_labels.append(labels)
    
    # Calculate pairwise adjusted rand index
    stability_scores = []
    for i in range(len(all_labels)):
        for j in range(i+1, len(all_labels)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            stability_scores.append(ari)
    
    return np.mean(stability_scores), np.std(stability_scores)

mean_stability, std_stability = stability_analysis(X_scaled)
print(f"Clustering Stability: {mean_stability:.3f} (+/- {std_stability:.3f})")
```

---

## Semi-Supervised Learning

### Self-Training

Use labeled data to label unlabeled data.

```python
from sklearn.semi_supervised import SelfTrainingClassifier

# Assume we have some labeled data
n_labeled = 50
labeled_indices = np.random.choice(len(X_scaled), n_labeled, replace=False)
y_semi = np.full(len(X_scaled), -1)  # -1 = unlabeled
y_semi[labeled_indices] = y[labeled_indices]  # Some labeled examples

# Self-training
base_classifier = LogisticRegression(random_state=42, max_iter=1000)
self_training = SelfTrainingClassifier(base_classifier, threshold=0.8)
self_training.fit(X_scaled, y_semi)

# Predictions
predictions = self_training.predict(X_scaled)
print(f"Accuracy: {accuracy_score(y, predictions):.3f}")
```

### Label Propagation

Propagate labels through similarity graph.

```python
from sklearn.semi_supervised import LabelPropagation

# Label propagation
label_prop = LabelPropagation(kernel='rbf', gamma=0.1)
label_prop.fit(X_scaled, y_semi)

predictions = label_prop.predict(X_scaled)
print(f"Label Propagation Accuracy: {accuracy_score(y, predictions):.3f}")
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Not Scaling Features

**Problem**: Features on different scales bias clustering

**Solution**:
```python
# Always scale before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)  # Not X!
```

### Pitfall 2: Choosing Wrong k

**Problem**: Arbitrary choice of k leads to poor clusters

**Solution**:
```python
# Use multiple methods
# 1. Elbow method
# 2. Silhouette score
# 3. Gap statistic
# 4. Domain knowledge
```

### Pitfall 3: Ignoring Cluster Quality

**Problem**: Assuming clusters are good without validation

**Solution**:
```python
# Always evaluate
silhouette = silhouette_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)
davies = davies_bouldin_score(X_scaled, clusters)

# Visualize clusters
# Check cluster sizes
# Validate with domain experts
```

### Pitfall 4: Overfitting in Anomaly Detection

**Problem**: Too many false positives or negatives

**Solution**:
```python
# Tune contamination parameter
# Use multiple methods
# Validate with domain experts
# Consider context (temporal, spatial)
```

### Pitfall 5: Misinterpreting t-SNE

**Problem**: Using t-SNE for feature reduction

**Solution**:
```python
# t-SNE is for visualization only
# Use PCA or UMAP for feature reduction
# t-SNE distances don't preserve global structure
```

---

## Key Takeaways

1. **Advanced Clustering**: GMM, Spectral, Mean Shift for complex data
2. **Advanced Dimensionality Reduction**: Kernel PCA, ICA, NMF for non-linear data
3. **Advanced Anomaly Detection**: Autoencoders, ensemble methods
4. **Association Rules**: Market basket analysis with Apriori/FP-Growth
5. **Evaluation**: Use multiple internal metrics when no labels
6. **Semi-Supervised**: Leverage unlabeled data with few labels
7. **Avoid Pitfalls**: Scale features, validate clusters, interpret correctly

---

## Next Steps

- Practice with real-world datasets
- Experiment with advanced methods
- Learn about deep clustering
- Explore graph-based clustering
- Move to neural networks module

**Remember**: Advanced methods require more computation but can handle complex patterns!

