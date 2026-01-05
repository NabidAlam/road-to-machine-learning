# Unsupervised Learning Quick Reference Guide

Quick reference for unsupervised learning algorithms, metrics, and best practices.

## Table of Contents

- [Algorithm Selection](#algorithm-selection)
- [Code Snippets](#code-snippets)
- [Clustering Methods](#clustering-methods)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Anomaly Detection](#anomaly-detection)
- [Evaluation Metrics](#evaluation-metrics)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Algorithm Selection

### Quick Decision Tree

```
Need unsupervised learning?
│
├─ Want to group similar data?
│  ├─ Know number of groups? → K-Means
│  ├─ Don't know number? → Hierarchical or DBSCAN
│  ├─ Non-spherical clusters? → DBSCAN or Spectral
│  └─ Need probabilities? → Gaussian Mixture Model
│
├─ Want to reduce dimensions?
│  ├─ Linear relationships? → PCA
│  ├─ Non-linear, visualization? → t-SNE or UMAP
│  └─ Need to transform new data? → PCA or UMAP
│
└─ Want to find outliers?
   ├─ High-dimensional data? → Isolation Forest
   ├─ Local anomalies? → LOF
   └─ Need probabilities? → One-Class SVM
```

### Clustering Comparison

| Algorithm | When to Use | Pros | Cons | Code |
|-----------|-------------|------|------|------|
| **K-Means** | Known k, spherical clusters | Fast, simple, scalable | Requires k, assumes spherical | `KMeans(n_clusters=k)` |
| **Hierarchical** | Unknown k, need dendrogram | No k needed, interpretable | Slow, O(n³) | `AgglomerativeClustering()` |
| **DBSCAN** | Unknown k, arbitrary shapes | Finds outliers, no k needed | Sensitive to parameters | `DBSCAN(eps, min_samples)` |
| **GMM** | Need probabilities | Probabilistic, soft clustering | Assumes Gaussian | `GaussianMixture()` |
| **Spectral** | Non-convex clusters | Handles complex shapes | Slow, memory intensive | `SpectralClustering()` |

---

## Code Snippets

### Basic Clustering Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Scale features (CRITICAL!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.3f}")
```

### Find Optimal k

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method + Silhouette
inertias = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose k with highest silhouette
optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
```

---

## Clustering Methods

### K-Means

```python
from sklearn.cluster import KMeans

# Basic
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Get centroids
centroids = kmeans.cluster_centers_

# Get inertia
inertia = kmeans.inertia_
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = hierarchical.fit_predict(X_scaled)

# Dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.show()
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal eps
neighbors = NearestNeighbors(n_neighbors=4)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, 3], axis=0)
optimal_eps = np.percentile(distances, 90)

# DBSCAN
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# -1 = noise/outliers
n_outliers = (clusters == -1).sum()
```

### Gaussian Mixture Model

```python
from sklearn.mixture import GaussianMixture

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X_scaled)

# Get probabilities
probabilities = gmm.predict_proba(X_scaled)

# Model selection
aics = []
for n in range(1, 11):
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    aics.append(gmm.aic(X_scaled))

optimal_n = np.argmin(aics) + 1
```

---

## Dimensionality Reduction

### PCA

```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum():.3f}")

# Find components for 95% variance
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumulative >= 0.95) + 1
```

### t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE (for visualization only!)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Note: Can't transform new data easily
```

### UMAP

```python
try:
    import umap
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)
    
    # Can transform new data
    X_new_umap = umap_reducer.transform(X_new)
except ImportError:
    print("Install: pip install umap-learn")
```

### When to Use

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **PCA** | Linear reduction, feature engineering | Fast, interpretable, can transform | Assumes linearity |
| **t-SNE** | Visualization only | Great visualization, preserves local | Slow, can't transform |
| **UMAP** | Non-linear reduction | Fast, preserves structure, can transform | Less interpretable |

---

## Anomaly Detection

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)

# -1 = outlier, 1 = normal
n_outliers = (outliers == -1).sum()

# Anomaly scores
scores = iso_forest.score_samples(X_scaled)
```

### Local Outlier Factor

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
outliers = lof.fit_predict(X_scaled)

# Anomaly scores
scores = -lof.negative_outlier_factor_
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel='rbf')
outliers = oc_svm.fit_predict(X_scaled)
```

### Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Isolation Forest** | High-dim, fast | Fast, scalable | Less interpretable |
| **LOF** | Local anomalies | Detects local outliers | Sensitive to parameters |
| **One-Class SVM** | Need probabilities | Probabilistic | Slow for large data |

---

## Evaluation Metrics

### Internal Metrics (No Labels)

```python
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score)

# Silhouette Score: -1 to 1 (higher is better)
silhouette = silhouette_score(X_scaled, clusters)

# Calinski-Harabasz: Higher is better
calinski = calinski_harabasz_score(X_scaled, clusters)

# Davies-Bouldin: Lower is better
davies = davies_bouldin_score(X_scaled, clusters)
```

### External Metrics (With Labels)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Adjusted Rand Index: -1 to 1 (1 = perfect)
ari = adjusted_rand_score(true_labels, clusters)

# Normalized Mutual Information: 0 to 1 (1 = perfect)
nmi = normalized_mutual_info_score(true_labels, clusters)
```

### Metric Comparison

| Metric | Range | Best Value | Use Case |
|--------|-------|------------|----------|
| **Silhouette** | -1 to 1 | 1 | General clustering quality |
| **Calinski-Harabasz** | 0 to ∞ | Higher | Between vs within variance |
| **Davies-Bouldin** | 0 to ∞ | 0 | Cluster separation |
| **ARI** | -1 to 1 | 1 | Compare with ground truth |
| **NMI** | 0 to 1 | 1 | Compare with ground truth |

---

## Common Issues & Solutions

### Issue 1: Not Scaling Features

**Problem**: Features on different scales bias clustering

**Solution**:
```python
# ALWAYS scale before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)  # Not X!
```

### Issue 2: Wrong Number of Clusters

**Problem**: Arbitrary k leads to poor clusters

**Solution**:
```python
# Use multiple methods
# 1. Elbow method
# 2. Silhouette score
# 3. Gap statistic
# 4. Domain knowledge
```

### Issue 3: Poor Cluster Quality

**Problem**: Clusters don't make sense

**Solution**:
```python
# Evaluate with multiple metrics
silhouette = silhouette_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)

# Visualize clusters
# Check cluster sizes
# Validate with domain experts
```

### Issue 4: Too Many/Few Outliers

**Problem**: Contamination parameter wrong

**Solution**:
```python
# Tune contamination based on domain knowledge
# Try multiple values: 0.01, 0.05, 0.1, 0.2
# Validate with domain experts
# Use ensemble of methods
```

### Issue 5: DBSCAN Finds Only One Cluster

**Problem**: eps too large

**Solution**:
```python
# Use k-NN distance plot to find optimal eps
neighbors = NearestNeighbors(n_neighbors=4)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, 3], axis=0)
optimal_eps = np.percentile(distances, 90)  # Try different percentiles
```

### Issue 6: t-SNE Misinterpretation

**Problem**: Using t-SNE for feature reduction

**Solution**:
```python
# t-SNE is for VISUALIZATION ONLY
# Use PCA or UMAP for feature reduction
# t-SNE distances don't preserve global structure
```

---

## Best Practices Checklist

### Data Preparation
- [ ] Scale all features before clustering
- [ ] Handle missing values
- [ ] Remove or handle outliers appropriately
- [ ] Check for multicollinearity

### Clustering
- [ ] Use multiple methods to find optimal k
- [ ] Evaluate with multiple metrics
- [ ] Visualize clusters
- [ ] Check cluster sizes (avoid tiny clusters)
- [ ] Validate with domain knowledge
- [ ] Document cluster characteristics

### Dimensionality Reduction
- [ ] Scale before PCA
- [ ] Check explained variance
- [ ] Use t-SNE only for visualization
- [ ] Use UMAP for non-linear reduction
- [ ] Don't use t-SNE for feature engineering

### Anomaly Detection
- [ ] Set contamination based on domain knowledge
- [ ] Use multiple methods
- [ ] Validate anomalies with experts
- [ ] Consider context (temporal, spatial)
- [ ] Balance false positives/negatives

### Evaluation
- [ ] Use internal metrics when no labels
- [ ] Use external metrics when labels available
- [ ] Compare multiple methods
- [ ] Visualize results
- [ ] Document findings

### Interpretation
- [ ] Analyze cluster characteristics
- [ ] Generate actionable insights
- [ ] Validate with domain experts
- [ ] Document business implications

---

## Quick Code Templates

### Complete Clustering Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 1. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Find optimal k
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_k = range(2, 11)[np.argmax(silhouette_scores)]

# 3. Cluster
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 4. Evaluate
silhouette = silhouette_score(X_scaled, clusters)
print(f"Optimal k: {optimal_k}, Silhouette: {silhouette:.3f}")
```

### Anomaly Detection Template

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Method 1: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(X_scaled)

# Method 2: LOF
lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
outliers_lof = lof.fit_predict(X_scaled)

# Compare
overlap = ((outliers_iso == -1) & (outliers_lof == -1)).sum()
print(f"Outliers detected by both: {overlap}")
```

### PCA Template

```python
from sklearn.decomposition import PCA

# Find optimal components
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumulative >= 0.95) + 1

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced from {X_scaled.shape[1]} to {n_components} components")
print(f"Explained variance: {cumulative[n_components-1]:.3f}")
```

---

## Key Takeaways

1. **Always scale features** before clustering and dimensionality reduction
2. **Use multiple methods** to find optimal parameters (k, eps, etc.)
3. **Evaluate with multiple metrics** when no ground truth
4. **Visualize results** to understand clusters and anomalies
5. **Validate with domain experts** - unsupervised learning is subjective
6. **t-SNE is for visualization only** - use PCA or UMAP for feature reduction
7. **Document everything** - interpretation is key in unsupervised learning

---

## Next Steps

- Practice with real datasets
- Experiment with different algorithms
- Learn advanced techniques
- Explore semi-supervised learning
- Move to next module

**Remember**: Unsupervised learning reveals hidden patterns - use domain knowledge to validate!

