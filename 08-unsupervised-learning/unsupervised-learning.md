# Unsupervised Learning Complete Guide

Comprehensive guide to finding patterns in unlabeled data.

## Table of Contents

- [Introduction](#introduction)
- [Clustering](#clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Anomaly Detection](#anomaly-detection)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is Unsupervised Learning?

Learning from unlabeled data - finding hidden patterns without guidance.

**Key Characteristics:**
- No target variable (labels)
- Discovers hidden structures
- Exploratory in nature
- Often used for data understanding

**When to use:**
- No labels available
- Exploratory data analysis
- Finding hidden structures
- Data compression
- Anomaly detection
- Feature extraction
- Data preprocessing

### Types of Unsupervised Learning

1. **Clustering**: Group similar data points
2. **Dimensionality Reduction**: Reduce feature space
3. **Anomaly Detection**: Find outliers
4. **Association Rules**: Find patterns in transactions

### Challenges

- No ground truth for evaluation
- Harder to validate results
- Requires domain knowledge
- Interpretation is subjective

---

## Clustering

### What is Clustering?

Grouping similar data points together without prior knowledge of groups.

**Applications:**
- Customer segmentation
- Image segmentation
- Document clustering
- Market research
- Anomaly detection

### K-Means

Partition data into k clusters by minimizing within-cluster variance.

**How it works:**
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Update centroids to cluster means
4. Repeat until convergence

**Pros:**
- Fast and scalable
- Simple to understand
- Works well with spherical clusters

**Cons:**
- Requires k to be specified
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Scale features (important for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
clusters = kmeans.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='x', s=200, c='red', linewidths=3, label='Centroids')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('K-Means Clustering', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Evaluate with multiple metrics
silhouette = silhouette_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)
davies = davies_bouldin_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette:.3f} (higher is better, range: -1 to 1)")
print(f"Calinski-Harabasz Score: {calinski:.2f} (higher is better)")
print(f"Davies-Bouldin Score: {davies:.3f} (lower is better)")

# Cluster statistics
plt.subplot(1, 2, 2)
unique, counts = np.unique(clusters, return_counts=True)
plt.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Points', fontsize=12)
plt.title('Cluster Sizes', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Choosing K (Optimal Number of Clusters)

**Methods to choose k:**

1. **Elbow Method**: Find the "elbow" in inertia plot
2. **Silhouette Score**: Maximize silhouette score
3. **Gap Statistic**: Compare with null reference
4. **Domain Knowledge**: Use business requirements

#### Elbow Method

```python
# Elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot both metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette score plot
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal k
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k (Silhouette): {optimal_k_silhouette}")
print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
```

#### Gap Statistic

```python
def calculate_gap_statistic(X, k_max=10, n_refs=10):
    """Calculate gap statistic for optimal k"""
    gaps = []
    
    for k in range(1, k_max + 1):
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        log_wk = np.log(kmeans.inertia_)
        
        # Generate reference datasets
        ref_log_wks = []
        for _ in range(n_refs):
            # Generate random data with same bounds
            random_data = np.random.uniform(
                low=X.min(axis=0),
                high=X.max(axis=0),
                size=X.shape
            )
            ref_kmeans = KMeans(n_clusters=k, random_state=None, n_init=1)
            ref_kmeans.fit(random_data)
            ref_log_wks.append(np.log(ref_kmeans.inertia_))
        
        # Calculate gap
        gap = np.mean(ref_log_wks) - log_wk
        gaps.append(gap)
    
    return gaps

# Calculate gap statistic
gaps = calculate_gap_statistic(X_scaled, k_max=10)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), gaps, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Gap Statistic', fontsize=12)
plt.title('Gap Statistic Method', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_k_gap = np.argmax(gaps) + 1
print(f"Optimal k (Gap Statistic): {optimal_k_gap}")
```

### Hierarchical Clustering

Builds a tree-like structure of clusters. Can be agglomerative (bottom-up) or divisive (top-down).

**Linkage Methods:**
- **Ward**: Minimizes variance (most common)
- **Complete**: Maximum distance between clusters
- **Average**: Average distance between clusters
- **Single**: Minimum distance (can create long chains)

**Pros:**
- No need to specify k
- Produces dendrogram for visualization
- Works with any cluster shape

**Cons:**
- Computationally expensive (O(n³))
- Sensitive to noise and outliers
- Difficult with large datasets

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster.fit_predict(X_scaled)

# Create linkage matrix for dendrogram
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Hierarchical Clustering Results', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Evaluate
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.3f}")

# Cut dendrogram at different levels
for n_clusters in [2, 3, 4, 5]:
    labels_cut = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    silhouette = silhouette_score(X_scaled, labels_cut)
    print(f"k={n_clusters}: Silhouette Score = {silhouette:.3f}")
```

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise. Finds clusters of varying shapes and identifies outliers.

**Parameters:**
- **eps**: Maximum distance between samples in same neighborhood
- **min_samples**: Minimum samples in neighborhood to form core point

**Point Types:**
- **Core Point**: Has at least min_samples neighbors within eps
- **Border Point**: Within eps of core point but has fewer neighbors
- **Noise Point**: Not a core point and not within eps of any core point

**Pros:**
- No need to specify k
- Finds clusters of arbitrary shape
- Identifies outliers automatically
- Robust to noise

**Cons:**
- Sensitive to eps and min_samples
- Struggles with varying densities
- Can be slow for large datasets

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Method to find optimal eps
def find_optimal_eps(X, k=4):
    """Use k-nearest neighbors to find optimal eps"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    return distances

# Find optimal eps
k_distances = find_optimal_eps(X_scaled, k=4)

plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
plt.xlabel('Points Sorted by Distance', fontsize=12)
plt.ylabel(f'k-NN Distance (k=4)', fontsize=12)
plt.title('k-NN Distance Plot for Optimal eps', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Choose eps from "knee" of the curve
optimal_eps = np.percentile(k_distances, 90)  # Example: 90th percentile
print(f"Suggested eps: {optimal_eps:.3f}")

# DBSCAN with optimal parameters
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
unique_labels = set(clusters)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black for noise
        col = 'black'
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (clusters == k)
    xy = X_scaled[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
               s=50, alpha=0.6, label=label, edgecolors='black', linewidths=0.5)

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('DBSCAN Clustering', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Cluster statistics
plt.subplot(1, 2, 2)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = (clusters == -1).sum()

stats = {
    'Number of Clusters': n_clusters,
    'Number of Noise Points': n_noise,
    'Number of Core Points': (clusters != -1).sum()
}

plt.barh(list(stats.keys()), list(stats.values()), color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Count', fontsize=12)
plt.title('DBSCAN Statistics', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Silhouette Score: {silhouette_score(X_scaled[clusters != -1], clusters[clusters != -1]):.3f}")
```

---

## Dimensionality Reduction

### Why Reduce Dimensions?

- Visualize high-dimensional data
- Reduce computational cost
- Remove noise and redundancy
- Improve model performance
- Handle curse of dimensionality

### Principal Component Analysis (PCA)

Linear dimensionality reduction that finds directions of maximum variance.

**How it works:**
1. Center the data
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto principal components

**Pros:**
- Fast and scalable
- Preserves global structure
- Interpretable (explained variance)
- Good for linear relationships

**Cons:**
- Assumes linear relationships
- May lose non-linear structure
- Sensitive to scaling

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale data (important for PCA!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained variance per component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Find optimal number of components
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Explained variance plot
axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[0].set_xlabel('Number of Components', fontsize=12)
axes[0].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[0].set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Visualize in 2D
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
axes[1].set_xlabel('First Principal Component', fontsize=12)
axes[1].set_ylabel('Second Principal Component', fontsize=12)
axes[1].set_title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Component loadings (feature contributions)
if hasattr(pca, 'components_'):
    print("\nFirst Principal Component Loadings:")
    print(pca.components_[0])
```

### t-SNE

t-Distributed Stochastic Neighbor Embedding. Non-linear dimensionality reduction for visualization.

**Parameters:**
- **perplexity**: Balance between local and global structure (typically 5-50)
- **learning_rate**: Step size (typically 10-1000)

**Pros:**
- Excellent for visualization
- Preserves local structure
- Good for non-linear relationships

**Cons:**
- Slow (O(n²))
- Not for feature reduction
- Results vary with perplexity
- Can't transform new data easily

```python
from sklearn.manifold import TSNE

# t-SNE (use subset for large datasets)
X_sample = X_scaled[:1000] if len(X_scaled) > 1000 else X_scaled

# Try different perplexities
perplexities = [5, 30, 50]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=50)
    axes[idx].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[idx].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[idx].set_title(f't-SNE (perplexity={perplexity})', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Note: t-SNE is for visualization, not feature engineering
```

### UMAP

Uniform Manifold Approximation and Projection. Modern alternative to t-SNE.

```python
try:
    import umap
    
    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = umap_reducer.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=50)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title('UMAP Visualization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("UMAP advantages:")
    print("- Faster than t-SNE")
    print("- Better preserves global structure")
    print("- Can transform new data")
    
except ImportError:
    print("Install UMAP: pip install umap-learn")
```

---

## Anomaly Detection

### What is Anomaly Detection?

Identifying data points that deviate significantly from the norm.

**Applications:**
- Fraud detection
- Network intrusion detection
- Manufacturing defect detection
- Medical diagnosis
- System monitoring

### Isolation Forest

Tree-based anomaly detection. Isolates anomalies instead of profiling normal points.

**How it works:**
- Randomly selects features and split values
- Anomalies are easier to isolate (fewer splits needed)
- Uses average path length as anomaly score

**Pros:**
- Fast and scalable
- Works with high-dimensional data
- No assumptions about data distribution
- Handles mixed data types

**Cons:**
- Requires contamination parameter
- May struggle with local anomalies
- Less interpretable

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
outliers = iso_forest.fit_predict(X_scaled)

# -1 = outlier, 1 = normal
n_outliers = (outliers == -1).sum()
print(f"Detected {n_outliers} outliers ({n_outliers/len(X_scaled)*100:.1f}%)")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
normal_mask = outliers == 1
outlier_mask = outliers == -1

plt.scatter(X_scaled[normal_mask, 0], X_scaled[normal_mask, 1], 
           c='blue', alpha=0.6, s=50, label='Normal')
plt.scatter(X_scaled[outlier_mask, 0], X_scaled[outlier_mask, 1], 
           c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='Outlier')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Isolation Forest Anomaly Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Anomaly scores
anomaly_scores = iso_forest.score_samples(X_scaled)

plt.subplot(1, 2, 2)
plt.hist(anomaly_scores[normal_mask], bins=50, alpha=0.7, label='Normal', color='blue')
plt.hist(anomaly_scores[outlier_mask], bins=50, alpha=0.7, label='Outlier', color='red')
plt.xlabel('Anomaly Score (lower = more anomalous)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Local Outlier Factor (LOF)

Density-based anomaly detection. Compares local density of a point with its neighbors.

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
outliers_lof = lof.fit_predict(X_scaled)

n_outliers_lof = (outliers_lof == -1).sum()
print(f"LOF detected {n_outliers_lof} outliers")

# Compare methods
print("\nMethod Comparison:")
print(f"Isolation Forest: {n_outliers} outliers")
print(f"LOF: {n_outliers_lof} outliers")
```

### One-Class SVM

Support Vector Machine for anomaly detection. Learns a boundary around normal data.

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
outliers_svm = oc_svm.fit_predict(X_scaled)

n_outliers_svm = (outliers_svm == -1).sum()
print(f"One-Class SVM detected {n_outliers_svm} outliers")
```

---

## Clustering Evaluation Metrics

### Internal Metrics (No Ground Truth)

```python
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score, adjusted_rand_score)

# Silhouette Score: Measures how similar points are to their cluster vs other clusters
# Range: -1 to 1 (higher is better)
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.3f}")

# Calinski-Harabasz Score: Ratio of between-cluster to within-cluster variance
# Higher is better
calinski = calinski_harabasz_score(X_scaled, clusters)
print(f"Calinski-Harabasz Score: {calinski:.2f}")

# Davies-Bouldin Score: Average similarity ratio of clusters
# Lower is better
davies = davies_bouldin_score(X_scaled, clusters)
print(f"Davies-Bouldin Score: {davies:.3f}")
```

### External Metrics (With Ground Truth)

```python
# Adjusted Rand Index: Measures similarity between clusters and true labels
# Range: -1 to 1 (1 = perfect match)
ari = adjusted_rand_score(true_labels, clusters)
print(f"Adjusted Rand Index: {ari:.3f}")
```

## Practice Exercises

### Exercise 1: Customer Segmentation

**Task:** Cluster customers using K-Means and analyze segments.

**Solution:**
```python
# Load customer data
# Features: age, income, spending_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using multiple methods
inertias = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose optimal k
optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Optimal k: {optimal_k}")

# Fit with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Analyze clusters
df['cluster'] = clusters
print("\nCluster Analysis:")
print(df.groupby('cluster').agg({
    'age': 'mean',
    'income': 'mean',
    'spending_score': 'mean',
    'cluster': 'count'
}).rename(columns={'cluster': 'count'}))

# Visualize
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
for i in range(optimal_k):
    cluster_data = X_scaled[clusters == i]
    ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
               label=f'Cluster {i}', alpha=0.6)
ax1.set_xlabel('Age')
ax1.set_ylabel('Income')
ax1.set_zlabel('Spending Score')
ax1.set_title('3D Cluster Visualization')
ax1.legend()

plt.tight_layout()
plt.show()
```

### Exercise 2: Anomaly Detection

**Task:** Detect anomalies in transaction data.

**Solution:**
```python
# Load transaction data
# Features: amount, time, location, etc.

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply multiple anomaly detection methods
methods = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'LOF': LocalOutlierFactor(contamination=0.05, n_neighbors=20),
    'One-Class SVM': OneClassSVM(nu=0.05, kernel='rbf')
}

results = {}

for name, method in methods.items():
    outliers = method.fit_predict(X_scaled)
    n_outliers = (outliers == -1).sum()
    results[name] = {
        'outliers': outliers,
        'count': n_outliers
    }
    print(f"{name}: {n_outliers} outliers detected")

# Compare results
print("\nOverlap Analysis:")
for i, (name1, result1) in enumerate(results.items()):
    for name2, result2 in list(results.items())[i+1:]:
        overlap = ((result1['outliers'] == -1) & (result2['outliers'] == -1)).sum()
        print(f"{name1} & {name2}: {overlap} common outliers")
```

---

## Key Takeaways

1. **Clustering**: Find groups in data using K-Means, Hierarchical, or DBSCAN
2. **Dimensionality Reduction**: Visualize high-dim data with PCA, t-SNE, or UMAP
3. **Anomaly Detection**: Find outliers using Isolation Forest, LOF, or One-Class SVM
4. **No labels needed**: Work with unlabeled data
5. **Evaluation is challenging**: Use internal metrics (silhouette, etc.) when no ground truth
6. **Scale features**: Most algorithms require scaled features
7. **Choose parameters carefully**: eps, k, perplexity significantly affect results

---

## Best Practices

### Clustering
- Always scale features before clustering
- Use multiple methods to validate results
- Visualize clusters to understand structure
- Consider domain knowledge when choosing k
- Use silhouette score to evaluate quality

### Dimensionality Reduction
- Scale features before PCA
- Use explained variance to choose components
- t-SNE is for visualization, not feature reduction
- UMAP is often better than t-SNE
- PCA preserves global structure, t-SNE preserves local

### Anomaly Detection
- Set contamination rate based on domain knowledge
- Use multiple methods and compare results
- Validate anomalies with domain experts
- Consider context (temporal, spatial)
- Balance false positives and false negatives

---

## Resources and Further Learning

### Books

1. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - Comprehensive coverage of clustering and dimensionality reduction
   - Chapter 9: Mixture Models and EM
   - Chapter 12: Continuous Latent Variables (PCA, etc.)

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani & Friedman
   - [Free Online](https://web.stanford.edu/~hastie/ElemStatLearn/)
   - Chapter 14: Unsupervised Learning

3. **"Hands-On Unsupervised Learning"** - Ankur A. Patel
   - Practical guide with code examples

### Important Papers

1. **"A Tutorial on Principal Component Analysis"** - Shlens, 2014
2. **"Visualizing Data using t-SNE"** - van der Maaten & Hinton, 2008
3. **"UMAP: Uniform Manifold Approximation and Projection"** - McInnes et al., 2018
4. **"Some methods for classification and analysis of multivariate observations"** - MacQueen, 1967 (K-Means)
5. **"A density-based algorithm for discovering clusters"** - Ester et al., 1996 (DBSCAN)
6. **"Isolation Forest"** - Liu et al., 2008

### Online Courses

1. **Unsupervised Learning** - Coursera (Andrew Ng)
   - Part of Machine Learning course
   - Covers K-Means, PCA

2. **CS229: Machine Learning** - Stanford
   - [Course Website](http://cs229.stanford.edu/)
   - Covers clustering, dimensionality reduction

### Datasets

1. **Clustering**:
   - [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
   - [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
   - [Customer Segmentation](https://www.kaggle.com/datasets?search=customer+segmentation)

2. **Dimensionality Reduction**:
   - [MNIST](http://yann.lecun.com/exdb/mnist/)
   - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

3. **Anomaly Detection**:
   - [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - [Network Intrusion](https://www.kaggle.com/datasets?search=intrusion)

### Tools and Libraries

1. **scikit-learn**: Comprehensive ML library
   - [Documentation](https://scikit-learn.org/)
   - K-Means, DBSCAN, PCA, t-SNE, Isolation Forest

2. **UMAP**: Uniform Manifold Approximation and Projection
   - [Documentation](https://umap-learn.readthedocs.io/)
   - Better alternative to t-SNE

3. **hdbscan**: Hierarchical DBSCAN
   - [Documentation](https://hdbscan.readthedocs.io/)
   - Improved clustering algorithm

4. **PyOD**: Python Outlier Detection
   - [Documentation](https://pyod.readthedocs.io/)
   - Comprehensive anomaly detection library

---

## Next Steps

- Practice with real datasets
- Experiment with different algorithms
- Learn association rules (market basket analysis)
- Move to [09-neural-networks-basics](../09-neural-networks-basics/README.md)

**Remember**: Unsupervised learning reveals hidden patterns! Use domain knowledge to validate results.

