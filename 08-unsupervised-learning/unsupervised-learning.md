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

**When to use:**
- No labels available
- Exploratory data analysis
- Finding hidden structures
- Data compression

---

## Clustering

### K-Means

Partition data into k clusters.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           marker='x', s=200, c='red', linewidths=3)
plt.title('K-Means Clustering')
plt.show()

# Evaluate
silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette:.3f}")
```

### Choosing K (Elbow Method)

```python
# Elbow method
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=3)
labels = cluster.fit_predict(X)

# Dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.show()
```

### DBSCAN

Density-based clustering.

```python
from sklearn.cluster import DBSCAN

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# Note: -1 indicates outliers
n_outliers = (clusters == -1).sum()
print(f"Number of outliers: {n_outliers}")
```

---

## Dimensionality Reduction

### PCA

```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum():.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Visualization')
plt.show()
```

### t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE Visualization')
plt.show()
```

---

## Anomaly Detection

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X)

# -1 = outlier, 1 = normal
n_outliers = (outliers == -1).sum()
print(f"Detected {n_outliers} outliers")
```

---

## Practice Exercises

### Exercise 1: Customer Segmentation

**Task:** Cluster customers using K-Means.

**Solution:**
```python
# Load customer data
# Features: age, income, spending_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Choose k (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Analyze clusters
df['cluster'] = clusters
print(df.groupby('cluster').mean())
```

---

## Key Takeaways

1. **Clustering**: Find groups in data
2. **Dimensionality Reduction**: Visualize high-dim data
3. **Anomaly Detection**: Find outliers
4. **No labels needed**: Work with unlabeled data

---

## Next Steps

- Practice with real datasets
- Experiment with different algorithms
- Move to [09-neural-networks-basics](../09-neural-networks-basics/README.md)

**Remember**: Unsupervised learning reveals hidden patterns!

