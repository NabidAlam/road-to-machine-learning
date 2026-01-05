# Complete Unsupervised Learning Project Tutorial

Step-by-step walkthrough of a comprehensive unsupervised learning project.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Exploration](#step-1-data-loading-and-exploration)
- [Step 2: Data Preprocessing](#step-2-data-preprocessing)
- [Step 3: Dimensionality Reduction](#step-3-dimensionality-reduction)
- [Step 4: Clustering Analysis](#step-4-clustering-analysis)
- [Step 5: Cluster Evaluation](#step-5-cluster-evaluation)
- [Step 6: Anomaly Detection](#step-6-anomaly-detection)
- [Step 7: Results Interpretation](#step-7-results-interpretation)

---

## Project Overview

**Project**: Customer Segmentation and Anomaly Detection

**Dataset**: Customer data with features like age, income, spending score, etc.

**Goals**:
1. Segment customers into meaningful groups
2. Detect anomalous customers
3. Visualize high-dimensional customer data
4. Provide actionable insights

**Type**: Unsupervised Learning (Clustering + Anomaly Detection)

**Difficulty**: Intermediate

**Time**: 2-3 hours

---

## Step 1: Data Loading and Exploration

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data (example with synthetic data)
# In practice, load from CSV or database
np.random.seed(42)
n_samples = 1000

# Create synthetic customer data
df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 70, n_samples),
    'annual_income': np.random.normal(50000, 15000, n_samples),
    'spending_score': np.random.randint(1, 100, n_samples),
    'purchase_frequency': np.random.poisson(5, n_samples),
    'avg_transaction_value': np.random.exponential(50, n_samples),
    'total_purchases': np.random.poisson(20, n_samples)
})

# Ensure non-negative values
df['annual_income'] = np.abs(df['annual_income'])
df['avg_transaction_value'] = np.abs(df['avg_transaction_value'])

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
```

### Data Exploration

```python
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Visualize distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age distribution
axes[0, 0].hist(df['age'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Income distribution
axes[0, 1].hist(df['annual_income'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Annual Income Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Annual Income')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Spending score distribution
axes[0, 2].hist(df['spending_score'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].set_title('Spending Score Distribution', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Spending Score')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Purchase frequency
axes[1, 0].hist(df['purchase_frequency'], bins=20, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].set_title('Purchase Frequency Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Purchase Frequency')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Average transaction value
axes[1, 1].hist(df['avg_transaction_value'], bins=30, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_title('Average Transaction Value Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Average Transaction Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=axes[1, 2], fmt='.2f')
axes[1, 2].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Step 2: Data Preprocessing

### Select Features

```python
# Select features for clustering
feature_cols = ['age', 'annual_income', 'spending_score', 
                'purchase_frequency', 'avg_transaction_value', 'total_purchases']
X = df[feature_cols].values

print(f"Features selected: {feature_cols}")
print(f"Data shape: {X.shape}")
```

### Scale Features

```python
# Scale features (critical for clustering!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled successfully!")
print(f"Scaled data shape: {X_scaled.shape}")
print(f"Mean: {X_scaled.mean(axis=0).round(3)}")
print(f"Std: {X_scaled.std(axis=0).round(3)}")
```

---

## Step 3: Dimensionality Reduction

### Apply PCA

```python
# PCA for dimensionality reduction and visualization
pca = PCA()
pca.fit(X_scaled)

# Cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontsize=12)
plt.title('PCA Explained Variance', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Apply PCA with 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('PCA Visualization (2D)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Explained variance (2D): {pca_2d.explained_variance_ratio_.sum():.3f}")
```

### Apply t-SNE (Optional)

```python
# t-SNE for visualization (use subset for large datasets)
if len(X_scaled) > 1000:
    sample_indices = np.random.choice(len(X_scaled), 1000, replace=False)
    X_sample = X_scaled[sample_indices]
else:
    X_sample = X_scaled
    sample_indices = np.arange(len(X_scaled))

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=50)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.title('t-SNE Visualization', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Step 4: Clustering Analysis

### Find Optimal k for K-Means

```python
# Find optimal k using multiple methods
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Choose optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k (Silhouette): {optimal_k}")
print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
```

### Apply K-Means Clustering

```python
# Apply K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['cluster_kmeans'] = clusters_kmeans

# Visualize clusters in 2D (PCA space)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, 
                     cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('K-Means Clustering (PCA Space)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Cluster sizes
plt.subplot(1, 2, 2)
unique, counts = np.unique(clusters_kmeans, return_counts=True)
plt.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.title('Cluster Sizes', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("K-Means Clustering Results:")
print(f"Number of clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters_kmeans):.3f}")
```

### Try Other Clustering Methods

```python
# DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal eps
neighbors = NearestNeighbors(n_neighbors=4)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, 3], axis=0)
optimal_eps = np.percentile(distances, 90)

dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = (clusters_dbscan == -1).sum()

print(f"\nDBSCAN Results:")
print(f"Number of clusters: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise}")

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
clusters_hierarchical = hierarchical.fit_predict(X_scaled)

print(f"\nHierarchical Clustering Results:")
print(f"Number of clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters_hierarchical):.3f}")

# Compare methods
comparison = pd.DataFrame({
    'Method': ['K-Means', 'DBSCAN', 'Hierarchical'],
    'Silhouette': [
        silhouette_score(X_scaled, clusters_kmeans),
        silhouette_score(X_scaled[clusters_dbscan != -1], clusters_dbscan[clusters_dbscan != -1]) if n_noise < len(X_scaled) else 0,
        silhouette_score(X_scaled, clusters_hierarchical)
    ],
    'N_Clusters': [optimal_k, n_clusters_dbscan, optimal_k]
})

print("\nClustering Method Comparison:")
print(comparison)
```

---

## Step 5: Cluster Evaluation

### Analyze Cluster Characteristics

```python
# Analyze K-Means clusters
print("Cluster Analysis (K-Means):")
cluster_analysis = df.groupby('cluster_kmeans')[feature_cols].agg(['mean', 'std'])
print(cluster_analysis)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, feature in enumerate(feature_cols):
    row = idx // 3
    col = idx % 3
    
    cluster_means = df.groupby('cluster_kmeans')[feature].mean()
    cluster_stds = df.groupby('cluster_kmeans')[feature].std()
    
    axes[row, col].bar(cluster_means.index, cluster_means.values, 
                       yerr=cluster_stds.values, alpha=0.7, 
                       color='steelblue', edgecolor='black', capsize=5)
    axes[row, col].set_xlabel('Cluster', fontsize=11)
    axes[row, col].set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
    axes[row, col].set_title(f'{feature.replace("_", " ").title()} by Cluster', 
                            fontsize=12, fontweight='bold')
    axes[row, col].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### Comprehensive Evaluation

```python
# Evaluate all clustering methods
def evaluate_clustering(X, labels, name):
    """Comprehensive clustering evaluation"""
    if len(set(labels)) < 2:
        return None
    
    # Remove noise points for DBSCAN
    if -1 in labels:
        mask = labels != -1
        X_eval = X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels
    
    metrics = {
        'Method': name,
        'N_Clusters': len(set(labels_eval)),
        'Silhouette': silhouette_score(X_eval, labels_eval),
        'Calinski-Harabasz': calinski_harabasz_score(X_eval, labels_eval),
        'Davies-Bouldin': davies_bouldin_score(X_eval, labels_eval)
    }
    
    return metrics

# Evaluate all methods
results = []
results.append(evaluate_clustering(X_scaled, clusters_kmeans, 'K-Means'))
results.append(evaluate_clustering(X_scaled, clusters_dbscan, 'DBSCAN'))
results.append(evaluate_clustering(X_scaled, clusters_hierarchical, 'Hierarchical'))

results_df = pd.DataFrame([r for r in results if r is not None])
print("\nComprehensive Clustering Evaluation:")
print(results_df.round(3))
```

---

## Step 6: Anomaly Detection

### Isolation Forest

```python
# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
outliers_iso = iso_forest.fit_predict(X_scaled)

n_outliers_iso = (outliers_iso == -1).sum()
print(f"Isolation Forest detected {n_outliers_iso} outliers ({n_outliers_iso/len(X_scaled)*100:.1f}%)")

# Anomaly scores
anomaly_scores_iso = iso_forest.score_samples(X_scaled)
```

### Local Outlier Factor

```python
# LOF
lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
outliers_lof = lof.fit_predict(X_scaled)

n_outliers_lof = (outliers_lof == -1).sum()
print(f"LOF detected {n_outliers_lof} outliers ({n_outliers_lof/len(X_scaled)*100:.1f}%)")

# Anomaly scores
anomaly_scores_lof = -lof.negative_outlier_factor_
```

### Compare Anomaly Detection Methods

```python
# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Isolation Forest
normal_mask_iso = outliers_iso == 1
outlier_mask_iso = outliers_iso == -1

axes[0].scatter(X_pca[normal_mask_iso, 0], X_pca[normal_mask_iso, 1], 
               c='blue', alpha=0.6, s=50, label='Normal')
axes[0].scatter(X_pca[outlier_mask_iso, 0], X_pca[outlier_mask_iso, 1], 
               c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='Outlier')
axes[0].set_xlabel('First Principal Component', fontsize=12)
axes[0].set_ylabel('Second Principal Component', fontsize=12)
axes[0].set_title('Isolation Forest Anomaly Detection', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LOF
normal_mask_lof = outliers_lof == 1
outlier_mask_lof = outliers_lof == -1

axes[1].scatter(X_pca[normal_mask_lof, 0], X_pca[normal_mask_lof, 1], 
               c='blue', alpha=0.6, s=50, label='Normal')
axes[1].scatter(X_pca[outlier_mask_lof, 0], X_pca[outlier_mask_lof, 1], 
               c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='Outlier')
axes[1].set_xlabel('First Principal Component', fontsize=12)
axes[1].set_ylabel('Second Principal Component', fontsize=12)
axes[1].set_title('LOF Anomaly Detection', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare overlap
overlap = ((outliers_iso == -1) & (outliers_lof == -1)).sum()
print(f"\nOverlap between methods: {overlap} outliers detected by both")
```

---

## Step 7: Results Interpretation

### Customer Segments

```python
# Create customer segment profiles
print("=" * 60)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 60)

for cluster_id in sorted(df['cluster_kmeans'].unique()):
    cluster_data = df[df['cluster_kmeans'] == cluster_id]
    
    print(f"\nCluster {cluster_id} ({len(cluster_data)} customers):")
    print(f"  Average Age: {cluster_data['age'].mean():.1f} years")
    print(f"  Average Annual Income: ${cluster_data['annual_income'].mean():.0f}")
    print(f"  Average Spending Score: {cluster_data['spending_score'].mean():.1f}")
    print(f"  Average Purchase Frequency: {cluster_data['purchase_frequency'].mean():.1f}")
    print(f"  Average Transaction Value: ${cluster_data['avg_transaction_value'].mean():.2f}")
    print(f"  Total Purchases: {cluster_data['total_purchases'].mean():.1f}")
```

### Actionable Insights

```python
# Generate insights
print("\n" + "=" * 60)
print("ACTIONABLE INSIGHTS")
print("=" * 60)

# Find high-value customers
high_value_cluster = df.groupby('cluster_kmeans')['annual_income'].mean().idxmax()
print(f"\n1. High-Value Customers (Cluster {high_value_cluster}):")
print(f"   - Target with premium products")
print(f"   - Offer loyalty programs")
print(f"   - Provide personalized service")

# Find high-spending customers
high_spending_cluster = df.groupby('cluster_kmeans')['spending_score'].mean().idxmax()
print(f"\n2. High-Spending Customers (Cluster {high_spending_cluster}):")
print(f"   - Focus on upselling")
print(f"   - Recommend complementary products")
print(f"   - Send frequent promotions")

# Anomaly insights
print(f"\n3. Anomalous Customers ({n_outliers_iso} detected):")
print(f"   - Review for fraud prevention")
print(f"   - Investigate unusual patterns")
print(f"   - Consider special handling")
```

### Final Summary

```python
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"\n1. Data: {len(df)} customers, {len(feature_cols)} features")
print(f"2. Optimal Clusters: {optimal_k} segments identified")
print(f"3. Clustering Quality: Silhouette Score = {silhouette_score(X_scaled, clusters_kmeans):.3f}")
print(f"4. Anomalies Detected: {n_outliers_iso} customers ({n_outliers_iso/len(df)*100:.1f}%)")
print(f"5. Dimensionality Reduction: {pca_2d.explained_variance_ratio_.sum():.3f} variance explained in 2D")

print("\nKey Deliverables:")
print("  - Customer segmentation with {optimal_k} distinct groups")
print("  - Anomaly detection for fraud prevention")
print("  - Visualizations for stakeholder presentation")
print("  - Actionable insights for marketing strategy")
```

---

## Key Takeaways

1. **Data Preprocessing**: Always scale features before clustering
2. **Optimal k**: Use multiple methods (elbow, silhouette, gap statistic)
3. **Multiple Methods**: Compare different clustering algorithms
4. **Evaluation**: Use multiple metrics to assess cluster quality
5. **Visualization**: Use PCA/t-SNE to visualize high-dimensional clusters
6. **Anomaly Detection**: Use multiple methods and compare results
7. **Interpretation**: Translate clusters into actionable business insights

---

**Congratulations!** You've completed a comprehensive unsupervised learning project!

