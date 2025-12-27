# Phase 8: Unsupervised Learning

Learn to find patterns in data without labels.

##  What You'll Learn

- Clustering Algorithms (K-Means, Hierarchical, DBSCAN)
- Dimensionality Reduction (PCA, t-SNE)
- Anomaly Detection
- Association Rules
- Real-world Applications

##  Topics Covered

### 1. Clustering
- **K-Means**: Partition data into k clusters
  - Choosing k (Elbow method, Silhouette score)
  - Pros and cons
- **Hierarchical Clustering**: Tree-like cluster structure
  - Agglomerative vs Divisive
  - Dendrograms
- **DBSCAN**: Density-based clustering
  - Handles non-spherical clusters
  - Identifies outliers

### 2. Dimensionality Reduction
- **PCA**: Linear dimensionality reduction
  - Explained variance
  - When to use
- **t-SNE**: Non-linear visualization
  - Great for visualization
  - Not for feature reduction
- **UMAP**: Modern alternative
  - Faster than t-SNE
  - Better global structure

### 3. Anomaly Detection
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector approach
- **Local Outlier Factor (LOF)**: Density-based
- **Applications**: Fraud detection, system monitoring

### 4. Association Rules
- **Market Basket Analysis**: Find item associations
- **Apriori Algorithm**: Find frequent itemsets
- **Support, Confidence, Lift**: Key metrics

##  Learning Objectives

By the end of this module, you should be able to:
- Apply clustering algorithms to unlabeled data
- Reduce dimensionality for visualization
- Detect anomalies in data
- Find associations in transactional data

##  Projects

1. **Customer Segmentation**: Cluster customers by behavior
2. **Anomaly Detection**: Detect fraudulent transactions
3. **Market Basket Analysis**: Find product associations
4. **Data Visualization**: Use t-SNE to visualize high-dim data

##  Key Concepts

- **No Labels**: Unsupervised learning works without targets
- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Reduce features while keeping information
- **Anomaly Detection**: Find unusual patterns
- **Evaluation**: Harder without labels (use silhouette score, etc.)

##  Additional Resources

- [Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)
- [Dimensionality Reduction - Scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html)

---

**Previous Phase:** [07-feature-engineering](../07-feature-engineering/README.md)  
**Next Phase:** [09-neural-networks-basics](../09-neural-networks-basics/README.md)

