# Clustering & Dimensionality Reduction Summary

## Clustering Overview

**Clustering** is an unsupervised machine learning technique that groups data points based on similarity.

### Common Applications

- Customer segmentation
- Anomaly / outlier detection
- Pattern recognition
- Feature engineering
- Data compression

---

## K-Means Clustering

**K-means** partitions data into _k_ clusters by minimizing the distance between data points and their assigned cluster **centroids**.

### Strengths

- Fast and scalable
- Easy to interpret
- Works well with spherical, similarly sized clusters

### Limitations

- Requires specifying _k_ in advance
- Struggles with **imbalanced** or **non-convex** clusters
- Sensitive to initial centroid placement and outliers

### Heuristics for Choosing _k_

| Method                   | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| **Elbow Method**         | Plot inertia vs. _k_; look for the “elbow” point                        |
| **Silhouette Analysis**  | Measures how well points fit their own cluster vs. neighboring clusters |
| **Davies-Bouldin Index** | Lower values indicate better cluster separation                         |

---

## Density-Based Clustering

### DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) forms clusters based on local density.

| Parameter     | Meaning                                        |
| ------------- | ---------------------------------------------- |
| `eps`         | Neighborhood search radius                     |
| `min_samples` | Minimum points required to form a dense region |

**Advantages**

- Discovers clusters of arbitrary shape
- Automatically identifies noise / outliers (label = –1)
- Does not require specifying the number of clusters

**Limitations**

- Sensitive to the choice of `eps` and `min_samples`
- Struggles with clusters of varying density

### HDBSCAN

**HDBSCAN** is a hierarchical extension of DBSCAN.

- Does **not** require a fixed `eps` parameter
- Uses **cluster stability** to extract the most persistent clusters
- Better handles clusters of varying density
- Still identifies noise points

---

## Hierarchical Clustering

Hierarchical methods build a tree of clusters (a **dendrogram**).

| Approach          | Direction | Description                                        |
| ----------------- | --------- | -------------------------------------------------- |
| **Agglomerative** | Bottom-up | Start with each point as its own cluster and merge |
| **Divisive**      | Top-down  | Start with one cluster and recursively split       |

The dendrogram provides a visual way to choose the number of clusters by cutting the tree at a desired height.

---

## Dimensionality Reduction

**Dimensionality reduction** simplifies data structure, reduces noise, and often improves downstream clustering or classification performance.

### Principal Component Analysis (PCA)

- **Linear** method that finds orthogonal directions (principal components) of maximum variance
- Minimizes information loss while reducing dimensions
- Useful for noise reduction and feature compression
- Classic application: **eigenfaces** in face recognition

| Concept                           | Description                                             |
| --------------------------------- | ------------------------------------------------------- |
| **Explained Variance Ratio**      | Proportion of total variance captured by each component |
| **Cumulative Explained Variance** | Running total of variance retained                      |

### Non-linear Methods

| Algorithm | Key Idea                               | Best For                                    |
| --------- | -------------------------------------- | ------------------------------------------- |
| **t-SNE** | Preserves local neighborhood structure | Visualization of high-dimensional data      |
| **UMAP**  | Balances local and global structure    | Visualization + general dimension reduction |

**Notes**

- Both t-SNE and UMAP are primarily used for **visualization**
- Results can vary with hyperparameters (perplexity for t-SNE, `min_dist` / `spread` for UMAP)
- PCA is often a strong, fast baseline and sometimes outperforms more complex methods on simple datasets

---

## Clustering + Dimensionality Reduction

Combining the two techniques is a powerful workflow:

1. Reduce noise and redundant features with PCA (or another method)
2. Apply clustering on the reduced feature space
3. Result: cleaner clusters and lower computational cost

---

## Quick Comparison of Clustering Algorithms

| Algorithm        | Shape Assumption | Needs _k_? | Handles Noise? | Density Variation | Scalability |
| ---------------- | ---------------- | ---------- | -------------- | ----------------- | ----------- |
| **K-Means**      | Spherical        | Yes        | Poor           | Poor              | Excellent   |
| **DBSCAN**       | Arbitrary        | No         | Yes            | Limited           | Good        |
| **HDBSCAN**      | Arbitrary        | No         | Yes            | Excellent         | Good        |
| **Hierarchical** | Arbitrary        | Optional   | Possible       | Moderate          | Fair        |

---

## Practical Tips

- Always **scale** features before distance-based methods (K-Means, DBSCAN, HDBSCAN).
- Use the **elbow method** + **silhouette score** together to choose _k_ for K-Means.
- Prefer **DBSCAN / HDBSCAN** when clusters have irregular shapes or when you need automatic outlier detection.
- Start with **PCA** for linear structure and noise reduction; move to t-SNE or UMAP when you need non-linear visualization.
- Inspect results visually whenever possible — metrics alone do not tell the full story.
