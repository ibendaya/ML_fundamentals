### K-Means

**Description**:  
K-Means is an unsupervised machine learning algorithm used for clustering tasks. It groups data points into a specified number of clusters based on their similarity. The algorithm works by assigning data points to the nearest centroid and then updating the centroids based on the mean of the points in each cluster. This process iterates until the centroids stabilize or a set number of iterations is reached.

**How It Works**:

1. **Initialization**:  
   The algorithm starts by randomly selecting `k` data points as the initial centroids for the clusters.

2. **Assignment Step**:  
   Each data point is assigned to the nearest centroid, forming clusters. The distance between data points and centroids is usually measured using **Euclidean distance**.

3. **Update Step**:  
   After assigning all data points to clusters, the centroids are recalculated by taking the mean of all points in each cluster.

4. **Convergence**:  
   The algorithm repeats the assignment and update steps until the centroids no longer change, indicating convergence. Alternatively, the algorithm will stop after a maximum number of iterations.

5. **Cluster Labels**:  
   After convergence, each data point is assigned to a cluster, and the final centroids represent the center of each cluster.

**Key Components**:

- **Centroids**: The central points of the clusters. Initially selected randomly, the centroids are updated as the algorithm progresses.
- **Clusters**: Groups of data points that are closest to each centroid. Each point belongs to the cluster whose centroid is the closest.
- **Euclidean Distance**: The algorithm uses Euclidean distance to measure the proximity between data points and centroids.

**Training and Prediction**:
- **Training**: The algorithm trains by iteratively assigning points to clusters and updating the centroids until convergence.
- **Prediction**: After training, you can classify new data points by determining which centroid is closest to each point.

**Customization**:
- `k`: The number of clusters. It must be specified before running the algorithm.
- `max_iters`: The maximum number of iterations to run the algorithm. Default is 100.
- `plot_steps`: A boolean flag that, when set to `True`, visualizes the clusters and centroids after each iteration.

**Visualization**:  
The algorithm includes an optional visualization feature, which plots the clusters and centroids during the training process. This helps in understanding how the algorithm converges over time.

---

This implementation provides a simple yet effective way to apply K-Means clustering to any dataset, with the ability to visualize the clustering process. You can customize the number of clusters, maximum iterations, and whether or not to visualize the steps.

