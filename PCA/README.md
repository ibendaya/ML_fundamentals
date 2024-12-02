### Principal Component Analysis (PCA)

**Description**:  
Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique commonly used for feature extraction and data compression. It transforms data into a new coordinate system where the greatest variance in the data comes to lie along the first few components (principal components). PCA is widely used for visualizing high-dimensional data, noise reduction, and preprocessing data for other machine learning algorithms.

**How It Works**:

1. **Mean Centering**:  
   PCA begins by centering the data around the origin by subtracting the mean of each feature from the corresponding feature values. This ensures that the data has zero mean, which is a necessary step for PCA.

2. **Covariance Matrix**:  
   The next step is to compute the **covariance matrix** of the centered data. The covariance matrix captures the pairwise covariances between the features. This matrix reflects how much the features vary together.

   \[
   \text{Cov}(X) = \frac{1}{n-1} \cdot X^T X
   \]

   where \( X \) is the mean-centered data.

3. **Eigenvalues and Eigenvectors**:  
   The core idea of PCA is to find the **eigenvalues** and **eigenvectors** of the covariance matrix. The eigenvectors (also called **principal components**) represent the directions of maximum variance in the data, and the eigenvalues represent the magnitude of the variance along those directions.

4. **Sorting by Eigenvalues**:  
   The eigenvectors are sorted in descending order of their corresponding eigenvalues. This ordering allows the first few eigenvectors to capture the most variance in the data.

5. **Dimensionality Reduction**:  
   The next step is to select the top `n_components` eigenvectors (principal components) corresponding to the largest eigenvalues. These components are then used to project the original data into a new, lower-dimensional space. This transformation reduces the dimensionality of the data while retaining as much variance as possible.

6. **Transformation**:  
   To transform the data, we multiply the mean-centered data by the matrix of the selected principal components. This gives the data in the new lower-dimensional space.

**Key Components**:

- **Mean Centering**: The data is centered by subtracting the mean of each feature.
- **Covariance Matrix**: A matrix that captures the covariance between features.
- **Eigenvectors and Eigenvalues**: Used to determine the directions of maximum variance in the data and how much variance there is along those directions.
- **Principal Components**: The eigenvectors corresponding to the largest eigenvalues, used for dimensionality reduction.
- **Transformation**: The data is projected into the new space defined by the selected principal components.

**Advantages**:
- Reduces the dimensionality of the data, helping to mitigate the curse of dimensionality.
- Often improves the performance of machine learning algorithms by removing noise and irrelevant features.
- Can help with data visualization by reducing high-dimensional data to 2 or 3 dimensions.

**Disadvantages**:
- PCA assumes that the principal components are linear combinations of the original features. This may not always hold for non-linear relationships.
- The interpretability of the transformed data can be difficult, as the new features (principal components) are combinations of the original features.
- Sensitive to scaling: features with larger variance may dominate the principal components unless the data is properly scaled (e.g., standardization).

**Customization**:
- `n_components`: The number of principal components to keep for the reduced dataset. It can be set when initializing the PCA object.

---

This implementation provides a straightforward approach to applying PCA for dimensionality reduction. It is effective for transforming high-dimensional data into a lower-dimensional space while retaining the most significant variance in the data.
