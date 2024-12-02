### K-Nearest Neighbors (KNN)

**Description**:  
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its `k` nearest neighbors in the feature space. For regression, it predicts the output as the average of the outputs of the nearest neighbors. The algorithm is simple, instance-based, and non-parametric.

**How It Works**:

1. **Training**:  
   In KNN, the training phase consists of simply storing the training data. No model is explicitly trained, making it a lazy learner. The entire training dataset is used during prediction.

2. **Prediction**:  
   To predict the class of a new data point, the algorithm:
   - Computes the **Euclidean distance** between the new point and all points in the training set.
   - Sorts the distances in ascending order and selects the `k` nearest neighbors.
   - For classification, the algorithm assigns the most frequent class among the `k` neighbors to the new point.

3. **Majority Voting**:  
   For classification, KNN uses a majority voting system where the most common class among the `k` nearest neighbors is chosen as the predicted class for the new data point.

4. **Distance Metric**:  
   The default distance metric used in this implementation is **Euclidean distance**, but other distance metrics (e.g., Manhattan, Minkowski) can be used depending on the problem and dataset.

5. **Hyperparameter (`k`)**:  
   The number of nearest neighbors, `k`, is a key hyperparameter that affects the performance of the algorithm:
   - Small values of `k` make the model sensitive to noise in the data.
   - Large values of `k` provide smoother decision boundaries but may overlook local patterns.

**Key Components**:

- **Training Data**: The `fit` method stores the training data (`X_train` and `y_train`), which will be used later for predictions.
- **Prediction**: The `predict` method makes predictions for new data points by calculating the distances to all points in the training set and performing majority voting among the `k` nearest neighbors.
- **Distance Metric**: The `euc_distance` function computes the Euclidean distance between two data points in the feature space.

**Advantages**:
- Simple and easy to understand.
- Effective for small datasets with low-dimensional feature spaces.
- No training phase (lazy learning).

**Disadvantages**:
- Computationally expensive at prediction time since it needs to compute distances to all training points.
- Performance can degrade with high-dimensional data (curse of dimensionality).
- Sensitive to the choice of `k` and the scaling of the data.

**Customization**:
- `k`: The number of nearest neighbors. It can be customized when initializing the `KNN` object.
  