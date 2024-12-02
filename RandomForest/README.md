### Random Forest

**Description**:  
Random Forest is an ensemble learning method that combines multiple decision trees to improve classification accuracy and generalization. Each decision tree is trained on a random subset of the data, and the final prediction is made by aggregating the predictions of all trees. The Random Forest algorithm is widely used due to its ability to handle large datasets, provide high accuracy, and reduce overfitting compared to individual decision trees.

**How It Works**:

1. **Bootstrap Aggregating (Bagging)**:  
   Random Forest uses **bootstrap sampling**, where each decision tree is trained on a random subset of the data with replacement. This means that each tree in the forest may see a different subset of data, reducing the variance and preventing overfitting.

2. **Training Multiple Trees**:  
   - A set of decision trees is trained independently on different bootstrap samples of the training data.
   - Each tree is trained with a random subset of features (controlled by the `n_features` parameter), which introduces additional randomness and helps in reducing the correlation between trees.

3. **Prediction**:  
   - For classification tasks, each tree in the forest predicts a class label, and the final prediction is determined by majority voting (the most common label predicted by the trees).
   - For regression tasks, the final prediction is typically the average of the predictions made by all trees.

4. **Hyperparameters**:
   - `n_trees`: The number of decision trees in the forest. More trees generally lead to better performance but require more computational resources.
   - `max_depth`: The maximum depth of each tree. Limiting the depth of trees helps prevent overfitting.
   - `min_samples_split`: The minimum number of samples required to split a node. This parameter helps control the growth of each tree.
   - `n_features`: The number of features to consider when splitting a node. By default, all features are considered, but using a subset of features can help reduce overfitting.

5. **Advantages**:
   - Random Forest typically outperforms individual decision trees by reducing overfitting and improving generalization.
   - The model is robust to noise and outliers in the data.
   - It can handle both classification and regression tasks and is capable of modeling complex relationships.
   - Random Forest provides feature importance scores, allowing for insight into which features contribute most to the model’s predictions.

6. **Disadvantages**:
   - Random Forest models can be computationally expensive and memory-intensive, especially with a large number of trees and features.
   - The model can be more challenging to interpret compared to individual decision trees, as it lacks a clear decision-making path.

**Customization**:
- `n_trees`: The number of trees in the forest. You can increase this number to improve model performance at the cost of higher computational resources.
- `max_depth`: Controls the depth of each tree, helping to avoid overfitting.
- `min_samples_split`: Controls the minimum number of samples required to split a node.
- `n_features`: The number of features to consider when splitting a node. By default, this is set to consider all features.

---

This implementation of Random Forest leverages multiple decision trees to provide a powerful and versatile machine learning model. It is especially effective for handling large, high-dimensional datasets while preventing overfitting and improving generalization.
