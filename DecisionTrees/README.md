### Decision Trees

**Description**:  
A decision tree is a supervised learning algorithm that splits data into subsets based on feature values. It is used for classification and regression tasks. This implementation constructs a binary decision tree using recursive splitting and the information gain criterion based on entropy. The tree is built by selecting the feature and threshold that maximizes the reduction in entropy at each step.

**How It Works**:

1. **Node Class**:  
   Each node in the tree is represented by the `Node` class. A node can either be an internal node or a leaf node:
   - Internal nodes store a feature index and a threshold value.
   - Leaf nodes store the class label or target value (in regression).

2. **Tree Construction**:  
   The `DecisionTree` class builds the tree using the `_grow_tree` method. The tree is built recursively by selecting the best feature and threshold that maximizes information gain (based on entropy).

3. **Stopping Criteria**:  
   The recursion stops when any of the following conditions are met:
   - The maximum depth of the tree is reached (`max_depth`).
   - All samples in the current node belong to the same class (pure node).
   - There are fewer samples than the minimum required for a split (`min_samples_split`).

4. **Information Gain**:  
   The algorithm uses **information gain** to determine the best feature and threshold for splitting. Information gain is calculated by comparing the entropy of the parent node with the weighted average entropy of the child nodes.

5. **Entropy**:  
   Entropy is a measure of impurity or disorder. For classification tasks, the entropy of a node is computed using the formula:

   \[
   \text{Entropy}(S) = - \sum_{i=1}^{n} p_i \log_2(p_i)
   \]

   where \( p_i \) is the proportion of each class in the node.

6. **Predictions**:  
   To make predictions, the algorithm traverses the tree from the root to a leaf node, following the feature values in the input and the decision rules at each node.

**Key Components**:
- **Feature Selection**: At each node, a random subset of features (`n_features`) is considered for splitting.
- **Splitting Criterion**: The algorithm uses **information gain** to choose the best feature and threshold to split the data.
- **Stopping Criteria**: The tree is pruned when reaching a certain depth or when further splits would not improve the model.

**Example**:

```python
from decision_tree import DecisionTree  # Assuming the class is in decision_tree.py
import numpy as np

# Sample dataset (features and labels)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and train the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Make predictions on new data
predictions = tree.predict(np.array([[3, 3], [1, 1]]))
print(predictions)  # Output: [1, 0]
```
**Advantages**:
- Simple to understand and implement.
- The tree structure makes it easy to visualize.
- Works well for both classification and regression tasks.
- No need for feature scaling or normalization.

**Disadvantages**:
- Prone to overfitting, especially with deep trees.
- Can be unstable with small changes in the data.
- Performance may degrade on high-dimensional datasets without feature selection or pruning.

**Customization**:
- You can customize the tree by setting the `min_samples_split`, `max_depth`, and `n_features` parameters when creating an instance of the `DecisionTree` class.

```python
tree = DecisionTree(min_samples_split=4, max_depth=5, n_features=2)
```
This implementation provides the flexibility to control the depth of the tree, the minimum number of samples for splitting a node, and the number of features to consider at each split, helping you manage the complexity of the model.