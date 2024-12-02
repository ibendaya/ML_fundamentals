### Support Vector Machine (SVM)

**Description**:  
Support Vector Machine (SVM) is a supervised learning algorithm primarily used for classification tasks, though it can also be adapted for regression. The goal of SVM is to find the optimal hyperplane that maximally separates the classes in the feature space. SVM aims to maximize the margin between the closest points (support vectors) of each class and the hyperplane. SVM is known for its effectiveness in high-dimensional spaces and its ability to handle non-linear decision boundaries using kernel functions.

**How It Works**:

1. **Hyperplane and Margin**:  
   SVM finds a hyperplane (a decision boundary) that separates data points of different classes. In a two-dimensional space, this hyperplane is simply a line. The SVM algorithm maximizes the **margin**, which is the distance between the hyperplane and the closest data points from each class, known as **support vectors**.

2. **Optimization Problem**:  
   The objective of SVM is to maximize the margin while minimizing classification errors. This is formulated as an optimization problem, where we want to:
   
   \[
   \text{minimize } \frac{1}{2} \| w \|^2
   \]
   subject to the constraint that for each data point \( (x_i, y_i) \), the following holds:
   
   \[
   y_i (w \cdot x_i - b) \geq 1
   \]
   
   where:
   - \( w \) is the weight vector (normal to the hyperplane),
   - \( b \) is the bias (offset from the origin),
   - \( y_i \) is the class label (either +1 or -1),
   - \( x_i \) is the feature vector of the data point.

3. **Regularization**:  
   To prevent overfitting, a regularization term is introduced, controlled by the **lambda parameter**. The regularization term penalizes large weights, preventing the hyperplane from fitting noise in the data.

   The loss function becomes:

   \[
   J(w, b) = \frac{1}{2} \| w \|^2 + \lambda \sum_{i=1}^n \max(0, 1 - y_i (w \cdot x_i - b))
   \]

   where \( \lambda \) is the regularization parameter, controlling the trade-off between maximizing the margin and minimizing classification errors.

4. **Training**:  
   SVM uses an iterative process to adjust the weights and bias. In each iteration:
   - If a data point satisfies the margin condition \( y_i (w \cdot x_i - b) \geq 1 \), only the weights are updated using the regularization term.
   - If the margin condition is violated, both the weights and the bias are updated to minimize the classification error.

5. **Prediction**:  
   After training, the model predicts the class label for a new data point by calculating the sign of the decision function:

   \[
   y_{\text{pred}} = \text{sign}(w \cdot x - b)
   \]

   where \( w \) and \( b \) are the learned parameters.

**Key Components**:

- **Weights and Bias**: The model learns the optimal weight vector \( w \) and bias \( b \) during training that define the decision hyperplane.
- **Regularization**: The regularization term (controlled by the lambda parameter) helps prevent overfitting by penalizing large weights.
- **Loss Function**: The objective function combines a margin maximization term and a penalty for misclassified points.
- **Prediction**: After training, the model uses the learned weights and bias to classify new data points.

**Advantages**:
- Effective in high-dimensional spaces and for problems where the number of dimensions exceeds the number of samples.
- Can handle non-linear decision boundaries using kernel tricks.
- Robust to overfitting, especially when regularization is applied.

**Disadvantages**:
- Computationally expensive, especially for large datasets.
- Requires careful tuning of hyperparameters (e.g., learning rate, regularization parameter).
- Sensitive to the choice of kernel function for non-linear data.

**Customization**:
- `lr`: The learning rate controls the step size during the weight update process.
- `lambda_param`: The regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.
- `n_iters`: The number of iterations to train the model. Increasing the number of iterations can improve convergence.

---

This implementation provides a basic linear SVM model that can be adapted for classification tasks. It leverages regularization to improve generalization and uses an iterative optimization process to learn the optimal hyperplane.
