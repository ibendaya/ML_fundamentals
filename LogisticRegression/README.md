### Logistic Regression

**Description**:  
Logistic Regression is a supervised machine learning algorithm used for binary classification tasks. It models the probability that a given input belongs to a particular class. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of a data point belonging to a specific class, outputting values between 0 and 1. The algorithm uses the **sigmoid function** to map the linear combination of features to a probability.

**How It Works**:

1. **Model Equation**:  
   Logistic regression uses the following equation to predict the probability of the positive class:

   \[
   P(y=1|X) = \sigma(w^T X + b)
   \]

   where:
   - \( X \) is the input feature vector,
   - \( w \) is the weight vector,
   - \( b \) is the bias term,
   - \( \sigma \) is the **sigmoid function**, which maps any real-valued number to a value between 0 and 1.

2. **Sigmoid Function**:  
   The sigmoid function is used to squish the output of the linear equation into a probability:

   \[
   \sigma(x) = \frac{1}{1 + e^{-x}}
   \]

   The output is interpreted as the probability of the data point belonging to the positive class (label 1). The prediction is typically made by classifying any value greater than 0.5 as class 1 and anything less than 0.5 as class 0.

3. **Training with Gradient Descent**:  
   The weights and bias are learned by minimizing the **binary cross-entropy loss** function using **gradient descent**. The loss function measures the error between the predicted probabilities and the actual binary labels. The parameters are updated by taking steps proportional to the negative gradient of the loss.

4. **Optimization**:  
   - The gradient of the loss with respect to the weights and bias is computed.
   - The weights and bias are updated using the following equations:

   \[
   w = w - \eta \cdot \frac{1}{n} \sum (h(x) - y) \cdot X
   \]
   \[
   b = b - \eta \cdot \frac{1}{n} \sum (h(x) - y)
   \]

   where:
   - \( \eta \) is the learning rate,
   - \( h(x) \) is the predicted probability,
   - \( y \) is the true label,
   - \( n \) is the number of samples.

5. **Prediction**:  
   Once the model is trained, the `predict` method computes the probability of the positive class using the learned weights and bias. The predicted class is 1 if the probability is greater than 0.5, and 0 otherwise.

**Key Components**:

- **Weights and Bias**: The model learns the optimal weights and bias during training through gradient descent.
- **Sigmoid Activation**: The sigmoid function is used to produce class probabilities between 0 and 1.
- **Training**: The model is trained by iteratively updating the weights and bias using gradient descent to minimize the loss function.
- **Prediction**: After training, the model predicts the class labels by classifying probabilities greater than 0.5 as 1 (positive class) and others as 0 (negative class).

**Advantages**:
- Simple and efficient for binary classification problems.
- Provides probabilities for predictions, offering more insight than just class labels.
- Interpretable, as the model coefficients indicate the strength and direction of relationships between features and the target.

**Disadvantages**:
- Assumes a linear relationship between the features and the log-odds of the target, which may not always hold.
- Sensitive to outliers, which can distort the model’s predictions.
- May struggle with complex, non-linear decision boundaries, requiring transformations or additional models (e.g., kernel methods) for better performance.

**Customization**:
- You can adjust the learning rate and the number of iterations for gradient descent to control the training process.

---

This implementation provides a simple yet powerful method for binary classification using logistic regression. The algorithm is widely used due to its simplicity and effectiveness for linearly separable problems.
