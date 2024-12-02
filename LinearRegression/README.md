### Linear Regression

**Description**:  
Linear Regression is a fundamental supervised learning algorithm used for predicting continuous values. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data. The algorithm learns the coefficients (weights) that minimize the difference between predicted and actual values, typically using a method called **Gradient Descent**.

**How It Works**:

1. **Model Equation**:  
   The linear regression model assumes the relationship between the input features \( X \) and the target \( y \) is linear. The prediction is made using the equation:

   \[
   y = Xw + b
   \]

   where:
   - \( X \) is the matrix of input features,
   - \( w \) is the vector of weights (coefficients),
   - \( b \) is the bias (intercept).

2. **Gradient Descent**:  
   The model parameters (weights and bias) are learned by minimizing the **Mean Squared Error (MSE)** between the predicted values \( y_{\text{pred}} \) and the actual target values \( y \). This is achieved by iteratively updating the weights using gradient descent, a first-order optimization algorithm.

   - The update rule for the weights \( w \) and bias \( b \) is:

   \[
   w = w - \eta \cdot \frac{1}{n} \sum (y_{\text{pred}} - y) X
   \]
   \[
   b = b - \eta \cdot \frac{1}{n} \sum (y_{\text{pred}} - y)
   \]

   where \( \eta \) is the learning rate and \( n \) is the number of samples.

3. **Training Process**:  
   - **Initialization**: The weights are initialized to zeros, and the bias is set to zero.
   - **Iterations**: The model iteratively updates the weights and bias for a predefined number of iterations (`n_iters`), using the gradient descent optimization algorithm.

4. **Prediction**:  
   Once the model is trained, the `predict` method computes the predicted values using the learned weights and bias.

5. **Hyperparameters**:
   - **Learning Rate (`lr`)**: Controls the step size during the gradient descent updates. A smaller learning rate leads to slower convergence, while a larger rate may cause overshooting.
   - **Number of Iterations (`n_iters`)**: Determines the number of iterations for updating the weights. More iterations may lead to better convergence.

**Key Components**:

- **Weights and Bias**: The model learns the optimal weights and bias values during training.
- **Gradient Descent**: A method used to minimize the error between predicted and actual values by adjusting the model parameters.
- **Prediction**: After training, the model predicts the target values by computing the linear combination of the input features and learned weights.

**Advantages**:
- Simple and easy to implement.
- Provides interpretable coefficients that describe the relationship between features and the target.
- Performs well when the relationship between features and target is approximately linear.

**Disadvantages**:
- Assumes a linear relationship between features and target, which may not hold for all datasets.
- Sensitive to outliers, as they can heavily influence the line of best fit.
- Requires features to be scaled or normalized for optimal performance (especially when regularization is applied).

**Customization**:
- You can adjust the learning rate and the number of iterations to improve convergence speed and stability.
