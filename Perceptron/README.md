### Perceptron

**Description**:  
The Perceptron is one of the simplest types of artificial neural networks, used for binary classification tasks. It is a linear classifier that learns a decision boundary to separate two classes by adjusting its weights and bias during training. The Perceptron is a foundational model in machine learning and forms the basis for more advanced neural networks.

**How It Works**:

1. **Model Representation**:  
   The Perceptron model is represented by a set of weights \( w \) and a bias \( b \). Given an input vector \( X \), the model computes a weighted sum of the inputs:

   \[
   \text{output} = X \cdot w + b
   \]

   The output is passed through an **activation function**, which in the case of the Perceptron is the **unit step function**. This function outputs 1 if the weighted sum is greater than 0, and 0 otherwise:

   \[
   \text{unit step function}(x) = 
   \begin{cases} 
   1 & \text{if } x > 0 \\
   0 & \text{if } x \leq 0 
   \end{cases}
   \]

2. **Training**:  
   The Perceptron is trained using a supervised learning process where it adjusts the weights and bias to minimize classification errors. The training algorithm uses the following steps:
   - For each training example, compute the output using the current weights and bias.
   - Compare the predicted output with the actual label.
   - If there is an error (i.e., the predicted output does not match the target), update the weights and bias:
   
   \[
   \Delta w = \eta \cdot (y_{\text{true}} - y_{\text{pred}}) \cdot X
   \]
   \[
   \Delta b = \eta \cdot (y_{\text{true}} - y_{\text{pred}})
   \]

   where:
   - \( \eta \) is the learning rate,
   - \( y_{\text{true}} \) is the true label,
   - \( y_{\text{pred}} \) is the predicted label.

3. **Learning Rate**:  
   The learning rate (\( \eta \)) controls the step size when updating the weights and bias. A small learning rate leads to slow learning, while a large rate may cause the model to overshoot and fail to converge.

4. **Convergence**:  
   The Perceptron algorithm continues iterating over the dataset for a specified number of iterations (`iters`) or until the weights stabilize and no further updates are needed.

5. **Prediction**:  
   After training, the model can predict the class of new samples by computing the linear output and applying the unit step function.

**Key Components**:

- **Weights and Bias**: The model learns a set of weights and a bias term during training that help separate the two classes.
- **Unit Step Activation**: The unit step function is used to classify inputs based on the linear output of the model.
- **Learning Rule**: The Perceptron adjusts its weights using the difference between the predicted and true values to minimize errors.

**Advantages**:
- Simple and easy to implement.
- Works well for linearly separable data.
- Provides a good foundation for understanding more advanced neural networks.

**Disadvantages**:
- Limited to linearly separable problems and cannot model non-linear decision boundaries.
- Sensitive to the choice of learning rate and the number of iterations.
- May require feature scaling for optimal performance.

**Customization**:
- `lr`: The learning rate, which can be adjusted to control how quickly the model updates its parameters during training.
- `iters`: The number of iterations to run during training. Increasing this can help improve the model's accuracy if the data is more complex.

---

This implementation provides a basic binary classifier using the Perceptron algorithm. Despite its simplicity, it demonstrates key concepts used in neural networks and can serve as a starting point for more complex models.
