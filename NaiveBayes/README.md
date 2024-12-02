### Naive Bayes

**Description**:  
Naive Bayes is a family of probabilistic classifiers based on applying **Bayes' theorem** with strong (naive) independence assumptions between the features. It is widely used for classification tasks, particularly for text classification, where the assumption of feature independence is often reasonable. Despite its simplicity, Naive Bayes performs surprisingly well in many real-world tasks.

The algorithm calculates the probability of each class based on the features of the data, and the class with the highest posterior probability is selected as the predicted class.

**How It Works**:

1. **Bayes' Theorem**:  
   Naive Bayes classifiers use **Bayes' theorem** to predict the probability of a class given a set of features:

   \[
   P(C|X) = \frac{P(X|C) P(C)}{P(X)}
   \]

   where:
   - \( P(C|X) \) is the posterior probability of class \( C \) given the features \( X \),
   - \( P(X|C) \) is the likelihood of observing features \( X \) given class \( C \),
   - \( P(C) \) is the prior probability of class \( C \),
   - \( P(X) \) is the probability of the features (constant across all classes and can be ignored during classification).

2. **Naive Assumption**:  
   The "naive" assumption in Naive Bayes is that the features are conditionally independent given the class. This simplifies the computation of \( P(X|C) \) as the product of the individual probabilities of each feature:

   \[
   P(X|C) = P(x_1|C) \cdot P(x_2|C) \cdot \dots \cdot P(x_n|C)
   \]

3. **Gaussian Naive Bayes**:  
   In this implementation, we assume that the features follow a **Gaussian distribution** for each class. The likelihood of each feature \( x_i \) is calculated using the probability density function (PDF) of the Gaussian distribution:

   \[
   P(x_i|C) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x_i - \mu)^2}{2 \sigma^2} \right)
   \]

   where:
   - \( \mu \) is the mean of the feature for class \( C \),
   - \( \sigma^2 \) is the variance of the feature for class \( C \).

4. **Training**:  
   During training, the algorithm computes the following for each class:
   - The **mean** and **variance** of each feature for each class,
   - The **prior** probability of each class, which is simply the proportion of samples in each class.

5. **Prediction**:  
   For prediction, the algorithm calculates the posterior probability for each class using the learned means, variances, and priors, and selects the class with the highest posterior probability.

**Key Components**:

- **Class Means and Variances**: The model computes the mean and variance of each feature for each class in the training data.
- **Prior Probabilities**: The prior probability of each class is calculated based on the frequency of each class in the training data.
- **Likelihood Calculation**: The likelihood of the features given a class is calculated using the Gaussian PDF.
- **Posterior Probability**: The class with the highest posterior probability is predicted for each sample.

**Advantages**:
- Simple and easy to implement.
- Efficient with a small training dataset and fast during prediction.
- Works well for high-dimensional data (e.g., text classification, spam filtering).
- Performs well even with the naive independence assumption when the features are not strongly correlated.

**Disadvantages**:
- The naive assumption of feature independence is often unrealistic, which can lead to suboptimal performance in some cases.
- Sensitive to irrelevant features, which may negatively affect the model's performance.

**Customization**:
- This implementation works with continuous features, assuming they follow a Gaussian distribution. For categorical features, you may need to modify the likelihood computation.
  
---

This implementation provides a straightforward approach to classification with Naive Bayes, particularly for datasets where features are conditionally independent and normally distributed. Despite its simplicity, Naive Bayes often serves as a strong baseline for many classification tasks.
