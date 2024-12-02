# Machine Learning Fundamentals 

This project showcases my implementations of fundamental machine learning algorithms and concepts. Each algorithm is implemented from scratch using mostly Python, numpy, and in the case of RNNs and GANs: pytorch, with explanations and visualizations provided to aid understanding. The goal of this repository is to cement my understand of the fundamentals of ML.

## Algorithms Implemented

### Supervised Learning

1. **Decision Trees**  
   Decision trees are a popular model for classification and regression tasks. They recursively partition the feature space into regions that are as pure as possible, based on the target variable. This implementation uses recursive splitting and the information gain criterion to build a decision tree from scratch.

2. **Linear Regression**  
   Linear regression is used for predicting continuous values. It models the relationship between input features and the target variable by fitting a linear equation to the data. This implementation uses gradient descent to optimize the weights.

3. **Logistic Regression**  
   Logistic regression is used for binary classification tasks. It models the probability that a given input belongs to a particular class by applying the sigmoid function to a linear combination of the input features. This implementation uses gradient descent for optimization.

4. **K-Nearest Neighbors (KNN)**  
   KNN is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their `k` nearest neighbors. This implementation uses Euclidean distance to compute similarity and classify data points.

5. **Naive Bayes**  
   Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption that the features are conditionally independent. This implementation uses Gaussian Naive Bayes, where the features are assumed to follow a normal distribution.

6. **Support Vector Machines (SVM)**  
   SVM is a powerful classification algorithm that finds the optimal hyperplane that maximally separates data points from different classes. This implementation uses a linear SVM with gradient descent and L2 regularization.

7. **Random Forest**  
   Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. This implementation uses bootstrap aggregation (bagging) and random feature selection for training each tree.

### Unsupervised Learning

1. **K-Means Clustering**  
   K-Means is an unsupervised learning algorithm for clustering data into `k` groups based on similarity. Each cluster is represented by its centroid, and data points are assigned to the closest centroid.

2. **Principal Component Analysis (PCA)**  
   PCA is a dimensionality reduction technique that transforms data into a new coordinate system, where the greatest variance lies along the first few components. This implementation computes the covariance matrix, eigenvalues, and eigenvectors to reduce data dimensions while retaining the most significant variance.


### Neural Networks

1. **Perceptron**  
   The Perceptron is a simple neural network used for binary classification tasks. It is a linear classifier that updates its weights based on classification errors using the perceptron learning rule.

2. **Recurrent Neural Networks (RNN)**  
   RNNs are designed for sequential data processing, where each output is dependent on previous outputs. This implementation builds a basic RNN with a single hidden layer and softmax output for sequence classification.

## Potential TODOs

- **Support Vector Machine (SVM) with Kernel Trick**  
   Extend the linear SVM to use the kernel trick, enabling SVMs to handle non-linear decision boundaries by mapping data into higher-dimensional spaces.

- **Gradient Boosting Machines (GBM)**  
   Gradient Boosting is an ensemble learning algorithm that builds decision trees sequentially, where each tree corrects the errors of the previous one. This could be implemented using decision trees as weak learners.

- **k-Nearest Neighbors (KNN) with Custom Distance Metrics**  
   Extend the KNN implementation to support various distance metrics (e.g., Manhattan distance, Minkowski distance).

- **Convolutional Neural Networks (CNNs)**  
   CNNs are specialized deep learning networks for processing image data, where they use convolutional layers to automatically learn spatial hierarchies in the data.

- **Long Short-Term Memory (LSTM)**  
   LSTMs are a type of RNN designed to address the vanishing gradient problem by using memory cells to retain information for longer periods. They are particularly useful for long-term sequence learning.

- **Autoencoders**  
   Autoencoders are unsupervised neural networks used for dimensionality reduction and anomaly detection. They learn to compress the input data into a latent space and then reconstruct it back to the original input.
