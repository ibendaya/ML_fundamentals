### Recurrent Neural Network (RNN)

**Description**:  
A Recurrent Neural Network (RNN) is a type of neural network designed for processing sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form cycles within the network, allowing them to maintain a hidden state that can capture information from previous time steps. This makes RNNs particularly well-suited for tasks involving sequential data, such as time series forecasting, language modeling, and text classification.

**How It Works**:

1. **Basic RNN Structure**:  
   The basic unit of an RNN is composed of an input layer, a hidden layer, and an output layer. Each unit in the hidden layer receives input from both the current input and the previous hidden state. This allows the network to capture temporal dependencies in the data.

   In the implementation:
   - **i2h** (input to hidden): A linear layer that takes the current input and the previous hidden state and computes the new hidden state.
   - **i2o** (input to output): A linear layer that generates the output from the current input and the hidden state.
   - **softmax**: A softmax activation function applied to the output to ensure the predictions sum to 1 (for classification tasks).

2. **Forward Pass**:  
   At each time step in a sequence, the RNN updates its hidden state based on the current input and the previous hidden state. The output for each time step is computed based on this updated hidden state. The entire sequence is processed iteratively, with the final output being generated after the last time step.

3. **Hidden State Initialization**:  
   The hidden state is initialized as a tensor of zeros, and it is updated at each time step based on the input and previous state. The hidden state allows the RNN to retain memory of previous inputs, which is crucial for processing sequential data.

4. **Training**:  
   During training, the model uses the **Negative Log Likelihood Loss (NLLLoss)** function to compute the error between the predicted output and the true category. The optimizer used is **Stochastic Gradient Descent (SGD)**, and the model is trained using backpropagation through time (BPTT), which is a variant of backpropagation tailored for sequential data.

5. **Prediction**:  
   After training, the model can predict the category of a new input by iterating through the input sequence, updating the hidden state at each step, and using the final hidden state to make a classification decision.

**Key Components**:

- **Hidden Layer**: The hidden layer in an RNN retains information from previous time steps, allowing the network to learn dependencies across time.
- **Input and Output Layers**: The input layer receives the data at each time step, while the output layer produces the predictions for classification.
- **Activation Function**: A **LogSoftmax** activation is applied to the output to ensure the outputs are probabilities, especially useful for classification tasks with multiple classes.

**Training and Evaluation**:
- The model is trained using **cross-entropy loss** (Negative Log Likelihood Loss) and optimized with **SGD**.
- The training process involves iterating through sequences of data, updating weights through backpropagation, and reducing the loss after each update.

**Customization**:
- `lr`: The learning rate controls how much the model weights are updated during training.
- `n_iters`: The number of training iterations. More iterations generally result in better model performance.
- `hidden_size`: The number of hidden units in the network, which controls the capacity of the model to store temporal information.

**Advantages**:
- Can handle sequences of arbitrary length and capture dependencies over time.
- Suitable for tasks involving sequential or time-dependent data, such as language modeling, speech recognition, and more.

**Disadvantages**:
- Training RNNs can be computationally expensive, especially with long sequences.
- Standard RNNs can struggle with long-range dependencies due to the vanishing gradient problem. This can be mitigated by using more advanced architectures like LSTMs or GRUs.

**Applications**:
- Language modeling and text generation.
- Time series forecasting and prediction.
- Speech recognition and sequence labeling.

---

This implementation provides a basic but powerful framework for processing sequential data using an RNN. It is particularly useful for tasks like text classification and language modeling. With customization options like learning rate and hidden state size, this RNN can be adapted for a variety of sequence-based tasks.
