# Artificial Neural Networks and Forward Propagation Summary

## 1. Introduction to Deep Learning

Deep learning is currently one of the most prominent fields in data science, enabling a variety of advanced applications:

* **Color Restoration:** Automatically converting grayscale images into colored ones.
* **Speech Enactment:** Synthesizing audio clips with synchronized lip movements in videos.
* **Handwriting Generation:** Creating highly realistic cursive handwriting in various styles.
* **Pattern Recognition:** Mimicking human decision-making and recognition capabilities.

## 2. Biological Inspiration

Artificial Neural Networks (ANNs) are inspired by the structure and function of the human brain:

* **Soma:** The main body of a biological neuron.
* **Dendrites:** An extensive network of arms that receive electrical impulses from other neurons and carry them to the soma.
* **Axon:** A long arm that carries processed information from the nucleus to the synapses.
* **Synapses:** Whiskers at the end of the axon that transmit output to other neurons.
* **Learning:** In the brain, learning occurs by repeatedly activating and reinforcing specific neural connections.

## 3. Artificial Neural Network (ANN) Architecture

An artificial neuron mimics its biological counterpart through a structured mathematical approach:

* **Input Layer:** The entry point for data where each node represents a specific feature or variable.
* **Hidden Layers:** Layers between input and output that perform pattern extraction. A "Deep" network contains multiple hidden layers.
* **Output Layer:** The final layer that produces the prediction or classification.
* **Weights ($w$):** Represent the strength and influence of a connection between neurons.
* **Biases ($b$):** A threshold added to the calculation to allow the model to shift activation functions for better data fitting.

## 4. The Forward Propagation Process

Forward propagation is the process of data moving from the input layer to the output layer to make a prediction.

### Mathematical Steps:

1. **Weighted Sum ($z$):** For each neuron, calculate the dot product of inputs and weights, then add the bias:
   $$
   z = \sum (weight \cdot input) + bias
   $$
2. **Activation ($a$):** Apply a non-linear transformation (like the **Sigmoid function**) to the weighted sum:
   $$
   a = \frac{1}{1 + e^{-z}}
   $$
3. **Layer Transition:** The output of one layer becomes the input for the next until the final prediction is reached.

## 5. The Learning Cycle

While forward propagation generates predictions, the network "learns" through:

* **Error Calculation:** Measuring the difference between the prediction and the actual truth.
* **Backpropagation:** Moving backward through the network to update weights and biases to minimize future errors.

## 6. Implementation Summary (from Lab)

The provided lab demonstrated how to:

* **Initialize a Network:** Use Python and Numpy to create a dictionary-based network structure with random weights and biases for any number of inputs and layers.
* **Automate Predictions:** Create functions (`compute_weighted_sum`, `node_activation`, and `forward_propagate`) to handle complex networks efficiently rather than manual calculation.

$$
a = \frac{1}{1 + e^{-z}}
$$
