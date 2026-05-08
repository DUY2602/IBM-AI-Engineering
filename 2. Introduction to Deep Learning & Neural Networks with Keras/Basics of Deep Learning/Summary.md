# Backpropagation & Neural Network Training — Concept Summary

---

## 1. Gradient Descent

Gradient descent is an iterative optimization algorithm used to find the minimum of a function (i.e., minimize the network's error). The **learning rate** controls the step size at each iteration:

- **Too large** → big steps that may overshoot and miss the minimum.
- **Too small** → tiny steps that make convergence extremely slow.

---

## 2. Neural Network Training Loop

Weights and biases are initialized to random values, then the following cycle repeats until a stopping condition is met:

1. **Forward Propagation** — compute the network output from the current weights.
2. **Error Calculation** — measure the difference between the predicted output and the ground truth.
3. **Backpropagation** — propagate the error backward and update weights/biases.
4. **Repeat** until the number of epochs is reached or the error falls below a threshold.

---

## 3. Backpropagation in Detail (from the Notebook)

The notebook implements backpropagation on the **XOR problem** using a small two-layer network. The five stages are:

| Stage | What Happens |
|---|---|
| **Forward Pass** | Input → hidden layer (weighted sum + sigmoid) → output layer (weighted sum + sigmoid) |
| **Error Calculation** | `error = d - a2` (target minus prediction) |
| **Output Layer Gradient** | `da2 = error × sigmoid′(a2)` |
| **Hidden Layer Gradient** | `da1 = W2ᵀ · dz2`, then `dz1 = da1 × sigmoid′(a1)` |
| **Weight & Bias Update** | `W += lr × (gradient · activations)` for each layer |

### Key Implementation Details
- Weights initialized randomly in `[-1, 1]`.
- Learning rate `lr = 0.1`, trained for `180,000` epochs by default.
- A lower learning rate (`lr = 0.01`) requires more epochs (`1,000,000`) to converge.
- Error is tracked every 10,000 epochs and plotted to visualize learning progress.

---

## 4. The Vanishing Gradient Problem

A critical challenge in training deep networks:

- Sigmoid-based networks multiply gradients together during backpropagation. Because sigmoid derivatives are always less than 1, gradients shrink exponentially as they travel backward through layers.
- **Effect**: neurons in earlier layers receive very small gradients and learn much more slowly than neurons in later layers.
- **Consequence**: training takes too long, and prediction accuracy is compromised.
- **Solution**: avoid sigmoid (and similar saturating functions) as hidden-layer activations; use alternatives like ReLU instead.

---

## 5. Activation Functions

Activation functions play a major role in how well a neural network trains and generalizes.

| Function | Key Characteristics |
|---|---|
| **Sigmoid** | Outputs (0, 1); prone to vanishing gradients — avoid in hidden layers. |
| **Tanh (Hyperbolic Tangent)** | Scaled sigmoid symmetric around the origin; outputs (−1, 1); still susceptible to vanishing gradients in very deep nets. |
| **ReLU** | Most widely used today; does not activate all neurons simultaneously; largely avoids vanishing gradients. |
| **Softmax** | Used in the output layer for multi-class classification; converts raw scores to class probabilities. |

Seven activation functions can be used to build a neural network; ReLU is the recommended default for hidden layers due to its efficiency and gradient properties.

---

## 6. Summary of Key Takeaways

- Gradient descent + backpropagation is the backbone of neural network learning.
- Choosing the right learning rate is critical — balance speed vs. stability.
- Backpropagation works by chaining gradients from output back to input (chain rule).
- The vanishing gradient problem slows or stalls training in deep networks with sigmoid/tanh activations.
- ReLU is the preferred activation function for hidden layers; softmax is standard for classification outputs.
- Training progress can be monitored by plotting error over epochs.