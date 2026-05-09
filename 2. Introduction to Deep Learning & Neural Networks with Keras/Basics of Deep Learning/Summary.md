
# Backpropagation, Neural Network Training & Activation Functions — Concept Summary

---

## 1. Gradient Descent

Gradient descent is an iterative optimization algorithm used to find the minimum of a function (i.e., minimize the network's error). The **learning rate** controls the step size at each iteration:

* **Too large** → big steps that may overshoot and miss the minimum.
* **Too small** → tiny steps that make convergence extremely slow.

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

| Stage                           | What Happens                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------- |
| **Forward Pass**          | Input → hidden layer (weighted sum + sigmoid) → output layer (weighted sum + sigmoid) |
| **Error Calculation**     | `error = d - a2`(target minus prediction)                                             |
| **Output Layer Gradient** | `da2 = error × sigmoid′(a2)`                                                        |
| **Hidden Layer Gradient** | `da1 = W2ᵀ · dz2`, then `dz1 = da1 × sigmoid′(a1)`                              |
| **Weight & Bias Update**  | `W += lr × (gradient · activations)`for each layer                                  |

### Key Implementation Details

* Weights initialized randomly in `[-1, 1]`.
* Learning rate `lr = 0.1`, trained for `180,000` epochs by default.
* A lower learning rate (`lr = 0.01`) requires more epochs (`1,000,000`) to converge.
* Error is tracked every 10,000 epochs and plotted to visualize learning progress.

---

## 4. The Vanishing Gradient Problem

A critical challenge in training deep networks:

* Sigmoid-based networks multiply gradients together during backpropagation. Because sigmoid derivatives are always less than 1, gradients shrink exponentially as they travel backward through layers.
* **Effect** : neurons in earlier layers receive very small gradients and learn much more slowly than neurons in later layers.
* **Consequence** : training takes too long, and prediction accuracy is compromised.
* **Solution** : avoid sigmoid (and similar saturating functions) as hidden-layer activations; use alternatives like ReLU instead.

---

## 5. Activation Functions

Activation functions play a major role in how well a neural network trains and generalizes. Below are the key functions covered across both notebooks.

---

### 5.1 Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

* Outputs values in the range **(0, 1)** — S-shaped curve.
* Suitable for **binary classification** output layers and probabilistic outputs.
* Continuous, differentiable, and monotonically increasing.
* Derivative: `σ′(z) = σ(z) · (1 − σ(z))` — simple to compute, essential for backpropagation.

**Limitations:**

* **Vanishing gradient** : for large positive or negative inputs, the gradient approaches 0, stalling learning in deep networks.
* Computationally expensive due to the exponential operation.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
```

---

### 5.2 ReLU (Rectified Linear Unit)

$$
f(x) = \max(0, x)
$$

* Outputs **0** for negative inputs, and the **input value** for positive inputs.
* Derivative: `f′(x) = 1` for `x > 0`, and `f′(x) = 0` for `x ≤ 0`.
* **Most widely used** activation function for hidden layers today.
* Does not activate all neurons at the same time → sparse, efficient representations.
* Maintains a constant gradient of 1 for positive inputs → largely **avoids vanishing gradients** and enables faster convergence.

**Limitations:**

* **Dead neurons** : neurons can get stuck outputting 0 permanently if they receive consistently negative inputs. Variants like **Leaky ReLU** mitigate this.

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
```

---

### 5.3 Tanh (Hyperbolic Tangent)

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

* A  **scaled version of sigmoid** , symmetric around the origin.
* Outputs values in **(−1, 1)** — better than sigmoid for hidden layers since it is zero-centered.
* Derivative: `tanh′(z) = 1 − tanh²(z)`.
* Still susceptible to vanishing gradients in very deep networks for large input values.

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
```

---

### 5.4 Softmax

* Used in the **output layer** of multi-class classifiers.
* Converts raw scores into a **probability distribution** over classes (all outputs sum to 1).
* Each class probability represents the likelihood that the input belongs to that class.

---

### Comparison Summary

| Function          | Output Range      | Vanishing Gradient | Best Used For                       |
| ----------------- | ----------------- | ------------------ | ----------------------------------- |
| **Sigmoid** | (0, 1)            | Yes                | Binary output layer                 |
| **Tanh**    | (−1, 1)          | Partial            | Hidden layers (older architectures) |
| **ReLU**    | [0, ∞)           | No (for x > 0)     | Hidden layers (default choice)      |
| **Softmax** | (0, 1), sums to 1 | N/A                | Multi-class output layer            |

Seven activation functions can be used to build a neural network; **ReLU** is the recommended default for hidden layers due to its efficiency and gradient properties.

---

## 6. Summary of Key Takeaways

* Gradient descent + backpropagation is the backbone of neural network learning.
* Choosing the right learning rate is critical — balance speed vs. stability.
* Backpropagation works by chaining gradients from output back to input (chain rule).
* The vanishing gradient problem slows or stalls training in deep networks with sigmoid/tanh activations.
* ReLU is the preferred activation function for hidden layers; softmax is standard for classification outputs.
* Training progress can be monitored by plotting error over epochs.
