# Keras Functional API & Custom Layers — Concept Summary

---

## 1. Overview

| Topic | Key Idea |
|-------|---------|
| **Keras** | High-level neural network API in Python; runs on TensorFlow, Theano, CNTK |
| **Functional API** | Flexible model-building approach; handles multiple inputs/outputs and shared layers |
| **Custom Layers** | Extend Keras to implement novel research ideas and task-specific behavior |
| **TensorFlow 2.x** | Powers Keras with eager execution, high-level APIs, and rich ecosystem |

Keras is used across healthcare, finance, autonomous driving, image/speech recognition, NLP, and recommendation systems.

---

## 2. Keras Functional API

### Sequential vs Functional API

| | Sequential API | Functional API |
|--|---------------|----------------|
| Structure | Linear stack of layers | Graph of layers |
| Multiple inputs/outputs | ❌ | ✅ |
| Shared layers | ❌ | ✅ |
| Flexibility | Limited | High |

### Building a Model — Step by Step

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Define input
input_layer = Input(shape=(20,))

# Step 2: Add hidden layers
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

# Step 3: Define output
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Step 4: Create model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
```

- Each layer is **called as a function** on the previous layer's output
- `Model(inputs=..., outputs=...)` explicitly defines the computation graph

### Compile, Train, Evaluate

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)
```

---

## 3. Regularization Techniques

### Dropout

Randomly sets a fraction of neurons to zero during training to prevent overfitting.

- Applied during **training only**, not inference
- `rate` hyperparameter controls the fraction of neurons dropped

```python
from tensorflow.keras.layers import Dropout

hidden = Dense(64, activation='relu')(input_layer)
dropped = Dropout(rate=0.5)(hidden)
```

### Batch Normalization

Normalizes layer outputs to have mean 0 and variance 1, stabilizing and speeding up training.

- Reduces **internal covariate shift**
- Allows higher learning rates → faster convergence
- Applied during both training and inference (behavior differs slightly)
- Introduces two learnable parameters: **scale (γ)** and **shift (β)**

```python
from tensorflow.keras.layers import BatchNormalization

hidden = Dense(64, activation='relu')(input_layer)
normed = BatchNormalization()(hidden)
```

### Combined Example

```python
input_layer = Input(shape=(20,))
x = Dense(64, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=output_layer)
```

---

## 4. Custom Layers

Creating custom layers allows you to:
- Tailor models to specific needs
- Implement novel research ideas
- Optimize performance for unique tasks

### Structure of a Custom Layer

Subclass `tf.keras.layers.Layer` and implement three methods:

| Method | Purpose |
|--------|---------|
| `__init__` | Store configuration (e.g. number of units) |
| `build` | Create layer weights using `add_weight()` |
| `call` | Define the forward pass computation |

### Example: `CustomDenseLayer`

```python
from tensorflow.keras.layers import Layer
import tensorflow as tf

class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
```

- `add_weight()` registers tensors as trainable parameters
- `call()` computes `ReLU(X·W + b)` — same as a standard Dense layer with ReLU

### Model Architecture with Custom Layers

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax

model = Sequential([
    CustomDenseLayer(128),   # Input (1000, 20) → Output (1000, 128)
    CustomDenseLayer(10),    # Input (1000, 128) → Output (1000, 10)
    Softmax()                # Input (1000, 10)  → Output (1000, 10)
])
```

Visual architecture (from `plot_model`):

```
Input (1000, 20)
      ↓
CustomDenseLayer → Output (1000, 128)
      ↓
CustomDenseLayer → Output (1000, 10)
      ↓
Softmax → Output (1000, 10)
```

### Compile, Train, Evaluate

```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.build((1000, 20))

# One-hot encode labels for multi-class classification
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

model.fit(x_train, y_train, epochs=10, batch_size=32)
loss = model.evaluate(x_test, y_test)
```

- Loss: **Categorical Crossentropy** (multi-class, one-hot labels)
- Output activation: **Softmax** (probabilities sum to 1)

### Visualize Model Architecture

```python
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
```

### Add Dropout to Custom Model

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10),
    Dropout(0.5),
    Softmax()
])
```

---

## 6. Glossary

| Term | Definition |
|------|-----------|
| **Build** | A method that creates the layer's weights, called once during the first invocation of the layer |
| **Call** | A method that defines the forward pass logic of the layer |
| **Custom layer** | A user-defined layer that allows customization of operations in a neural network, providing flexibility for specific tasks and experimentation |
| **Eager execution** | A TensorFlow feature that executes operations immediately, making it more intuitive and useful for debugging and interactive programming |
| **Init** | A method that initializes the layer's attributes |
| **Input layer** | The first layer in a model that defines the input shape |
| **Keras** | A high-level neural network API written in Python that can run on top of TensorFlow, Theano, and CNTK |
| **Keras Functional API** | A powerful API for creating complex models with multiple inputs and outputs, shared layers, and non-sequential data flows |
| **Keras Sequential API** | Creates models with layers in a linear stack |
| **ReLU** | An activation function that outputs the input directly if positive; otherwise outputs zero. Commonly used in hidden layers |
| **Shared layer** | Helpful when applying the same transformation to multiple inputs |
| **Softmax** | An activation function suitable for classification tasks; outputs a probability distribution summing to 1 |
| **TensorFlow 2.x** | An open-source ML platform by Google for building and deploying models across servers to edge devices |
| **TensorBoard** | A visualization toolkit for TensorFlow providing insights into training metrics, graphs, and other data |
| **TensorFlow Extended (TFX)** | An end-to-end platform for deploying production ML pipelines with tools for deployment, monitoring, and management |
| **TensorFlow Hub** | A repository of reusable ML modules easily integrated into TensorFlow applications to accelerate development |
| **TensorFlow.js** | A library for training and deploying ML models in JavaScript environments such as web browsers and Node.js |
| **TensorFlow Lite** | A lightweight framework for deploying ML models on mobile and embedded devices |

---

## 7. Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Functional API** | Builds models as a graph; supports multi-input/output and shared layers |
| **`Model(inputs, outputs)`** | Explicitly defines the model's computation graph |
| **`layer(previous_output)`** | Functional API syntax — call a layer as a function |
| **Dropout** | Randomly drops neurons during training; prevents overfitting |
| **Batch Normalization** | Normalizes activations; stabilizes training; allows higher learning rates |
| **Custom Layer** | Subclass `Layer`; implement `build()` for weights and `call()` for forward pass |
| **`add_weight()`** | Registers a tensor as a trainable parameter inside a custom layer |
| **Softmax** | Output activation for multi-class classification; outputs probability distribution |
| **`to_categorical()`** | Converts integer class labels to one-hot encoded arrays |
| **`plot_model()`** | Visualizes model architecture as an image with layer shapes |