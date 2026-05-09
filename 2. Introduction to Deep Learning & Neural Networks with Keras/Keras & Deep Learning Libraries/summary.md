# Deep Learning with Keras — Concept Summary

---

## 1. Deep Learning Libraries Overview

Three major deep learning libraries are widely used today:

| Library              | Key Characteristics                                                      |
| -------------------- | ------------------------------------------------------------------------ |
| **TensorFlow** | Used in production environments; very large community; low-level control |
| **PyTorch**    | Based on Torch (Lua); GPU support; preferred in academic research        |
| **Keras**      | High-level API; runs on top of TensorFlow; easy to use; fast development |

TensorFlow and PyTorch are powerful but have a  **steep learning curve** . Keras abstracts away complexity, allowing complex deep learning networks to be built with only a few lines of code. Keras normally runs on top of TensorFlow as its backend.

> To use Keras, install TensorFlow first. TensorFlow 2.16+ installs Keras by default.

---

## 2. Data Preparation

Before building any model in Keras, data must be properly prepared.

### General steps

* Split data into **predictors** (features) and **target** (label)
* Check for missing values
* **Normalize** the data: subtract the mean and divide by the standard deviation

```python
predictors_norm = (predictors - predictors.mean()) / predictors.std()
```

### For classification problems

* The target column must be transformed into **one-hot encoded binary arrays** using `to_categorical`:

```python
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
```

### For image data (MNIST)

* Images (28×28 pixels) must be **flattened** into 1D vectors of size 784
* Pixel values must be **normalized** from [0, 255] to [0, 1]

```python
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
```

---

## 3. Building a Model in Keras

Both regression and classification models follow the same structure using the `Sequential` API:

```python
from keras.models import Sequential
from keras.layers import Dense
```

### Regression model — Concrete Strength dataset

Predicts a **continuous value** (compressive strength of concrete).

```python
def regression_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

| Layer    | Nodes  | Activation    |
| -------- | ------ | ------------- |
| Input    | n_cols | —            |
| Hidden 1 | 50     | ReLU          |
| Hidden 2 | 50     | ReLU          |
| Output   | 1      | linear (none) |

* No activation on output → predicts raw continuous value
* Loss: **Mean Squared Error (MSE)**

---

### Classification model — MNIST dataset

Classifies **handwritten digits** (0–9) from images.

```python
def classification_model():
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

| Layer    | Nodes            | Activation |
| -------- | ---------------- | ---------- |
| Hidden 1 | num_pixels (784) | ReLU       |
| Hidden 2 | 100              | ReLU       |
| Output   | num_classes (10) | Softmax    |

* **Softmax** on output → converts raw scores into probabilities that sum to 1, one per class
* Loss: **Categorical Crossentropy**
* Metric: **Accuracy**

---

## 4. Regression vs Classification — Key Differences

|                   | Regression         | Classification               |
| ----------------- | ------------------ | ---------------------------- |
| Target            | Continuous value   | Discrete class               |
| Output nodes      | 1                  | num_classes                  |
| Output activation | None (linear)      | Softmax                      |
| Loss function     | Mean Squared Error | Categorical Crossentropy     |
| Metric            | —                 | Accuracy                     |
| Target encoding   | Raw values         | One-hot (`to_categorical`) |

---

## 5. Training and Evaluating

```python
# Regression — with validation split
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Classification — with explicit test set
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# Evaluate
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}%  Error: {}'.format(scores[1], 1 - scores[1]))
```

* `validation_split=0.3` → reserve 30% of training data for validation
* Monitor both `loss` and `val_loss` to detect **overfitting** (gap between the two widening)

---

## 6. Saving and Loading a Model

```python
# Save
model.save('classification_model.keras')

# Load
from keras.models import load_model
pretrained_model = load_model('classification_model.keras')
```

Saving is useful when training is expensive and you want to reuse the model without retraining.

---

## 7. Key Takeaways

* Keras makes building deep learning models fast and accessible with minimal code.
* Always normalize input data before training.
* Use `to_categorical` to one-hot encode target labels for classification tasks.
* Choose loss function and output activation based on the problem type: MSE + linear for regression, categorical crossentropy + softmax for multi-class classification.
* Watch the gap between `loss` and `val_loss` during training to detect overfitting early.
