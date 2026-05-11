# Transfer Learning & Advanced Data Augmentation — Concept Summary

---

## 1. Transfer Learning Overview

Transfer learning reuses a model pre-trained on a large dataset (e.g. ImageNet) for a new task, significantly saving time and computational resources.

### Key Tips for Effective Transfer Learning

| Tip | Details |
|-----|---------|
| **Choose the right model** | Select a model trained on data similar to your target task (VGG16, ResNet, InceptionV3 for images) |
| **Freeze early layers** | Preserve low-level features in the initial training stage; useful for small datasets |
| **Fine-tune later layers** | Gradually unfreeze deeper layers to capture task-specific features |
| **Lower learning rate** | Use a lower learning rate during fine-tuning to prevent catastrophic forgetting |
| **Data augmentation** | Increase dataset variability to prevent overfitting and improve generalization |
| **Domain adaptation** | Apply adaptation techniques when source and target domains differ significantly |

---

## 2. Transfer Learning with VGG16

### Step 1 — Load Pre-trained Model

```python
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all base model layers
for layer in base_model.layers:
    layer.trainable = False
```

- `include_top=False` → removes the original ImageNet classification head
- Frozen layers act as a fixed **feature extractor**

### Step 2 — Add Custom Classifier

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')     # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

| Layer | Details |
|-------|---------|
| VGG16 base | Frozen, pretrained feature extractor |
| Flatten | Converts 3D feature maps to 1D |
| Dense(256, ReLU) | Custom hidden layer |
| Dense(1, Sigmoid) | Binary output: class A vs class B |

### Step 3 — Train on New Dataset

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

model.fit(train_generator, epochs=10)
```

### Step 4 — Fine-Tune Top Layers

After initial training, unfreeze the last few layers of VGG16 and retrain at a lower learning rate:

```python
# Unfreeze the last 4 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile and retrain
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)
```

> Fine-tuning allows the deeper layers to adapt to the nuances of the new dataset while preserving the low-level features learned from ImageNet.

### Validation Split in Training

```python
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'sample_data', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='validation'
)

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

Monitor both `loss` and `val_loss` to detect overfitting.

---

## 3. Advanced Data Augmentation with Keras

Data augmentation artificially increases dataset size by applying random transformations, helping models generalize better.

### 3.1 Basic Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,        # Rotate up to 40 degrees
    width_shift_range=0.2,    # Shift horizontally up to 20%
    height_shift_range=0.2,   # Shift vertically up to 20%
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Zoom in/out up to 20%
    horizontal_flip=True,     # Randomly flip horizontally
    fill_mode='nearest'       # Fill empty pixels with nearest value
)
```

| Parameter | Effect |
|-----------|--------|
| `rotation_range` | Rotates image by random angle |
| `width/height_shift_range` | Translates image horizontally/vertically |
| `shear_range` | Applies shear transformation |
| `zoom_range` | Zooms in or out |
| `horizontal_flip` | Mirrors image left-right |
| `fill_mode` | How to fill pixels after transformation |

### 3.2 Feature-wise & Sample-wise Normalization

```python
datagen = ImageDataGenerator(
    featurewise_center=True,             # Subtract dataset mean (per feature)
    featurewise_std_normalization=True,  # Divide by dataset std (per feature)
    samplewise_center=True,              # Subtract sample mean
    samplewise_std_normalization=True    # Divide by sample std
)

datagen.fit(x_train)    # Required for featurewise operations
```

| Technique | Scope | Effect |
|-----------|-------|--------|
| `featurewise_center` | Whole dataset | Subtracts mean of each feature |
| `featurewise_std_normalization` | Whole dataset | Divides by std of each feature |
| `samplewise_center` | Per image | Subtracts mean of each individual image |
| `samplewise_std_normalization` | Per image | Divides by std of each individual image |

### 3.3 Custom Augmentation Function

`preprocessing_function` accepts any function that takes an image array and returns a transformed one:

```python
import numpy as np

def add_random_noise(image):
    noise = np.random.normal(0, 0.1, image.shape)
    return image + noise

datagen = ImageDataGenerator(preprocessing_function=add_random_noise)
```

This adds Gaussian noise (mean=0, std=0.1) to every image — useful for making models more robust to noisy input.

### 3.4 Generating & Visualizing Augmented Images

```python
x = img_to_array(load_img('sample.jpg'))
x = np.expand_dims(x, axis=0)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break
plt.show()
```

---

## 4. Comparing Optimizers

When fine-tuning, the choice of optimizer affects convergence speed and final accuracy:

| Optimizer | Characteristics |
|-----------|----------------|
| **Adam** | Adaptive learning rate; fast convergence; default choice |
| **SGD** | Simple gradient descent; slower but sometimes better generalization |
| **RMSprop** | Adaptive; good for non-stationary problems and RNNs |

```python
# Reset and test with SGD
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Reset and test with RMSprop
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 5. Transpose Convolution (Conv2DTranspose)

### What is Transpose Convolution?

Transpose convolution (also called **deconvolution**) is the reverse of a regular convolution — instead of downsampling, it **upsamples** the feature map back to a larger spatial size. It is commonly used in:

- **Image reconstruction** (autoencoders)
- **Image segmentation** (U-Net)
- **Generative models** (GANs)

### Conv2D vs Conv2DTranspose

| | `Conv2D` | `Conv2DTranspose` |
|--|----------|------------------|
| Purpose | Feature extraction (encode) | Image reconstruction (decode) |
| Spatial effect | Reduces or keeps size | Increases or keeps size |
| Use case | Encoder / feature extractor | Decoder / upsampler |

### Model Architecture — Image Reconstruction

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose

input_layer = Input(shape=(28, 28, 1))

# Encoder: extract features
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

# Decoder: reconstruct image
output_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')
```

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Input | (28, 28, 1) | Grayscale image |
| Conv2D | (28, 28, 32) | 32 filters, 3×3, ReLU, `padding='same'` |
| Conv2DTranspose | (28, 28, 1) | 1 filter, 3×3, Sigmoid, `padding='same'` |

- `padding='same'` preserves spatial dimensions through both layers
- Output shape matches input → suitable for **reconstruction tasks**
- Loss: **MSE** (measures pixel-level difference between input and output)

### Training for Image Reconstruction

```python
# For reconstruction: target = input
X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Visualizing Results

```python
y_pred = model.predict(X_test)

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")

    plt.subplot(2, 10, i + 11)
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
plt.show()
```

### Variations & Experiments

| Experiment | Change | Effect |
|-----------|--------|--------|
| Larger kernel (5×5) | `kernel_size=(5, 5)` | Captures wider spatial context |
| Add Dropout(0.5) | After `Conv2D` | Reduces overfitting |
| Change activation to `tanh` | Both layers | Output range (−1, 1) instead of (0, 1) |

---

## 6. Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Transfer Learning** | Reuse pretrained model weights for a new task |
| **Feature Extraction** | Freeze all base layers; only train custom top layers |
| **Fine-tuning** | Unfreeze top layers of base model and retrain at a low learning rate |
| **Catastrophic Forgetting** | Risk of overwriting pretrained knowledge with a high learning rate |
| **ImageDataGenerator** | Real-time augmentation and preprocessing of image batches |
| **`include_top=False`** | Loads VGG16 without its original classification head |
| **`featurewise_center`** | Dataset-level normalization: subtracts mean per feature |
| **`samplewise_center`** | Image-level normalization: subtracts mean per image |
| **`preprocessing_function`** | Hook for custom augmentation logic in `ImageDataGenerator` |
| **`validation_split`** | Reserves a fraction of training data for validation |
| **Domain adaptation** | Techniques to align source and target datasets when domains differ |
| **Transpose Convolution** | Upsamples feature maps; used in decoders, segmentation, and GANs |
| **`Conv2DTranspose`** | Keras layer for learnable upsampling via transposed convolution |
| **`padding='same'`** | Preserves spatial dimensions through conv layers |
| **Image Reconstruction** | Task where input = target; model learns to reproduce its input |