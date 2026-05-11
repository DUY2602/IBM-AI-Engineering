# Final Project: Aircraft Damage Classification & Captioning

---

## Project Overview

Automate the detection and description of aircraft damage using two deep learning approaches:

| Part             | Task                                              | Model Used                             |
| ---------------- | ------------------------------------------------- | -------------------------------------- |
| **Part 1** | Classify damage as `dent`or `crack`           | VGG16 (pretrained, feature extraction) |
| **Part 2** | Generate captions and summaries for damage images | BLIP (pretrained Transformer)          |

**Dataset:** Aircraft damage images organized into `train/`, `valid/`, `test/` splits, each with `dent/` and `crack/` subfolders.

---

## Part 1 — Image Classification with VGG16

### 1.1 Configuration

```python
batch_size = 32
n_epochs   = 5
img_rows, img_cols = 224, 224   # VGG16 expected input size
input_shape = (224, 224, 3)     # RGB color images
```

### 1.2 Data Preprocessing

`ImageDataGenerator` is used to load and preprocess images in real time without loading the entire dataset into memory.

```python
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)
```

Each generator uses `flow_from_directory()` to read images directly from folders:

| Generator           | Directory  | Shuffle | class_mode |
| ------------------- | ---------- | ------- | ---------- |
| `train_generator` | `train/` | True    | `binary` |
| `valid_generator` | `valid/` | False   | `binary` |
| `test_generator`  | `test/`  | False   | `binary` |

### 1.3 Model Definition — Transfer Learning with VGG16

**Pretrained models** like VGG16 have already learned useful features from millions of images (ImageNet). Instead of training from scratch, we use VGG16 as a **feature extractor** and add a custom classifier on top.

```python
# Load VGG16 without the top classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Flatten the output of VGG16
output = Flatten()(base_model.layers[-1].output)
base_model = Model(base_model.input, output)

# Freeze all VGG16 layers — weights will not be updated during training
for layer in base_model.layers:
    layer.trainable = False
```

**Custom classifier stacked on top:**

```python
model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))   # Binary output: dent vs crack
```

| Layer                 | Details                                           |
| --------------------- | ------------------------------------------------- |
| VGG16 base            | Feature extractor, frozen, pretrained on ImageNet |
| Dense(512, ReLU) × 2 | Custom classifier layers                          |
| Dropout(0.3) × 2     | Regularization to reduce overfitting              |
| Dense(1, Sigmoid)     | Binary classification output                      |

### 1.4 Compilation & Training

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

history = model.fit(
    x=train_generator,
    epochs=n_epochs,
    validation_data=valid_generator
)
```

* Loss: **Binary Crossentropy** (binary classification)
* Optimizer: **Adam** with low learning rate `0.0001` (fine-tuning on top of frozen base)

### 1.5 Evaluation & Visualization

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)

# Plot accuracy curves
plt.plot(train_history['accuracy'], label='Training Accuracy')
plt.plot(train_history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
```

Predictions are made by thresholding sigmoid output at 0.5:

```python
predicted_classes = (predictions > 0.5).astype(int)
# 0 → crack, 1 → dent  (based on class_indices from generator)
```

---

## Part 2 — Image Captioning with BLIP

### What is BLIP?

**BLIP (Bootstrapping Language-Image Pretraining)** is a pretrained vision-and-language model that generates natural language descriptions for images. It is trained on image-text pairs and is ideal for:

* Image captioning
* Image summarization
* Visual question answering

Loaded from Hugging Face:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### Custom Keras Layer — `BlipCaptionSummaryLayer`

A custom `tf.keras.layers.Layer` wraps the BLIP model to integrate it into the Keras/TensorFlow pipeline.

| Method            | Role                                                                              |
| ----------------- | --------------------------------------------------------------------------------- |
| `__init__`      | Stores the BLIP processor and model                                               |
| `call`          | Passes inputs through `tf.py_function`to run Python-native code                 |
| `process_image` | Loads image via PIL, sets task-specific prompt, runs BLIP, returns generated text |

**Task-specific prompts:**

| Task          | Prompt                                 |
| ------------- | -------------------------------------- |
| `"caption"` | `"This is a picture of"`             |
| `"summary"` | `"This is a detailed photo showing"` |

### Helper Function

```python
def generate_text(image_path, task):
    blip_layer = BlipCaptionSummaryLayer(processor=processor, model=model)
    return blip_layer(image_path, task)
```

### Usage

```python
image_path = tf.constant("path/to/image.jpg")

# Generate caption
caption = generate_text(image_path, tf.constant("caption"))
print("Caption:", caption.numpy().decode("utf-8"))

# Generate summary
summary = generate_text(image_path, tf.constant("summary"))
print("Summary:", summary.numpy().decode("utf-8"))
```

---

## Key Concepts Summary

| Concept                       | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| **Transfer Learning**   | Reuse a model pretrained on a large dataset (ImageNet) for a new task |
| **Feature Extraction**  | Freeze base model layers; only train the custom top layers            |
| **ImageDataGenerator**  | Real-time image loading, rescaling, and batching from directories     |
| **Binary Crossentropy** | Loss function for 2-class classification                              |
| **Sigmoid output**      | Outputs probability [0, 1]; threshold at 0.5 to get class             |
| **Dropout**             | Randomly deactivates neurons during training to reduce overfitting    |
| **BLIP**                | Pretrained vision-language model for captioning and summarization     |
| **Custom Keras Layer**  | Extends `tf.keras.layers.Layer`to wrap non-Keras models (e.g. BLIP) |
| **`tf.py_function`**  | Allows running arbitrary Python code inside a TensorFlow graph        |
