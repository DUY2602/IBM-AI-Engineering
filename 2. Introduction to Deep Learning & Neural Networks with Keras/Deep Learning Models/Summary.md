# Deep Learning — Comprehensive Concept Summary

---

## 1. Shallow vs Deep Neural Networks

| | Shallow Neural Network | Deep Neural Network |
|--|----------------------|-------------------|
| Hidden layers | 1 | Many |
| Neurons per layer | Few | Large number |
| Input type | Vectors only | Raw data: images, text, audio |
| Use cases | Simple classification/regression | Complex vision, NLP, generation tasks |

The boom in deep learning is attributed to three main factors: **advancements in algorithms**, **data availability**, and **greater computational power**.

---

## 2. Convolutional Neural Networks (CNN)

CNNs make the explicit assumption that inputs are images, making them best suited for **image recognition**, **object detection**, and other computer vision tasks.

### Input format
- Grayscale image: `(n × m × 1)`
- Color image: `(n × m × 3)`

### CNN Layer Types

| Layer | Purpose |
|-------|---------|
| **Conv2D** | Applies learnable filters across the image; followed by ReLU to pass only positive values |
| **MaxPooling2D** | Reduces spatial dimensions; keeps the most prominent features |
| **Flatten** | Converts the 3D feature map into a 1D vector for Dense layers |
| **Dense** | Fully connected layer — every node connects to every node in the next layer |

### CNN v1 — Single Conv Block

```
Input (28×28×1) → Conv2D(16, 5×5, ReLU) → MaxPooling2D(2×2)
→ Flatten(2304) → Dense(100, ReLU) → Dense(num_classes, Softmax)
```

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Input | (28, 28, 1) | Grayscale 28×28 |
| Conv2D | (24, 24, 16) | 16 filters, 5×5 kernel, stride 1×1, ReLU |
| MaxPooling2D | (12, 12, 16) | Pool 2×2, stride 2×2 |
| Flatten | (2304,) | 12 × 12 × 16 |
| Dense | (100,) | ReLU |
| Output | (num_classes,) | Softmax |

### CNN v2 — Two Conv Blocks

```
Input (28×28×1) → Conv2D(16, 5×5, ReLU) → MaxPooling2D(2×2)
→ Conv2D(8, 2×2, ReLU) → MaxPooling2D(2×2)
→ Flatten(200) → Dense(100, ReLU) → Dense(num_classes, Softmax)
```

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Input | (28, 28, 1) | Grayscale 28×28 |
| Conv2D | (24, 24, 16) | 16 filters, 5×5 kernel, ReLU |
| MaxPooling2D | (12, 12, 16) | Pool 2×2, stride 2×2 |
| Conv2D | (11, 11, 8) | 8 filters, 2×2 kernel, ReLU |
| MaxPooling2D | (5, 5, 8) | Pool 2×2, stride 2×2 |
| Flatten | (200,) | 5 × 5 × 8 |
| Dense | (100,) | ReLU |
| Output | (num_classes,) | Softmax |

### CNN v1 vs CNN v2

| | CNN v1 | CNN v2 |
|--|--------|--------|
| Conv blocks | 1 | 2 |
| Flatten output | 2304 | 200 |
| Parameters | More | Fewer |
| Feature abstraction | Low-level | Hierarchical |
| Trade-off | More spatial detail | More compact, faster |

---

## 3. Recurrent Neural Networks (RNN)

Standard neural networks treat data points as **independent instances**. RNNs differ by also taking the **output from the previous data point** as input — enabling them to model **sequences and patterns over time**.

- Best for: text, genomes, handwriting, stock market data
- Popular variant: **LSTM (Long Short-Term Memory)**
  - Applications: image generation, handwriting generation, automatic image captioning, automatic video descriptions

---

## 4. Transformers & Self-Attention

Transformers use a **sequence-to-sequence architecture with self-attention** for tasks like machine translation.

### Data Preparation Pipeline

```
Raw sentences
    ↓ Tokenization (words → integers)
    ↓ Padding (all sequences → same length)
    ↓ Decoder input: prepend "startseq", append "endseq"
    ↓ One-hot encode decoder output
```

### Self-Attention Mechanism

Self-attention allows the model to **focus on relevant parts of the input** while processing each word. It uses three components:

| Component | Role |
|-----------|------|
| **Query (Q)** | What this word is looking for |
| **Key (K)** | What this word represents |
| **Value (V)** | The actual information in the word |

**Steps:**
1. Compute Q, K, V by multiplying inputs with learned weight matrices `Wq`, `Wk`, `Wv`
2. Compute attention scores: dot product of Q and K
3. Scale scores by `√dk` to prevent large values
4. Apply **Softmax** → attention weights
5. Multiply attention weights by V → final output

> Output shape is the same as input shape — attention transforms but does not change dimensions.

### Encoder–Decoder Architecture

```
Encoder:
  Input → Embedding(256) → LSTM(256) → context states [state_h, state_c]

Decoder:
  Target → Embedding(256) → LSTM(256, initial_state=encoder_states)
         → SelfAttention([decoder_out, encoder_out, encoder_out])
         → Concatenate → Dense(vocab_size, Softmax)
```

| Component | Role |
|-----------|------|
| **Encoder** | Processes input sentence; produces context vectors via Embedding + LSTM |
| **Self-Attention** | Decoder attends to relevant encoder outputs at each step |
| **Decoder** | Generates target words one at a time using encoder states + attention |
| **Dense (Softmax)** | Predicts the next word in the target vocabulary |

- Loss: **Categorical Crossentropy** (one-hot encoded target words)
- Optimizer: **Adam**, trained for 100 epochs

---

## 5. Autoencoders

Autoencoders are **data compression algorithms** where both compression (encoder) and decompression (decoder) functions are **learned automatically from data**.

- **Data-specific**: trained on one type of data, does not generalize to others
- Flow: `Input image → Encoder → Compressed representation → Decoder → Reconstructed image`

### Applications
- Data denoising
- Dimensionality reduction for visualization

### Restricted Boltzmann Machines (RBM)
A popular type of autoencoder. Applications:
- Fixing imbalanced datasets
- Estimating missing values
- Automatic feature extraction

---

## 6. Summary — Types of Deep Learning Models

| Model | Best For | Key Feature |
|-------|---------|-------------|
| **Dense (MLP)** | Tabular data, regression, classification | Fully connected layers |
| **CNN** | Images, computer vision | Conv + Pooling layers; spatial awareness |
| **RNN / LSTM** | Sequences: text, time series, audio | Uses previous output as next input |
| **Transformer** | Translation, NLP, generation | Self-attention mechanism |
| **Autoencoder** | Compression, denoising, anomaly detection | Encoder–decoder, unsupervised |