# Classification Models & Tree-Based Methods Summary

## Classification Overview

**Classification** is a supervised machine learning method used to predict discrete labels on new data.

### Common Applications

- Customer churn prediction
- Customer segmentation
- Loan default prediction
- Multiclass drug prescription

---

## Binary vs Multiclass Classification

| Type                          | Description                                                    |
| ----------------------------- | -------------------------------------------------------------- |
| **Binary Classification**     | Predicts one of two possible classes (e.g., fraud / not fraud) |
| **Multiclass Classification** | Predicts one of three or more classes                          |

### Strategies to Extend Binary Classifiers to Multiclass

| Strategy             | Approach                                                   |
| -------------------- | ---------------------------------------------------------- |
| **One-vs-All (OvA)** | Train one binary classifier per class (class vs. the rest) |
| **One-vs-One (OvO)** | Train a binary classifier for every pair of classes        |

---

## Decision Trees

A **decision tree** classifies data by:

1. Testing a feature at each internal node
2. Branching based on the test result
3. Assigning a class label at the leaf nodes

### Training Process

- Select the feature that **best splits** the data at each node
- **Prune** the tree to reduce overfitting

### Split Quality Metrics

| Metric                       | Used For       | Description                                |
| ---------------------------- | -------------- | ------------------------------------------ |
| **Information Gain**         | Classification | Reduction in entropy after a split         |
| **Gini Impurity**            | Classification | Probability of incorrect classification    |
| **Mean Squared Error (MSE)** | Regression     | Measures how well a split reduces variance |

---

## Regression Trees

**Regression trees** are similar to decision trees but predict **continuous values**.

- Recursively split the data to maximize information gain (or minimize MSE)
- Leaf nodes contain a predicted numeric value (usually the mean of the samples in that leaf)

---

## K-Nearest Neighbors (k-NN)

**k-NN** is a supervised, instance-based algorithm used for both **classification** and **regression**.

### How it Works

- For a new data point, find the _k_ closest labeled training points
- Assign the majority class (classification) or average value (regression)

### Key Considerations

- **Normalize / standardize** features (distance-based method)
- Test multiple values of _k_ and choose the one with best validation accuracy
- Sensitive to class distribution and irrelevant features

---

## Support Vector Machines (SVM)

**SVM** finds a hyperplane that **maximizes the margin** between two classes.

### Strengths

- Effective in high-dimensional spaces
- Works well with clear class separation

### Limitations

- Sensitive to noise and outliers
- Can be computationally expensive on large datasets
- Requires careful feature scaling

---

## Bias-Variance Tradeoff & Ensemble Methods

| Concept      | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| **Bias**     | Error from overly simplistic assumptions (underfitting)                     |
| **Variance** | Error from sensitivity to small fluctuations in training data (overfitting) |

### Techniques to Manage Bias & Variance

| Method             | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| **Bagging**        | Train multiple models on different bootstrap samples and average results   |
| **Boosting**       | Sequentially train models, each focusing on previous errors                |
| **Random Forests** | Bagging applied to decision trees + random feature selection at each split |

### Random Forests

- Builds many decision trees on bootstrapped datasets
- At each split, considers only a random subset of features
- Final prediction = majority vote (classification) or average (regression)
- Reduces variance while keeping bias low → improves overall accuracy and robustness

---

## Quick Comparison of Key Algorithms

| Algorithm               | Type              | Key Strength                        | Key Weakness                          | Typical Use Case                          |
| ----------------------- | ----------------- | ----------------------------------- | ------------------------------------- | ----------------------------------------- |
| **Logistic Regression** | Linear classifier | Interpretable, probabilistic output | Assumes linear decision boundary      | Binary / multiclass classification        |
| **Decision Tree**       | Non-linear        | Highly interpretable                | Prone to overfitting                  | Rule-based decisions                      |
| **k-NN**                | Instance-based    | Simple, no training phase           | Slow inference, sensitive to features | Small-to-medium datasets                  |
| **SVM**                 | Margin-based      | Strong in high dimensions           | Hard to scale, sensitive to noise     | Text classification problems              |
| **Random Forest**       | Ensemble of trees | Robust, handles non-linearity well  | Less interpretable than single tree   | General-purpose classification/regression |
| **XGBoost**             | Gradient boosting | High accuracy, fast                 | More hyperparameters to tune          | High-performance tabular modeling         |

---

## Practical Tips

- Always **scale features** before using distance-based or margin-based models (k-NN, SVM).
- Use **cross-validation** to select hyperparameters (_k_, tree depth, regularization, etc.).
- Watch for **class imbalance** — accuracy alone can be misleading; prefer precision, recall, F1, or ROC-AUC.
- Prefer **ensemble methods** (Random Forest, XGBoost) when interpretability is less critical than predictive performance.
