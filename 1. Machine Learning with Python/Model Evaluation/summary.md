# Model Evaluation, Validation & Regularization Summary

## Supervised Learning Evaluation

**Supervised learning evaluation** assesses a model’s ability to predict outcomes for **unseen data**. The most common approach is a **train/test split** to estimate real-world performance.

---

## Classification Evaluation Metrics

| Metric               | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| **Accuracy**         | Overall proportion of correct predictions                                     |
| **Confusion Matrix** | Breakdown of true positives, true negatives, false positives, false negatives |
| **Precision**        | Of the predicted positives, how many were actually positive                   |
| **Recall**           | Of the actual positives, how many were correctly identified                   |
| **F1 Score**         | Harmonic mean of precision and recall (balances both metrics)                 |

> **Note:** In high-stakes domains (e.g., medical diagnosis), **false negatives** are often more costly than false positives.

---

## Regression Evaluation Metrics

| Metric                             | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| **MAE** (Mean Absolute Error)      | Average absolute difference between predictions and actuals  |
| **MSE** (Mean Squared Error)       | Average of squared errors (penalizes large errors more)      |
| **RMSE** (Root Mean Squared Error) | Square root of MSE (in the same units as the target)         |
| **R²** (R-squared)                 | Proportion of variance explained by the model                |
| **Explained Variance**             | Similar to R²; measures how well the model captures variance |

---

## Unsupervised Learning Evaluation

Unsupervised models (e.g., clustering) are evaluated for **pattern quality** and **consistency**:

| Metric                   | Purpose                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| **Silhouette Score**     | Measures how similar a point is to its own cluster vs. others (range: -1 to 1) |
| **Davies-Bouldin Index** | Lower values indicate better separation between clusters                       |
| **Adjusted Rand Index**  | Compares clustering results against known ground-truth labels                  |

---

## Dimensionality Reduction Evaluation

When reducing dimensions (e.g., PCA), assess how well the reduced data retains structure:

- **Explained Variance Ratio** — How much variance is retained by the selected components
- **Reconstruction Error** — Difference between original and reconstructed data
- **Neighborhood Preservation** — Whether local relationships between points are maintained

---

## Model Validation & Preventing Overfitting

### Data Splits

- **Training set** → Used to fit the model
- **Validation set** → Used for hyperparameter tuning
- **Test set** → Final unbiased performance estimate

### Cross-Validation

| Method                      | Description                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| **K-Fold Cross-Validation** | Data split into _k_ folds; model trained/tested _k_ times                |
| **Stratified K-Fold**       | Preserves class distribution in each fold (important for classification) |

Cross-validation provides a more robust estimate of model performance and helps avoid overfitting to a single test set.

---

## Regularization Techniques

Regularization adds a **penalty term** to the loss function to discourage overly complex models and reduce overfitting.

| Technique      | Penalty Type          | Effect                                                              | Use Case                             |
| -------------- | --------------------- | ------------------------------------------------------------------- | ------------------------------------ |
| **Ridge (L2)** | Squared coefficients  | Shrinks coefficients toward zero (but rarely to exact zero)         | Multicollinearity, many features     |
| **Lasso (L1)** | Absolute coefficients | Can drive some coefficients exactly to zero → **feature selection** | High-dimensional data, sparse models |

> **Lasso** is particularly useful for automatic feature selection before building a final model.

---

## Data Leakage

**Data leakage** occurs when the training data contains information that would not be available at prediction time in the real world.

### Prevention Strategies

- Strictly separate training, validation, and test data
- Fit scalers, encoders, and other transformers **only on the training set**
- Be careful with feature engineering that uses future or target-related information

---

## Common Modelling Pitfalls

| Pitfall                            | Description / Risk                                                      |
| ---------------------------------- | ----------------------------------------------------------------------- |
| Misinterpreting feature importance | Importance ≠ causation; correlated features can share importance        |
| Ignoring class imbalance           | Accuracy can be misleading; use precision/recall/F1 or resampling       |
| Making causal inferences           | Correlation does not imply causation without proper experimental design |
| Overfitting to the test set        | Repeatedly tuning on the same test set leads to optimistic results      |

### Feature Importance Best Practices

- Consider **redundancy** (correlated features)
- Be aware of **scale sensitivity**
- Avoid assuming that high importance implies a causal relationship

---

## Quick Reference: Evaluation by Learning Type

| Learning Type                | Primary Goal                   | Key Metrics / Tools                               |
| ---------------------------- | ------------------------------ | ------------------------------------------------- |
| **Classification**           | Correct class assignment       | Accuracy, Precision, Recall, F1, Confusion Matrix |
| **Regression**               | Accurate continuous prediction | MAE, MSE, RMSE, R²                                |
| **Clustering**               | Meaningful group discovery     | Silhouette Score, Davies-Bouldin Index            |
| **Dimensionality Reduction** | Structure preservation         | Explained Variance, Reconstruction Error          |
