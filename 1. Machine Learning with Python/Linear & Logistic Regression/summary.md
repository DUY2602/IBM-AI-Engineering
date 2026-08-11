# Regression Models Summary

## Overview of Regression

**Regression** models the relationship between a continuous target variable and one or more explanatory features. It includes both **simple** and **multiple** regression types.

| Type                    | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| **Simple Regression**   | Uses a single independent variable to estimate a dependent variable |
| **Multiple Regression** | Uses more than one independent variable                             |

### Common Applications

- Forecasting sales
- Estimating maintenance costs
- Predicting rainfall
- Modeling disease spread

---

## Simple Linear Regression

In **simple linear regression**, a best-fit line is found that minimizes prediction errors. The most common error metric is **Mean Squared Error (MSE)**.

This method is known as **Ordinary Least Squares (OLS)**.

### Key Characteristics of OLS

- Easy to interpret
- Sensitive to outliers (which can significantly impact accuracy)

---

## Multiple Linear Regression

**Multiple linear regression** extends simple linear regression by using multiple independent variables to:

- Predict outcomes more accurately
- Analyze relationships between variables

### Important Consideration

Adding too many variables can lead to **overfitting**. Careful feature selection is essential to build a balanced and generalizable model.

---

## Nonlinear Regression

When the relationship between variables does not follow a straight line, **nonlinear regression** is used. Common functional forms include:

- Polynomial
- Exponential
- Logarithmic

### Polynomial Regression Note

While polynomial regression can fit complex patterns well, it risks **overfitting** by capturing random noise instead of the true underlying relationship.

---

## Logistic Regression

**Logistic regression** is used for:

- Predicting probabilities
- Binary classification tasks

It is particularly suitable when the target variable is binary and when assessing the impact of individual features is important.

### Optimization

- Minimizes errors using **log-loss** (also known as binary cross-entropy)
- Commonly optimized with **Gradient Descent** or **Stochastic Gradient Descent**

### Gradient Descent

Gradient descent is an iterative optimization algorithm that minimizes the cost function. It is a core technique for training logistic regression models efficiently.

---

## Quick Comparison

| Model                      | Target Type          | Key Metric / Method         | Main Strength                                | Main Limitation                                 |
| -------------------------- | -------------------- | --------------------------- | -------------------------------------------- | ----------------------------------------------- |
| Simple Linear Regression   | Continuous           | MSE / OLS                   | Easy to interpret                            | Sensitive to outliers                           |
| Multiple Linear Regression | Continuous           | MSE / OLS                   | Captures multiple relationships              | Risk of overfitting                             |
| Nonlinear / Polynomial     | Continuous           | MSE                         | Models curved relationships                  | High risk of overfitting                        |
| Logistic Regression        | Binary / Probability | Log-loss / Gradient Descent | Interpretable probabilities & classification | Assumes linear decision boundary in logit space |
