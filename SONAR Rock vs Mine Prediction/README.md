# SONAR Rock vs. Mine Prediction

This repository contains a Python script that predicts whether an object is a rock or a mine based on the SONAR dataset.

## Dependencies

- numpy
- pandas
- scikit-learn

## Dataset
https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view

## Key Imports

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
```

### Explanation of sklearn.linear_model imports:


- **LogisticRegression**: Logistic regression for binary classification tasks.


## Data Collection and Processing

The dataset is loaded into a pandas dataframe, and some basic exploratory data analysis is performed to understand the data.

## Model Training

The Logistic Regression model from scikit-learn is used for training. Logistic regression is suitable for binary classification problems like this one.

## Model Evaluation

The model's accuracy is evaluated on both the training and test data.

## Predictive System

A sample input is provided to the trained model to predict whether the object is a rock or a mine.

---

