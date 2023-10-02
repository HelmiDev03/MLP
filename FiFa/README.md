## Overview

### Data Cleaning
The script begins by loading the FIFA dataset and removing unnecessary columns. It then cleans the Euro currency values in columns like "Value," "Wage," and "Release Clause," converting them to floating-point numbers.

### Missing Data Handling
Next, it checks for missing values and removes rows with missing data. This ensures that the dataset is clean and ready for modeling.

### Feature Encoding
The script converts string columns to ASCII integers to make them suitable for machine learning.

### Data Scaling
Before training the linear regression model, the script standardizes the data using `StandardScaler` to ensure that all features have the same scale.

### Model Training
The linear regression model is instantiated and trained on the preprocessed data. The script prints the R-squared scores for both the training and test datasets to evaluate the model's performance.

### Making Predictions
You can use the model to make predictions by providing a test data point. The script includes an example of making a prediction using a sample data point.

### Dependencies
- pandas
- scikit-learn
- numpy

### Dataset
https://docs.google.com/spreadsheets/d/1ANv8Di9IgGmpaLHuxKzWAbbLE4ifTTSeHEKfLB8nAcU/edit#gid=1655193775

### Kaggle Point
https://www.kaggle.com/code/kauvinlucas/working-with-fifa-19-dataset

