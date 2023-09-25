# Diabetes Prediction using SVM

This project aims to predict whether a person has diabetes or not, based on certain diagnostic measurements included in the dataset. The dataset used in this project is the diabetes dataset, and the model used for prediction is the Support Vector Machine (SVM).

## Dependencies

- numpy
- pandas
- scikit-learn

## Dataset

The dataset used in this project is the diabetes dataset. It contains several diagnostic measurements and a target variable `Outcome` which indicates whether a person has diabetes or not , https://www.dropbox.com/s/uh7o7uyeghqkhoy/diabetes.csv?dl=0

## Steps

1. **Data Collection and Analysis**: The diabetes dataset is loaded into a pandas DataFrame. Basic statistical analysis is performed on the dataset.

2. **Data Preprocessing**: The target variable `Outcome` is separated from the other features. The features are then standardized using `StandardScaler` from scikit-learn to bring all the features to the same level of magnitude.

3. **Train-Test Split**: The dataset is split into training and testing sets using `train_test_split` from scikit-learn. The split is stratified based on the target variable to ensure a correct distribution of data.

4. **Model Training**: The SVM classifier with a linear kernel is trained on the training data.

5. **Model Evaluation**: The accuracy of the model is evaluated on both the training and testing data.

6. **Making Predictions**: A predictive system is built where new data can be input to predict whether a person has diabetes or not.

## Usage

To use this code, ensure you have the required dependencies installed. Load your diabetes dataset in the specified path and run the Python script.

## Results

The accuracy of the model on the training and testing data will be printed. Additionally, for the provided sample input data, the prediction (diabetic or not) will be displayed.
