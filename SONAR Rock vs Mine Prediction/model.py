import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # this is for splitting data into train and test
from sklearn.linear_model import LinearRegression,LogisticRegression # this is for linear regression
'''
LinearRegression: Ordinary least squares linear regression.
LogisticRegression: Logistic regression for binary classification tasks.
Ridge: Linear regression with L2 regularization.
Lasso: Linear regression with L1 regularization.
ElasticNet: Linear regression with both L1 and L2 regularization.
SGDRegressor: Linear regression using stochastic gradient descent.
SGDClassifier: Linear classifiers (SVM, logistic regression, etc.) with SGD training.
Perceptron: Simple linear classifier.
RidgeClassifier: Classifier using Ridge regression.
PassiveAggressiveClassifier: Linear classifier with passive-aggressive updates.
PassiveAggressiveRegressor: Regression with passive-aggressive updates.
RANSACRegressor: RANdom SAmple Consensus (RANSAC) algorithm for robust linear regression.
TheilSenRegressor: Theil-Sen estimator for robust linear regression.
HuberRegressor: Linear regression model that is robust to outliers.
 and many more.
'''
from sklearn.metrics import mean_squared_error,accuracy_score # this is for checking the error

# data collection and data processing
sonar_data = pd.read_csv(r"C:\Users\helmi\OneDrive\Desktop\rockvsmine\Copy of sonar data.csv",header=None)#loading dataset to a pandas dataframe

print(sonar_data.head())#printing first 5 rows of the dataset

print(sonar_data.shape)         #checking for null values in the dataset
print("---------------------------------------")
print(sonar_data.describe())#printing some statistical values of the dataset


print(sonar_data[60].value_counts())#printing the number of rocks and mines in the dataset

print("---------------------------------------")
print(sonar_data.groupby(60).mean())#printing the mean values of the dataset

#seperting data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)

#Training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)
#Model Training --> Logistic Regression 
model = LogisticRegression() #logistic regression  is used for binary classification problems
model.fit(X_train, Y_train)
print(model)



#Model Evaluation


#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)


#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)


"""Making a Predictive System"""
input_data  =(0.0283,0.0599,0.0656,0.0229,0.0839,0.1673,0.1154,0.1098,0.1370,0.1767,0.1995,0.2869,0.3275,0.3769,0.4169,0.5036,0.6180,0.8025,0.9333,0.9399,0.9275,0.9450,0.8328,0.7773,0.7007,0.6154,0.5810,0.4454,0.3707,0.2891,0.2185,0.1711,0.3578,0.3947,0.2867,0.2401,0.3619,0.3314,0.3763,0.4767,0.4059,0.3661,0.2320,0.1450,0.1017,0.1111,0.0655,0.0271,0.0244,0.0179,0.0109,0.0147,0.0170,0.0158,0.0046,0.0073,0.0054,0.0033,0.0045,0.0079)
#changing input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
if (prediction[0]=='R'):
  print("The object is a Rock")
else:
    print("The object is a Mine")
