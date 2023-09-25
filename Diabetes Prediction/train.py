import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.model_selection import train_test_split
from sklearn import svm # for using the svm model
from sklearn.metrics import accuracy_score # for checking the accuracy of the model

# data collection and analysis

# loading the diabetes dataset to a pandas DataFrame 
diabetes_dataset = pd.read_csv(r"C:\Users\helmi\OneDrive\Desktop\rockvsmine\venv\Diabetes Prediction\diabetes.csv")
print(diabetes_dataset)
print(diabetes_dataset.describe())


X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X,Y)

#data standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

#data standardization is used to bring all the features to the same level of magnitude, which means the feature with the largest magnitude will not dominate the result
X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#stratify is used for correct distribution of data as of the original data
#random_state is used for reproducing the same result
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector machine classifier
classifier.fit(X_train, Y_train)
print(classifier)

#Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

#Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)

#making a predictive system
input_data = (0,109,88,30,0,32.5,0.855,38)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardize the input data
scaler.fit(input_data_reshaped)
std_data = scaler.transform(input_data_reshaped)
classifed_data= classifier.predict(std_data)
if (classifed_data == 0):
  print("The person is not diabetic")
else:
    print("The person is diabetic")


