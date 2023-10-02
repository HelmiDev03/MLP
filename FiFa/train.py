import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def clean_currency(x):
    if x.find("K") > -1 and x.find(".") == -1:
        return x.replace("K", "000").replace("€", "")
    elif x.find("K") > -1 and x.find(".") > -1:
        return x.replace("K", "00").replace(".", "").replace("€", "")
    elif x.find("M") > -1 and x.find(".") == -1:
        return x.replace("M", "000000").replace("€", "")
    elif x.find("M") > -1 and x.find(".") > -1:
        return x.replace("M", "00000").replace(".", "").replace("€", "")
    else:
        return x.replace("€", "")

# Data collection and data processing
features = pd.read_csv(r"C:\Users\helmi\OneDrive\Desktop\rockvsmine\FIFA.csv - FIFA.csv.csv")

# Removing unnecessary columns
features.drop(columns=["Loaned From"], inplace=True)

# For each column with euro price, remove the euro sign and convert the value to float
columns_in_euros = ["Value", "Wage", "Release Clause"]
for column in columns_in_euros:
    features[column] = features[column].astype("str")
    features[column] = features[column].apply(clean_currency).astype("float64")

print(features.head())

# Check missing values
missing_lst = []
for col in features.columns:
    column_name = col
    missing_values = int(features[col].isnull().sum())
    missing_lst.append([column_name, missing_values])
missing_df = pd.DataFrame(missing_lst, columns=["Column name", "Missing values"])

print(missing_df)

# Get column names with missing values
missing_columns = missing_df["Column name"].tolist()

# Remove rows with missing values
features = features.dropna(subset=missing_columns)

print(features.head())

# Convert string columns to ASCII integers
def string_to_ascii_int(s):
    return int(''.join(str(ord(char)) for char in str(s)))

for col in features.columns:
    if features[col].dtype == "object" and features[col].apply(lambda x: isinstance(x, str)).all():
        features[col] = features[col].apply(string_to_ascii_int)

print(features.head())

# Split the data
Y = features["Release Clause"]
X = features.drop(columns=["Release Clause"], inplace=False)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

# Instantiate and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print(model)

# Evaluation
from sklearn.metrics import r2_score

X_train_prediction = model.predict(X_train)
r2_train = r2_score(y_train, X_train_prediction)
print(f"R-squared for the training data: {r2_train * 100:.2f}%")

Y_test_prediction = model.predict(X_test)
r2_test = r2_score(y_test, Y_test_prediction)
print(f"R-squared for the test data: {r2_test * 100:.2f}%")

# Making a predictive system
# Sample player data


# Convert the player data to a numpy array and reshape it
test_data = X_scaled[1].reshape(1, -1)

# Make predictions for the test data
predicted_value = model.predict(test_data)

print("Predicted Value:", predicted_value)