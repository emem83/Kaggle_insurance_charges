import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# load the dataset
df = pd.read_csv('insurance.csv')

# TITLE OF THE APP
st.title('Insurance Charges Prediction')

# Data Overview
st.header('Data Overview for First 10 Rows')
st.write(df.head(10))

# one-hot encode data
categorical_features = ['sex', 'smoker', 'region'] # define categorical features
df2 = pd.get_dummies(df, columns = categorical_features) # one-hot encode categorical features

# split the data into input and output
X = df2.drop('charges', axis = 1) # input features
y = df2['charges'] # target variable

# split the data between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
rf_best = RandomForestRegressor(max_depth = 5, min_samples_leaf = 4, n_estimators = 200, random_state = 42)

# train the selected model
rf_best.fit(X_train, y_train)

# make predictions
y_pred_rf_best = rf_best.predict(X_test)

# Model evaluation
mae_rf_best = mean_absolute_error(y_test, y_pred_rf_best)
mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)
r2_rf_best = r2_score(y_test, y_pred_rf_best)

# Display the results
st.write(f'MAE Score : {mae_rf_best}')
st.write(f'MSE Score: {mse_rf_best}')
st.write(f'R2 Score: {r2_rf_best}')

# prompt for user input
st.write('Enter the input values for prediction:')

user_input = {}

numerical_float = ['bmi']
numerical_integer = ['age', 'children']
categorical_features = ['sex', 'smoker', 'region']

for column in numerical_float + numerical_integer + categorical_features:
    if column in numerical_float:
        user_input[column] = st.number_input(column,
            min_value=np.min(X[column]),
            max_value=np.max(X[column]),
            value=float(np.min(X[column]))  # Convert to float
        )
    elif column in numerical_integer:
        user_input[column] = st.number_input(
            column,
            min_value=int(np.min(X[column])),  # Convert to int
            max_value=int(np.max(X[column])),  # Convert to int
            value=int(np.min(X[column]))  # Convert to int
        )
    elif column in categorical_features:
        user_input[column] = st.selectbox(column, df[column].unique())

# convert user input to dataframe
user_input_df = pd.DataFrame(user_input, index=[0])

# one-hot encode the user input dataframe
user_input_encoded = pd.get_dummies(user_input_df, columns = categorical_features)

# Get missing columns in the user input
missing_cols = set(X.columns) - set(user_input_encoded.columns)
# Add a missing column in user input with default value equal to 0
for c in missing_cols:
    user_input_encoded[c] = 0
# Ensure the order of column in the test set
user_input_encoded = user_input_encoded[X.columns]

# standardize the user input dataframe the same way we standardized our test dataframe
user_input_sc = scaler.transform(user_input_encoded)

# predict the price
predicted_charge = rf_best.predict(user_input_sc)

# display the predicted price
st.write(f'Predicted Insurance Charge for the given inputs of the person is: {predicted_charge[0]}')