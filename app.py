import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# load the dataset
df = pd.read_csv('insurance.csv')

# TITLE OF THE APP
st.title('Insurance Charge Prediction')

# Data Overview
st.header('Data Overview for First 10 Rows')
st.write(df.head(10))

# one-hot encode data
categorical_features = ['sex', 'smoker', 'region'] # define categorical features
df = pd.get_dummies(df, columns = categorical_features) # one-hot encode categorical features

# split the data into input and output
X = df.drop('charges', axis=1) # input features
y = df['charges'] # target variable

# split the data between train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
rf_best = RandomForestRegressor(max_depth = 5, min_samples_leaf = 4, n_estimators=200, random_state = 42)

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

#user_input = {}
#for column in X.columns: # get columns from X
#    user_input[column] = st.number_input(column, min_value = np.min(X[column]), max_value = np.max(X[column]), value = np.min(X[column]))

user_input = []

st.title("Input for Prediction")  # Add a title

for column in X.columns:
    if column in ['bmi']:  # float numerical feature
        X[column] = pd.to_numeric(X[column], errors='coerce')
        min_val = float(np.nanmin(X[column]))
        max_val = float(np.nanmax(X[column]))
        user_input[column] = st.number_input(
            f"Enter a value for {column} (between {min_val:.0f} and {max_val:.0f}):",
            min_value = min_val,
            max_value = max_val,
            value = min_val,  # Or another appropriate default value
        )
    elif column in ['age', 'children']:  # integer numerical features
        X[column] = pd.to_numeric(X[column], errors='coerce')
        min_val = int(np.nanmin(X[column]))
        max_val = int(np.nanmax(X[column]))
        user_input[column] = st.number_input(
            f"Enter a value for {column} (between {min_val:.2f} and {max_val:.2f}):",
            min_value = min_val,
            max_value = max_val,
            value = min_val,  # Or another appropriate default value
        )        
    elif column in ['sex', 'smoker', 'region']:  # Categorical features
        user_input[column] = st.selectbox(
            f"Enter a value for {column}:",
            options=X[column].unique(),
        )
    else:
        # Handle other columns if needed
        pass

if st.button("Predict"):
    # Create a DataFrame from the user input
    user_input = pd.DataFrame([user_input])

# standadrize the user input dataframe the same way we standardized our test dataframe
user_input_sc_df = scaler.transform(user_input)

# predict the price
predicted_charge = rf_best.predict(user_input_sc_df)

# display the predicted price
st.write(f'Predicted Insurance Charge for the given inputs of the person is: {predicted_charge}')
