import streamlit as st
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV

# load the dataset
dataset_dir = kagglehub.dataset_download("mirichoi0218/insurance")
file_name = 'insurance.csv'
path = os.path.join(dataset_dir, file_name)
df = pd.read_csv(path)
df = pd.DataFrame(df.data,columns = df.feature_names)
# add the target column
df['charges'] = df.target
