# Data preprocessing

# Library used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Database
df = pd.read_csv('data\\raw\\cosmetic_emulsion_data.csv')

# X
x = df.iloc[:, 0:7].to_numpy()

# Y
y = df.iloc[:, 7].to_numpy()

# Shape
print(f'X Shape: {x.shape} | Y Shape: {y.shape}')

# Encoding
# Label encoder
encoder = LabelEncoder()
x[:,6] = encoder.fit_transform(x[:,6])
# OneHot encoder
onehotencoder = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(sparse_output=False), [6])], 
    remainder='passthrough')
x = onehotencoder.fit_transform(x)
print(f'New X shape: {x.shape}')

# Feature scaling
scaler = StandardScaler()
x = scaler.fit(x)