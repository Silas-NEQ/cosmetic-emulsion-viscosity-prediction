# Data preprocessing

# Library used
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib as jb
import pickle

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
labelencoder = LabelEncoder()
x[:,6] = labelencoder.fit_transform(x[:,6])
jb.dump(labelencoder, 'data\\processed\\labelencoder.pkl')
# OneHot encoder
onehotencoder = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(sparse_output=False), [6])], 
    remainder='passthrough')
x = onehotencoder.fit_transform(x)
print(f'New X shape: {x.shape}')
jb.dump(onehotencoder, 'data\\processed\\onehotencoder.pkl')

# Feature scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)
jb.dump(scaler, 'data\\processed\\scaler.pkl')

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
print(f'Shape X Train: {x_train.shape} | Shape X Test {x_test.shape}')
print(f'Shape Y Train: {y_train.shape} | Shape Y Test {y_test.shape}')

# Save processed data
with open('data\\processed\\processed_data.pkl', mode='wb') as f:
    pickle.dump([x_test, x_train, y_test, y_train], f)