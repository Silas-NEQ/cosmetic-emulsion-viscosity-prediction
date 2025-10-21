# Dimensionality Analyse

# Libraries used
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Database
with open('data\\processed\\processed_data.pkl', mode='rb') as f:
    x_test, x_train, y_test, y_train = pickle.load(f)
print(f'Shape X Train: {x_train.shape} | Shape X Test {x_test.shape}')
print(f'Shape Y Train: {y_train.shape} | Shape Y Test {y_test.shape}')
