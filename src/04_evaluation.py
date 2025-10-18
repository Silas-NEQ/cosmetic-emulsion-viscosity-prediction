# Evaluating models

# Libraries used
import pickle
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Database
with open('data\\processed\\processed_data.pkl', mode='rb') as f:
    x_test, x_train, y_test, y_train = pickle.load(f)
print(f'Shape X Train: {x_train.shape} | Shape X Test {x_test.shape}')
print(f'Shape Y Train: {y_train.shape} | Shape Y Test {y_test.shape}')
