# Dimensionality Analyse

# Libraries used
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Database
with open('data\\processed\\processed_data.pkl', mode='rb') as f:
    x_test, x_train, y_test, y_train = pickle.load(f)
print(f'Shape X Train: {x_train.shape} | Shape X Test {x_test.shape}')
print(f'Shape Y Train: {y_train.shape} | Shape Y Test {y_test.shape}')

# PCA
pca = PCA(n_components=0.95)
# New x_train
x_train_pca = pca.fit_transform(x_train)
# New x_test
x_test_pca = pca.transform(x_test)

# Models
# Neural Network
nn_model = MLPRegressor(activation='relu', alpha=0.01, hidden_layer_sizes= (50,),
                        learning_rate_init=0.001, solver='lbfgs', max_iter=2000)
# Polymial Regression
poly_model = Pipeline(
    [('poly', PolynomialFeatures(degree=2)),('regressor', LinearRegression())])
# XGBoost
xgboost_model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)