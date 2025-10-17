# Data Modeling

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

# Evaluation Function
def evaluate_model(model, name, x_train, x_test, y_train, y_test):
    time_start = time.time()
    model.fit(x_train, y_train)
    time_end = time.time()
    processing_time = time_end-time_start
    model_predict = model.predict(x_test)
    mae = mean_absolute_error(y_test, model_predict)
    r2 = r2_score(y_test, model_predict)
    print(f'Model: {name} | R²: {r2:.2f} | MAE: {mae:.2f} | Processing time: {processing_time:.2f}s')
    return {'Model': name, 'R2': r2, 'MAE': mae, 'Processing time(s)': processing_time}

# Evaluate models
results = []

# Neural Network Regressor
# R²: 0.89 | MAE: 256.59 | Processing time: 4.5s
nn_model = MLPRegressor(max_iter=2000, random_state=0)
results.append(evaluate_model(
    nn_model, 'Neural Network', x_train, x_test, y_train, y_test))

# SVM Regressor
# R²: 0.07 | MAE: 759.49 | Processing time: 0.07s
svm_model = SVR()
results.append(evaluate_model(svm_model, 'SVM', x_train, x_test, y_train, y_test))

# Random Forest Regressor
# R²: 0.88 | MAE: 267.52 | Processing time: 0.65s
rf_model = RandomForestRegressor(random_state=0)
results.append(evaluate_model(
    rf_model, 'Random Forest', x_train, x_test, y_train, y_test))

# Polynomial Regressor
# R²: 0.90 | MAE: 242.61 | Processing time: 0.01s
poly = PolynomialFeatures(degree=2)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)
poly_model = LinearRegression()
results.append(evaluate_model(
    poly_model, 'Polynomial Regression', x_poly_train, x_poly_test, y_train, y_test))

# XGBoost Regressor
# R²: 0.86 | MAE: 280.21 | Processing time: 0.81s
xgboost_model = XGBRegressor(random_state=0)
results.append(evaluate_model(
    xgboost_model, 'XGBoost', x_train, x_test, y_train, y_test))

print(f'Lista de reultados {results}')

# Tunning function
def tuning_model(model, name, params, x_train, x_test, y_train, y_test):
    grid_search = GridSearchCV(estimator=model, param_grid=params)
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    grid_search.fit(x, y)
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(f'Model: {name} | Best Result: {best_accuracy}')
    print(f'Best Parameters: {best_params}')
    return {'Model': name, 'Best Score': best_accuracy, 'Best Parameters': best_params}

# Tuning models
best_params = []
# Neural Network (0.8888)
# 'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50,), 
# 'learning_rate_init': 0.001, 'solver': 'lbfgs'
nn_grid_params = {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 
                  'activation': ['relu', 'tanh', 'logistic'], 
                  'alpha': [1e-4, 1e-3, 1e-2], 'learning_rate_init': [1e-3, 1e-4, 5e-4], 
                  'solver':['adam', 'lbfgs']}
best_params.append(tuning_model(
    MLPRegressor(), 'Neural Network', nn_grid_params, x_train, x_test,y_train, y_test))

# Random Forest (0.8743)
# 'max_depth': 20, 'max_features': 0.5, 'min_samples_leaf': 1, 
# 'min_samples_split': 2, 'n_estimators': 500
rf_grid_params = {'n_estimators':[100, 250, 500], 'max_depth': [None, 10, 20], 
                  'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 
                  'max_features': ['sqrt', 0.5, 'log2']}
best_params.append(tuning_model(RandomForestRegressor(), 'Random Forest', 
                                rf_grid_params, x_train, x_test,y_train, y_test))

# Polynomial Regressor (0.8931)
# 'poly__degree': 2
pipeline = Pipeline([('poly', PolynomialFeatures()),('regressor', LinearRegression())])
poly_grid_params = {'poly__degree': [2, 3]}
best_params.append(tuning_model(pipeline, 'Polynomial Regression', 
                        poly_grid_params, x_train, x_test, y_train, y_test))

# XGBoost (0.8827)
# 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300
xgboost_grid_params = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5], 
                       'learning_rate': [0.05, 0.1]}
best_params.append(tuning_model(XGBRegressor(), 'XGBoost', xgboost_grid_params, 
                                x_train, x_test, y_train, y_test))

print(best_params)