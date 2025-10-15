# Data Modeling

# Libraries used
import pickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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