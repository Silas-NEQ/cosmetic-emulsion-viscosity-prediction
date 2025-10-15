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
    print(f'Model: {name} | MAE: {mae:.2f} | R²: {r2:.2f} | Processing time: {processing_time} s')
    return {'Model': name, 'MAE': mae, 'R2': r2, 'Processing time(s)': {processing_time}}