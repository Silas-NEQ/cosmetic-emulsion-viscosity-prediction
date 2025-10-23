# Ensemble Test

# Library used
import pickle

# Database
with open('data\\processed\\processed_data.pkl', mode='rb') as f:
    x_test, x_train, y_test, y_train = pickle.load(f)
print(f'Shape X Train: {x_train.shape} | Shape X Test {x_test.shape}')
print(f'Shape Y Train: {y_train.shape} | Shape Y Test {y_test.shape}')

# Models
# Neural Network
nn_model = pickle.load(open('models\\neural_network_model.pkl', 'rb'))
# Polymial Regression
poly_model = pickle.load(open('models\\polynomial_regression.pkl', 'rb'))
# XGBoost
xgb_model = pickle.load(open('models\\xgboost_model.pkl', 'rb'))