# Final Models Set

# Libraries used
import pickle
import pandas as pd
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

# Models
# Neural Network
nn_model = MLPRegressor(activation='relu', alpha=0.01, hidden_layer_sizes= (50,),
                        learning_rate_init=0.001, solver='lbfgs', max_iter=2000)
nn_model.fit(x_train, y_train)
pickle.dump(nn_model, open('models\\neural_network_model.pkl', 'wb'))
# Polymial Regression
poly_model = Pipeline(
    [('poly', PolynomialFeatures(degree=2)),('regressor', LinearRegression())])
poly_model.fit(x_train, y_train)
pickle.dump(poly_model, open('models\\polynomial_regression.pkl', 'wb'))
# XGBoost
xgboost_model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
xgboost_model.fit(x_train, y_train)
pickle.dump(xgboost_model, open('models\\xgboost_model.pkl', 'wb'))

# Summary
summary = pd.DataFrame({
    'models': ['NeuralNetwork', 'PolynomialRegression', 'XGBoost'],
    'params': [
        str(nn_model.get_params()),
        str(poly_model.get_params()),
        str(xgboost_model.get_params())
    ]
})
summary.to_csv('models\\models_summary.csv', sep=';', index= False)