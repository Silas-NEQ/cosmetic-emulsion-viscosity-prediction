# Ensemble Test

# Library used
import pickle
from util import ensemble_score

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

# Analyse
finished_models = {'Neural Network': nn_model, 'Polynomial Regression': poly_model,
                   'XGBoost': xgb_model}
models_predicts, models_scores = ensemble_score(finished_models, x_test, y_test)
print(models_scores)