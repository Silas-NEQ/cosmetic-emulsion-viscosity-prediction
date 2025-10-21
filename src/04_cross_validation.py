# Evaluating models

# Libraries used
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
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

# Cross Validation Function
def cross_val_model(model, splits, range_test, x, y, name):
    scores_list = []
    start_time = time.time()
    for i in range(range_test):
        kfold_data = KFold(n_splits=splits, shuffle=True, random_state=i)
        scores = cross_validate(
            model, X=x, y=y, cv=kfold_data, 
            scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'})
        scores_list.append({
            'Run': i + 1,
            'R²': round(np.mean(scores['test_r2']), 4),
            'MAE': round(-np.mean(scores['test_mae']), 2)
        })
    total_time = time.time() - start_time
    df_results = pd.DataFrame(scores_list)
    df_summary = df_results.mean(numeric_only=True).to_dict()
    df_std = df_results.std(numeric_only=True).to_dict()
    for key, value in df_std.items():
        df_summary[f'{key}_std'] = round(value, 4)
    df_summary['Model'] = name
    df_summary['Total_time(s)'] = round(total_time, 2)
    df_summary['Avg_time_per_run(s)'] = round(total_time/range_test, 2)
    cols_order = ['Model', 'R²', 'R²_std', 'MAE', 'MAE_std', 
                  'Total_time(s)', 'Avg_time_per_run(s)']
    df_summary = {col: df_summary[col] for col in cols_order}    
    return df_results, df_summary

# Models
# Neural Network
nn_model = MLPRegressor(activation='relu', alpha=0.01, hidden_layer_sizes= (50,),
                        learning_rate_init=0.001, solver='lbfgs', max_iter=2000)
# Random Forest
rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, max_features=0.5,
                                 min_samples_leaf=1, min_samples_split=2)
# Polymial Regression
poly_model = Pipeline(
    [('poly', PolynomialFeatures(degree=2)),('regressor', LinearRegression())])
# XGBoost
xgboost_model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)

# Evaluation
evaluation_list = [(nn_model, 'Neural Network'), (rf_model, 'Random Forest'),
                   (poly_model, 'Polynomial Regression'), (xgboost_model, 'XGBoost')]
results = []
for model, name in evaluation_list:
    _, summary = cross_val_model(
        model, splits=10, range_test=30, x=x_train, y=y_train, name=name)
    results.append(summary)
df_results = pd.DataFrame(results)
print(df_results.sort_values(by='R²', ascending=False).to_string(index=False))
# Saving results
df_results.to_csv('models\\model_results.csv', index=False, sep=';')