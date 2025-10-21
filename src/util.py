# Functions

# Libraries used
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

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
