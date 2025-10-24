# Ensemble Test

# Library used
import pickle
from util import ensemble_score
import matplotlib.pyplot as plt
import seaborn as sns

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
# Table
finished_models = {'Neural Network': nn_model, 'Polynomial Regression': poly_model,
                   'XGBoost': xgb_model}
models_predicts, models_scores = ensemble_score(finished_models, x_test, y_test)
print(models_scores)
models_scores.to_csv('data\\results\\final_scores.csv', sep=';', index=False)
# Graphs
# Long Format Conversion
scores_long = models_scores.reset_index().melt(
    id_vars='index',
    var_name='Model',
    value_name='Score'
)
scores_long.rename(columns={'index': 'Metric'}, inplace=True)
# R²
r2_data = scores_long[scores_long['Metric'] == 'r2'].copy()
r2_data.sort_values(by='Score', ascending=False, inplace=True)
plt.figure(figsize=(8,5))
r2_plot = sns.barplot(data=r2_data,
                      x='Model', y='Score', capsize=0.2, palette='viridis')
for i, v in enumerate(r2_data['Score']):
    r2_plot.text(i, v + (0.01 * v), f"{v:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title('Model Comparison: R² Score', 
          fontsize=14, fontweight='bold')
plt.ylabel('R²')
plt.tight_layout()
plt.show()
# MAE
mae_data = scores_long[scores_long['Metric'] == 'mae'].copy()
mae_data.sort_values(by='Score', ascending=True, inplace=True)
plt.figure(figsize=(8,5))
mae_plot = sns.barplot(data=mae_data, 
            x='Model', y='Score', capsize=0.2, palette='viridis')
for i, v in enumerate(mae_data['Score']):
    mae_plot.text(i, v + (0.01 * v), f"{v:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title('Model Comparison: Mean Absolute Error (MAE)', 
          fontsize=14, fontweight='bold')
plt.ylabel('MAE')
plt.tight_layout()
plt.show()