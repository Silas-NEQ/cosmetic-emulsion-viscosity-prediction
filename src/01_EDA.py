# Exploratory Data Analysis

# Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Database
df = pd.read_csv('data\\raw\\cosmetic_emulsion_data.csv')

# Head
print(df.head())

# Shape
print(df.shape)

# Info
print(df.info())

# Columns
print(df.columns)

# Null values 
# No null values
print(df.isnull().sum())

# Numerical predictors
numerical_pred = df.select_dtypes(include=['number']).columns.drop('Viscosidade_final_cP')


# Scater graphs
for col in numerical_pred:
    sns.scatterplot(df, y='Viscosidade_final_cP', x=col)
    plt.title(f'{col} x Viscosidade_final_cP')
    #plt.show()
    #input('Press Enter to continue...')

# Boxplots
for col in numerical_pred:
    sns.boxplot(df, y=col)
    plt.title(f"{col} Distribution")
    #plt.show()
    #input('Press Enter to continue...')

# Correlation
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Map')
plt.show()