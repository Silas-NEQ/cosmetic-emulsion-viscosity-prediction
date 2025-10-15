# Data preprocessing

# Library used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Database
df = pd.read_csv('data\\raw\\cosmetic_emulsion_data.csv')

# X
x = df.iloc[:, 0:7].to_numpy()

# Y
y = df.iloc[:, 7].to_numpy()

# Shape
print(x.shape, y.shape)