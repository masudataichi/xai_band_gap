import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns

atom_feature = pd.read_csv("csv/feature_dict.csv").iloc[:, 1:-1]


cor = atom_feature.corr()


plt.figure(figsize=(30, 30))
plt.rcParams['font.family'] = 'Times New Roman'
sns.heatmap(cor, annot=True, square=True, cmap='RdBu_r')
plt.xticks(fontsize=22)  
plt.yticks(fontsize=22)
plt.tight_layout()

plt.savefig("img/r_atom_all.png")

