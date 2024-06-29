import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


atom_feature = pd.read_csv("../dataset/df_exp_merged.csv")
avg_density = atom_feature["avg_Density_(g/mL)"]
range_density = atom_feature["range_Density_(g/mL)"]

corr_coef, _ = pearsonr(avg_density, range_density)


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(avg_density, range_density, color="#1f77b4")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("avg_Density_(g/mL)",fontsize=30)
plt.ylabel("range_Density_(g/mL)", fontsize=30)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_avg_range_d.png")

