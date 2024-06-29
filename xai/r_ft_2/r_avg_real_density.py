import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


atom_feature = pd.read_csv("../dataset/density_comparision.csv")
avg_density = atom_feature["avg_Density_(g/mL)"]
real_density = atom_feature["density"]

corr_coef, _ = pearsonr(avg_density, real_density)


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(avg_density, real_density, color="#1f77b4")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("avg_Density_(g/mL)",fontsize=28)
plt.ylabel("Mass density (g/mL)", fontsize=28)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_avg_real_density.png")

