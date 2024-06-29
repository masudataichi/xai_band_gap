import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

atom_feature = pd.read_csv("../dataset/lattice_volume_density.csv")
density = atom_feature["density"]

df_train = pd.read_csv("../dataset/df_exp_merged.csv")
exp_eg = df_train.iloc[:, -1].to_numpy()


corr_coef, _ = pearsonr(density, exp_eg)


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(density, exp_eg, color="#1f77b4")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("density",fontsize=30)
plt.ylabel("exp_eg", fontsize=30)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_density_exp_eg.png")
