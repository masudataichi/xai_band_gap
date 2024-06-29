import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

atom_feature = pd.read_csv("csv/feature_dict.csv")
atom_radi = atom_feature["Atomic_Radus"]
atom_dens = atom_feature["Density_(g/mL)"]

corr_coef, _ = pearsonr(atom_radi, atom_dens)


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(atom_radi, atom_dens, color="#1f77b4")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Atomic_Radus",fontsize=30)
plt.ylabel("Density_(g/mL)", fontsize=30)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_atom_r_d.png")