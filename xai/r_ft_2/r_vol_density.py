import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


atom_feature = pd.read_csv("../dataset/lattice_volume_density.csv")
volume = atom_feature["volume"]
density = atom_feature["density"]

corr_coef, _ = pearsonr(volume, density)


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(volume, density, color="#1f77b4")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("volume  [Å^3]",fontsize=30)
plt.ylabel("density [g/cm^3]", fontsize=30)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_vol_density.png")
