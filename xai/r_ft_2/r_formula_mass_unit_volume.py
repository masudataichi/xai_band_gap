import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


atom_feature = pd.read_csv("../dataset/formula_unit_mass.csv")
formula_unit_mass = atom_feature["formula_unit_mass"] 
volume = atom_feature["volume"]

corr_coef, _ = pearsonr(formula_unit_mass, volume)

formula_unit_mass_reshape = np.array(formula_unit_mass).reshape(-1,1)
volume_reshape = np.array(volume).reshape(-1,1)
reg = LinearRegression()
reg.fit(formula_unit_mass_reshape, volume_reshape)

plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(formula_unit_mass, volume, color="#1f77b4")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Atomic mass per unit cell (amu)",fontsize=28)
plt.ylabel("Volume per unit cell (Å³)", fontsize=28)

plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', 
             fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_formula_unit_mass_volume.png")
