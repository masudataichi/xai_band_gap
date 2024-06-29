import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


atom_feature = pd.read_csv("../dataset/formula_unit_mass.csv")
volume = atom_feature["volume"]
formula_unit_mass = atom_feature["formula_unit_mass"]

corr_coef, _ = pearsonr(volume, formula_unit_mass)

volume_reshape = np.array(volume).reshape(-1,1)
formula_unit_mass_reshape = np.array(formula_unit_mass).reshape(-1,1)
reg = LinearRegression()
reg.fit(volume_reshape, formula_unit_mass_reshape)

plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(volume, formula_unit_mass, color="#1f77b4")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("volume",fontsize=30)
plt.ylabel("formula_unit_mass", fontsize=30)

plt.text(0.05, 0.95, f'Slope: {reg.coef_[0][0]:.2f}\nIntercept: {reg.intercept_[0]:.2f}\nCorrelation: {corr_coef:.2f}', 
             fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig("img/r_volume_formula_unit_mass.png")
