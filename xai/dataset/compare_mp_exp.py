import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

y_true = pd.read_csv('df_exp_merged.csv').iloc[:, -1].to_numpy()
y_mp = pd.read_csv('df_mp_merged.csv').iloc[:, -1].to_numpy()


x_line = [-3, 20]
y_line = [-3,20]

y_true_rh = np.array(y_true).reshape(-1,1)
y_mp = np.array(y_mp).reshape(-1,1)
reg_scheme2 = LinearRegression()
reg_scheme2.fit(y_true_rh, y_mp)

print('y_mp')
print('a=', reg_scheme2.coef_)
print('b=', reg_scheme2.intercept_)
y_line_reg_scheme2 = reg_scheme2.coef_[0] * x_line + reg_scheme2.intercept_[0]
plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.scatter(y_true, y_mp, color="#1f77b4")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Experimental band gap (eV)",fontsize=28)
plt.ylabel("MP band gap (eV)",fontsize=28)
plt.plot(x_line, y_line_reg_scheme2, color = "#1f77b4", label="Regression line ")
plt.plot(x_line, y_line, color="#1f77b4", linestyle="dashed", dashes=[5, 5], label="Identity line")
plt.legend(fontsize=18, frameon=False)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.tight_layout()
plt.savefig("compare_mp_exp.png")