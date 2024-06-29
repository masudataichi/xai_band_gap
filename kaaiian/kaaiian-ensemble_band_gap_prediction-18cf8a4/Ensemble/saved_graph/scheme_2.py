import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

y_true = pd.read_csv('../ExperimentalData/df_exp_test.csv').iloc[:, -1].to_numpy()
y_pred_scheme2 = pd.read_csv('scheme2.csv').iloc[:, -1].to_numpy()


x_line = [-3, 20]
y_line = [-3,20]

y_true_rh = np.array(y_true).reshape(-1,1)
y_pred_scheme2 = np.array(y_pred_scheme2).reshape(-1,1)
reg_scheme2 = LinearRegression()
reg_scheme2.fit(y_true_rh, y_pred_scheme2)

print('y_pred_scheme2')
print('a=', reg_scheme2.coef_)
print('b=', reg_scheme2.intercept_)
y_line_reg_scheme2 = reg_scheme2.coef_[0] * x_line + reg_scheme2.intercept_[0]
plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.scatter(y_true, y_pred_scheme2, color="#1f77b4")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Experimental band gap (eV)",fontsize=30)
plt.ylabel("Predicted band gap (eV)",fontsize=30)
plt.plot(x_line, y_line_reg_scheme2, color = "#1f77b4", label="Regression line ")
plt.plot(x_line, y_line, color="#1f77b4", linestyle="dashed", dashes=[7,8], label="Identity line")
plt.legend(fontsize=18, frameon=False)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.tight_layout()
plt.savefig("scheme2_saved.png")