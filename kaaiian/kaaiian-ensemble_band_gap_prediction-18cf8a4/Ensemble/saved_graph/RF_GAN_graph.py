import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

y_true = pd.read_csv('../ExperimentalData/df_exp_test.csv').iloc[:, -1].to_numpy()
y_pred_rf = pd.read_csv('../ExperimentalModels/RandomForestRegression/predictions/rf_test.csv', header=None).iloc[:, -1].to_numpy()
y_pred_gan = pd.read_csv('../ExperimentalModels/GAN/predictions/result_gan_test_modified.csv').iloc[:, -1].to_numpy()
y_pred_rf_gan = pd.read_csv('RF_GAN.csv').iloc[:, -1].to_numpy()


x_line = [-3, 20]
y_line = [-3,20]

y_true_rh = np.array(y_true).reshape(-1,1)
y_pred_rf_rh = np.array(y_pred_rf).reshape(-1,1)
y_pred_gan_rh = np.array(y_pred_gan).reshape(-1,1)
y_pred_rf_gan_rh = np.array(y_pred_rf_gan).reshape(-1,1)
reg_rf = LinearRegression()
reg_rf.fit(y_true_rh, y_pred_rf_rh)
reg_gan = LinearRegression()
reg_gan.fit(y_true_rh, y_pred_gan_rh)
reg_rf_gan = LinearRegression()
reg_rf_gan.fit(y_true_rh, y_pred_rf_gan_rh)

print('RF')
print('a=', reg_rf.coef_)
print('b=', reg_rf.intercept_)
print('GAN')
print('a=', reg_gan.coef_)
print('b=', reg_gan.intercept_)
print('RF_GAN')
print('a=', reg_rf_gan.coef_)
print('b=', reg_rf_gan.intercept_)
y_line_reg_rf = reg_rf.coef_[0] * x_line + reg_rf.intercept_[0]
y_line_reg_gan = reg_gan.coef_[0] * x_line + reg_gan.intercept_[0]
y_line_reg_rf_gan = reg_rf_gan.coef_[0] * x_line + reg_rf_gan.intercept_[0]
plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
scatter_rf = plt.scatter(y_true, y_pred_rf, color="#ff7f0e", label="RFR")
scatter_gan = plt.scatter(y_true, y_pred_gan, color="#2ca02c", label="CGAN")
scatter_rf_gan = plt.scatter(y_true, y_pred_rf_gan, color="#1f77b4", label="CGAN + RFR")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Experimental band gap (eV)",fontsize=30)
plt.ylabel("Predicted band gap (eV)",fontsize=30)
line_rf_gan = plt.plot(x_line, y_line_reg_rf_gan, color = "#1f77b4", label="Regression line of CGAN + RFR")
line_gan = plt.plot(x_line, y_line_reg_gan, color = "#2ca02c", label="Regression line of CGAN")
line_rf = plt.plot(x_line, y_line_reg_rf, color = "#ff7f0e", label="Regression line of RFR")
line_dash = plt.plot(x_line, y_line, color="#1f77b4", linestyle="dashed", dashes=[7,8], label="Identity line")
first_legend = plt.legend(handles=[scatter_rf_gan, scatter_gan, scatter_rf], loc="lower right", fontsize=18, frameon=False)

plt.legend(handles=[line_rf_gan[0], line_gan[0], line_rf[0], line_dash[0]], loc="upper left", fontsize=18, frameon=False)
plt.gca().add_artist(first_legend)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.tight_layout()
plt.savefig("RF_GAN_merged.png")