import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression



df_train = pd.read_csv("../../df_exp_train.csv")
df_test = pd.read_csv("../../df_exp_test.csv")


x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:,-1]
y_pred = pd.read_csv("../predictions/result_gan_test_modified.csv")
y_pred_train = pd.read_csv("../predictions/result_gan_train_combined.csv")

x_line = [-3, 20]
y_line = [-3,20]

print("MAE")
print(np.mean(abs(y_pred_train.iloc[:,0]-pd.DataFrame(y_train).iloc[:,0])))
print("R2")
print(r2_score(y_train, y_pred_train))
print("RMSE")
print(np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("MAPE")
print(np.mean((np.abs(np.array(y_train)-np.array(y_pred_train).ravel())/np.array(y_train)*100)))


ae_list = abs(y_pred_train.iloc[:,0]-pd.DataFrame(y_train).iloc[:,0]).to_list()
y_exp_list = pd.DataFrame(y_train).iloc[:,0].to_list()
y_pred_list = y_pred_train.iloc[:,0].to_list()

print("standard deviation")
print(np.std(y_pred_list))

print("outlier_MAE")
print(3*np.std(y_pred_list))

data_list = [ae_list, y_exp_list, y_pred_list]
index_list = df_train.iloc[:,0].to_list()
columns_list = ["ae", "exp", "pred"]

ae_df = pd.DataFrame(data=data_list, index=columns_list, columns=index_list).transpose()
ae_df_sorted = ae_df.sort_values('ae', ascending=False)
ae_df_sorted.to_csv("ae_df_sorted.csv")


x_reg_exp_pred_rh = np.array(y_exp_list).reshape(-1,1)
y_reg_exp_pred_rh = np.array(y_pred_list).reshape(-1,1)
# print(ae_dataframe_sorted)
reg = LinearRegression()
reg.fit(x_reg_exp_pred_rh, y_reg_exp_pred_rh)
# reg_2 = np.polyfit(y_exp_list, y_pred_list, deg=1)

print('a=', reg.coef_)
print('b=', reg.intercept_)

# x_line_reg = np.linspace(min(y_exp_list), max(y_pred_list))
y_line_reg = reg.coef_[0] * x_line + reg.intercept_[0]


plt.figure(figsize=(7, 6.5))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'

plt.scatter(y_train, y_pred_train)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Experimental band gap (eV)",fontsize=30)
plt.ylabel("Predicted band gap (eV)",fontsize=30)
plt.title("CGAN", fontsize=30)
plt.plot(x_line, y_line_reg, color = "#1f77b4", label="Regression line ")
plt.plot(x_line, y_line, linestyle="dashed", dashes=[7,8], label="Identity line")
plt.legend(fontsize=24, frameon=False)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.tight_layout()
plt.savefig("train_gan_widen" + ".png")