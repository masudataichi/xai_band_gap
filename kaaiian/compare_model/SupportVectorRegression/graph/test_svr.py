import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score



df_train = pd.read_csv("../../df_exp_train.csv")
df_test = pd.read_csv("../../df_exp_test.csv")


x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:,-1]
y_pred = pd.read_csv("../predictions/svr_test.csv", header=None)


x_line = [-3, 20]
y_line = [-3,20]

print("MAE")
print(np.mean(abs(y_pred.iloc[:,0]-pd.DataFrame(y_test).iloc[:,0])))
print("R2")
print(r2_score(y_test, y_pred))
print("RMSE")
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE")
print(np.mean((np.abs(np.array(y_test)-np.array(y_pred).ravel())/np.array(y_test)*100)))


plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.scatter(y_test, y_pred)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Experimental band gap [eV]",fontsize=30)
plt.ylabel("Predicted band gap [eV]",fontsize=30)
plt.plot(x_line, y_line)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.tight_layout()
plt.savefig("test_svr" + ".png")