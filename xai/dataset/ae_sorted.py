import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

df_exp = pd.read_csv("df_exp_merged.csv")

x_exp = df_exp.iloc[:, 1:-1]
y_exp = df_exp.iloc[:,-1]

df_mp = pd.read_csv("df_mp_merged.csv")

x_mp = df_mp.iloc[:, 1:-1]
y_mp = df_mp.iloc[:,-1]

x_line = [-3, 20]
y_line = [-3,20]

print("MAE")
print(np.mean(abs(pd.DataFrame(y_mp).iloc[:,0]-pd.DataFrame(y_exp).iloc[:,0])))
print("R2")
print(r2_score(y_exp, y_mp))
print("RMSE")
print(np.sqrt(mean_squared_error(y_exp, y_mp)))
print("MAPE")
print(np.mean((np.abs(np.array(y_exp)-np.array(y_mp))/np.array(y_exp)*100)))


ae_list = abs(pd.DataFrame(y_mp).iloc[:,0]-pd.DataFrame(y_exp).iloc[:,0]).to_list()
y_exp_list = pd.DataFrame(y_exp).iloc[:,0].to_list()
y_mp_list = pd.DataFrame(y_mp).iloc[:,0].to_list()

print("standard deviation")
print(np.std(y_mp_list))

print("outlier_MAE")
print(3*np.std(y_mp_list))

data_list = [ae_list, y_exp_list, y_mp_list]
index_list = df_exp.iloc[:,0].to_list()
columns_list = ["ae", "exp", "mp"]

ae_df = pd.DataFrame(data=data_list, index=columns_list, columns=index_list).transpose()

ae_df_sorted = ae_df.sort_values('ae', ascending=False)
ae_df_sorted.to_csv("ae_sorted.csv")

df_exp["mp"] = y_mp_list
df_exp["ae"] = ae_list
ae_df_sorted_with_feature = df_exp.sort_values('ae', ascending=False)
ae_df_sorted_with_feature.to_csv("ae_df_sorted_with_feature.csv")