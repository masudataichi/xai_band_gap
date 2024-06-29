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
y_pred = pd.read_csv("../predictions/test/y_exp_test_predicted NN combined Band Gap.csv")


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


err_list = (y_pred.iloc[:,0]-pd.DataFrame(y_test).iloc[:,0]).to_list()
ae_list = abs(y_pred.iloc[:,0]-pd.DataFrame(y_test).iloc[:,0]).to_list()
y_exp_list = pd.DataFrame(y_test).iloc[:,0].to_list()
y_pred_list = y_pred.iloc[:,0].to_list()

print("standard deviation")
print(np.std(y_pred_list))

print("outlier_MAE")
print(3*np.std(y_pred_list))

data_list = [err_list, ae_list, y_exp_list, y_pred_list]
index_list = df_test.iloc[:,0].to_list()
columns_list = ["err", "ae", "exp", "pred"]


err_df = pd.DataFrame(data=data_list, index=columns_list, columns=index_list).transpose()

err_df_sorted = err_df.sort_values('exp')
err_df_sorted_err_minus = err_df_sorted[err_df_sorted['err'] < 0]
err_df_sorted_err_plus = err_df_sorted[err_df_sorted['err'] >= 0]

err_df_sorted_num = len(err_df_sorted)
err_df_sorted_err_minus_num = len(err_df_sorted_err_minus)
err_df_sorted_err_plus_num = len(err_df_sorted_err_plus)

#########################################################################

err_df_sorted_non_outlier = err_df_sorted[err_df_sorted['ae'] < 3*np.std(y_pred_list)]
err_df_sorted_err_minus_non_outlier = err_df_sorted_non_outlier[err_df_sorted_non_outlier['err'] < 0]
err_df_sorted_err_plus_non_outlier = err_df_sorted_non_outlier[err_df_sorted_non_outlier['err'] >= 0]

err_df_sorted_len_non_outlier = len(err_df_sorted_non_outlier)
err_df_sorted_err_minus_len_non_outlier = len(err_df_sorted_err_minus_non_outlier)
err_df_sorted_err_plus_len_non_outlier = len(err_df_sorted_err_plus_non_outlier)

######################################################################

exp_num_total = [err_df_sorted_num]
exp_num_minus = [err_df_sorted_err_minus_num]
exp_num_plus = [err_df_sorted_err_plus_num]
exp_mae = [np.mean(err_df_sorted["ae"].to_list())]

exp_num_total_non_outlier = [err_df_sorted_len_non_outlier]
exp_num_minus_non_outlier = [err_df_sorted_err_minus_len_non_outlier]
exp_num_plus_non_outlier = [err_df_sorted_err_plus_len_non_outlier]
exp_mae_non_outlier = [np.mean(err_df_sorted_non_outlier["ae"].to_list())]



####################################################################
for i in range(7):
    if i == 0:
        err_df_sorted_exp = err_df_sorted[err_df_sorted['exp'] <= 2 * (i + 1)]
        err_df_sorted_non_outlier_exp = err_df_sorted_non_outlier[err_df_sorted_non_outlier['exp'] <= 2 * (i + 1)]
    elif 1 <= i <= 3: 
        err_df_sorted_exp = err_df_sorted[(err_df_sorted['exp'] > 2 * i) & (err_df_sorted['exp'] <= 2 * (i + 1))]
        err_df_sorted_non_outlier_exp = err_df_sorted_non_outlier[(err_df_sorted_non_outlier['exp'] > 2 * i) & (err_df_sorted_non_outlier['exp'] <= 2 * (i + 1))]
    elif i == 4:
        err_df_sorted_exp = err_df_sorted[err_df_sorted['exp'] > 2 * i]
        err_df_sorted_non_outlier_exp = err_df_sorted_non_outlier[err_df_sorted_non_outlier['exp'] > 2 * i]
    elif i == 5:
        err_df_sorted_exp = err_df_sorted[err_df_sorted['exp'] < 0.5]
        err_df_sorted_non_outlier_exp = err_df_sorted_non_outlier[err_df_sorted_non_outlier['exp'] < 0.5]      
    else:
        err_df_sorted_exp = err_df_sorted[err_df_sorted['exp'] > 4]
        err_df_sorted_non_outlier_exp = err_df_sorted_non_outlier[err_df_sorted_non_outlier['exp'] > 4]  
    
    err_df_sorted_exp_minus = err_df_sorted_exp[err_df_sorted_exp['err'] < 0]
    err_df_sorted_exp_plus = err_df_sorted_exp[err_df_sorted_exp['err'] >= 0]
    err_df_sorted_non_outlier_exp_minus = err_df_sorted_non_outlier_exp[err_df_sorted_non_outlier_exp['err'] < 0]
    err_df_sorted_non_outlier_exp_plus = err_df_sorted_non_outlier_exp[err_df_sorted_non_outlier_exp['err'] >= 0]

    exp_num_total.append(len(err_df_sorted_exp))
    exp_num_minus.append(len(err_df_sorted_exp_minus))
    exp_num_plus.append(len(err_df_sorted_exp_plus))
    exp_mae.append(np.mean(err_df_sorted_exp["ae"].to_list()))

    exp_num_total_non_outlier.append(len(err_df_sorted_non_outlier_exp))
    exp_num_minus_non_outlier.append(len(err_df_sorted_non_outlier_exp_minus))
    exp_num_plus_non_outlier.append(len(err_df_sorted_non_outlier_exp_plus))
    exp_mae_non_outlier.append(np.mean(err_df_sorted_non_outlier_exp["ae"].to_list()))
######################################################

exp_data_list = [exp_num_total, exp_num_minus, exp_num_plus, exp_mae, exp_num_total_non_outlier, exp_num_minus_non_outlier, exp_num_plus_non_outlier, exp_mae_non_outlier]
err_index_list = ["num_total", "num_minus", "num_plus", "mae", "num_total_non_outlier", "num_minus_non_outlier", "num_plus_non_outlier", "mae_non_outlier"]
err_columns_list = ["all_eg", "eg<=2", "2<eg<=4", "4<eg<=6", "6<eg<=8", "8<eg", "eg<0.5", "4<eg"]
exp_num_split_df = pd.DataFrame(data=exp_data_list, index=err_index_list, columns=err_columns_list)
print(exp_num_split_df)

exp_num_split_df.to_csv("exp_num_split_df_test.csv")

# print(ae_dataframe_sorted)

# plt.figure(figsize=(7, 6))
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.scatter(y_train, y_pred_train)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel("Experimental band gap [eV]",fontsize=30)
# plt.ylabel("Predicted band gap [eV]",fontsize=30)
# plt.plot(x_line, y_line)
# plt.xlim(-1, 13)
# plt.ylim(-1, 13)
# plt.tight_layout()
# plt.savefig("train_svr_calc" + ".png")