from sklearn.model_selection import KFold
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np

df = pd.read_csv("df_exp_train.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

kf = KFold(n_splits=5, random_state=1, shuffle=True)

y_actual = []
y_predicted = []
metrics = {}
split = 0
max_value = 0
data_index = []


all_mae_list = []
all_selected_list = []
all_best_selected = []
for train_index, test_index in kf.split(y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    ss = StandardScaler()
    X_train_std = ss.fit_transform(X_train)



    split += 1
    metrics[split] = {}


    model_sel_svr = SVR(C=10, gamma=1)
    selected_list = []
    mae_list = []
    best_mae = 10000
    
    for i in range(500):
        alpha = i * 0.001 + 0.001
        model = Lasso(alpha=alpha, max_iter=100000)
        model.fit(X_train_std,y_train)  
        X_selected = X_train.columns[model.coef_!=0]
        x_train_sel = X_train[X_selected]
        x_test_sel = X_test[X_selected]
        scaler_sel = StandardScaler().fit(x_train_sel)
        X_train_sel_svr = scaler_sel.transform(x_train_sel)
        X_test_sel_svr = scaler_sel.transform(x_test_sel)
        normalizer_sel = Normalizer().fit(X_train_sel_svr)
        X_train_sel_svr = pd.DataFrame(normalizer_sel.transform(X_train_sel_svr))
        X_test_sel_svr = pd.DataFrame(normalizer_sel.transform(X_test_sel_svr))
        model_sel_svr.fit(X_train_sel_svr, y_train)
        y_sel_predict = pd.Series(model_sel_svr.predict(X_test_sel_svr))
        mae = sum(abs(y_sel_predict-y_test.reset_index()["target"])) / len(y_sel_predict)
        mae_list.append(mae)
        selected_list.append(len(X_selected))
        
        if best_mae > mae:
            best_mae = mae
            best_alpha = alpha
            best_selected = X_selected
    print("best_mae")
    print(best_mae)
    print("best_alpha")
    print(best_alpha)
    all_mae_list.append(mae_list)
    print(len(best_selected))
    all_selected_list.append(selected_list)
    all_best_selected.append(best_selected)
all_mae_list_datframe = pd.DataFrame(all_mae_list)
all_mae_list_datframe.to_csv("all_mae_list_svr_cv.csv")
all_selected_list_datframe = pd.DataFrame(all_selected_list)
all_selected_list_datframe.to_csv("all_select_list_svr_cv.csv")
all_best_selected_datframe = pd.DataFrame(all_best_selected)
all_best_selected_datframe.to_csv("all_best_selected_svr_cv.csv")
all_mae_list_datframe_T = pd.DataFrame(np.array(all_mae_list).T)
all_mae_list_datframe_T.to_csv("all_mae_list_T_svr_cv.csv")
all_mae_list_T = np.array(all_mae_list).T
avg_mae_list = []
for i in range(len(all_mae_list_T)):
    avg_mae_list.append(sum(all_mae_list_T[i])/len(all_mae_list_T[i]))
print("min_avg_mae_list")
print(min(avg_mae_list))
avg_mae_list_datframe = pd.DataFrame(avg_mae_list)
avg_mae_list_datframe.to_csv("avg_mae_list_svr_cv.csv")


# for i in range(len(all_mae_list)):
#     all_mae_list[i]
    # metrics[split]['test_rmse'] = np.sqrt(mean_squared_error(y_test, predicted_test))
    # metrics[split]['test_MAE'] = np.mean(np.abs(np.array(y_test)-np.array(predicted_test))/np.array(y_test)*100)
    # metrics[split]['test_score'] = r2_score(y_test, predicted_test)
    # metrics[split]['alpha']
    









