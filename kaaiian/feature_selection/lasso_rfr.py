import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error


model_rfr = RandomForestRegressor(n_estimators=500, max_features='sqrt') 
df_train = pd.read_csv("df_exp_train.csv")
df_test = pd.read_csv("df_exp_test.csv")

x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:,-1]

ss = StandardScaler()
X_train_std = ss.fit_transform(x_train)

X_test_std = ss.transform(x_test)
best_score=0

score_list = []

scaler = StandardScaler().fit(x_train)
X_train_rfr = scaler.transform(x_train)
X_test_rfr = scaler.transform(x_test)
normalizer = Normalizer().fit(X_train_rfr)
X_train_rfr = pd.DataFrame(normalizer.transform(X_train_rfr))
X_test_rfr = pd.DataFrame(normalizer.transform(X_test_rfr))
model_rfr.fit(X_train_rfr, y_train)
y_predict = pd.Series(model_rfr.predict(X_test_rfr))
y_predict.to_csv("y_predict_rfr.csv")
print("MAE")
print(sum(abs(y_predict-y_test)) / len(y_predict))

mae_list = []
best_mae = 1000

model_sel_rfr = RandomForestRegressor(n_estimators=500, max_features='sqrt') 
selected_list = []
for i in range(1000):
    alpha = i * 0.001 + 0.001
    model = Lasso(alpha=alpha, max_iter=100000)
    model.fit(X_train_std,y_train)  
    X_selected = x_train.columns[model.coef_!=0]
    x_train_sel = x_train[X_selected]
    x_test_sel = x_test[X_selected]
    scaler_sel = StandardScaler().fit(x_train_sel)
    X_train_sel_rfr = scaler_sel.transform(x_train_sel)
    X_test_sel_rfr = scaler_sel.transform(x_test_sel)
    normalizer_sel = Normalizer().fit(X_train_sel_rfr)
    X_train_sel_rfr = pd.DataFrame(normalizer_sel.transform(X_train_sel_rfr))
    X_test_sel_rfr = pd.DataFrame(normalizer_sel.transform(X_test_sel_rfr))
    model_sel_rfr.fit(X_train_sel_rfr, y_train)
    y_sel_predict = pd.Series(model_sel_rfr.predict(X_test_sel_rfr))
    mae = sum(abs(y_sel_predict-y_test)) / len(y_sel_predict)
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
mae_list_datframe = pd.DataFrame(mae_list)
mae_list_datframe.to_csv("mae_list_rfr.csv")
print(len(best_selected))
selected_list_datframe = pd.DataFrame(selected_list)
selected_list_datframe.to_csv("select_list_rfr.csv")
best_selected_datframe = pd.DataFrame(best_selected)
best_selected_datframe.to_csv("best_selected_rfr.csv")
print(best_selected)