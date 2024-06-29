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


model_svr = SVR(C=10, gamma=1)
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
for i in range(100):
    alpha = i * 0.01 + 0.01
    model = Lasso(alpha=alpha)
    model.fit(X_train_std,y_train)  
    train_score = model.score(X_train_std,y_train)
    test_score = model.score(X_test_std,y_test)
    score_list.append(test_score)
    X_selected = x_train.columns[model.coef_!=0]
    if best_score < test_score:
        best_score = test_score
        best_alpha = alpha
        best_selected = X_selected

print(best_score)
print(best_alpha)
score_list_datframe = pd.DataFrame(score_list)
score_list_datframe.to_csv("score_list.csv")
print(len(best_selected))

scaler = StandardScaler().fit(x_train)
X_train_svr = scaler.transform(x_train)
X_test_svr = scaler.transform(x_test)
normalizer = Normalizer().fit(X_train_svr)
X_train_svr = pd.DataFrame(normalizer.transform(X_train_svr))
X_test_svr = pd.DataFrame(normalizer.transform(X_test_svr))
model_svr.fit(X_train_svr, y_train)
y_predict = pd.Series(model_svr.predict(X_test_svr))
y_predict.to_csv("y_predict.csv")
print("MAE")
print(sum(abs(y_predict-y_test)) / len(y_predict))

model_sel_svr = SVR(C=10, gamma=1)
x_train_sel = x_train[best_selected]
x_test_sel = x_test[best_selected]
scaler_sel = StandardScaler().fit(x_train_sel)
X_train_sel_svr = scaler_sel.transform(x_train_sel)
X_test_sel_svr = scaler_sel.transform(x_test_sel)
normalizer_sel = Normalizer().fit(X_train_sel_svr)
X_train_sel_svr = pd.DataFrame(normalizer_sel.transform(X_train_sel_svr))
X_test_sel_svr = pd.DataFrame(normalizer_sel.transform(X_test_sel_svr))
model_sel_svr.fit(X_train_sel_svr, y_train)
y_sel_predict = pd.Series(model_sel_svr.predict(X_test_sel_svr))
y_sel_predict.to_csv("y_predict_selection.csv")
print("MAE_selection")
print(sum(abs(y_sel_predict-y_test)) / len(y_sel_predict))