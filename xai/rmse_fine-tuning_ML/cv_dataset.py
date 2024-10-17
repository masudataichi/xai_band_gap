import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pandas as pd

# データの読み込み
df_exp = pd.read_csv('../dataset/df_exp_merged.csv')
df_mp = pd.read_csv('../dataset/df_mp_merged.csv')


# 特徴量とターゲットの分割
# x_train = df_exp.iloc[:, 1:-1]
y_exp_train = df_exp.iloc[:,-1]
y_mp_train = df_mp.iloc[:,-1]

print(y_exp_train)
y_exp_train.name = 'exp_target'
y_mp_train.name = 'mp_target'

kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)



split_num = 1
x_train_cv = df_exp.iloc[:, :-1]
for train_index, test_index in kf_cv.split(x_train_cv):
    X_train_split, X_test_split = x_train_cv.iloc[train_index], x_train_cv.iloc[test_index]
    y_exp_train_split, y_exp_test_split = y_exp_train.iloc[train_index], y_exp_train.iloc[test_index]
    y_mp_train_split, y_mp_test_split = y_mp_train.iloc[train_index], y_mp_train.iloc[test_index]
    
    # トレーニングセットの保存
    train_split = pd.concat([X_train_split, y_exp_train_split, y_mp_train_split], axis=1)
    train_split.to_csv(f'cv_dataset/train/split_train_{split_num}.csv', index=False)
    
    # テストセットの保存
    test_split = pd.concat([X_test_split, y_exp_test_split, y_mp_test_split], axis=1)
    test_split.to_csv(f'cv_dataset/test/split_test_{split_num}.csv', index=False)
    
    split_num += 1