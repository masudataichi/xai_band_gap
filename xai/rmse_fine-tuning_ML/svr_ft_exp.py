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


# 特徴量とターゲットの分割
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]

# 標準化と正規化の前処理
scaler = StandardScaler().fit(x_train)
X_train_std = scaler.transform(x_train)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = x_train.columns


# SVRモデルのインスタンス作成
model = SVR()

# ハイパーパラメータの設定
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
}

kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

# GridSearchCVによるハイパーパラメータの探索
grid_search = GridSearchCV(model, param_grid, cv=kf_cv, scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train_std, y_train)

# 最適なモデルの取得
best_model = grid_search.best_estimator_

# テストデータに対する予測
# y_pred = best_model.predict(X_test_std)
print("Best parameters found: ", grid_search.best_params_)
# Best parameters found:  {'C': 10, 'gamma': 1}
print("Best scores found: ", grid_search.best_score_)
# Best scores found:  -0.7009935980824347

results = pd.DataFrame(grid_search.cv_results_)
for i in range(10):
    results[f'split{i}_test_score'] = -results[f'split{i}_test_score']
results['mean_test_score'] = -results['mean_test_score']
output_results = results[['params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 
                          'split3_test_score', 'split4_test_score', 'split5_test_score', 
                          'split6_test_score', 'split7_test_score', 'split8_test_score', 
                          'split9_test_score', 'mean_test_score']]
output_results.to_csv('response_letter_grid_search/svr_exp.csv', index=False)

split_num = 1
x_train_cv = df_exp.iloc[:, :-1]
for train_index, test_index in kf_cv.split(x_train):
    X_train_split, X_test_split = x_train_cv.iloc[train_index], x_train_cv.iloc[test_index]
    y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # トレーニングセットの保存
    train_split = pd.concat([X_train_split, y_train_split], axis=1)
    train_split.to_csv(f'exp_cv_dataset/train/split_train_{split_num}.csv', index=False)
    
    # テストセットの保存
    test_split = pd.concat([X_test_split, y_test_split], axis=1)
    test_split.to_csv(f'exp_cv_dataset/test/split_test_{split_num}.csv', index=False)
    
    split_num += 1