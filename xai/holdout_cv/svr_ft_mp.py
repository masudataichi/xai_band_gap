import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

# データの読み込み
df_mp = pd.read_csv('../dataset/df_mp_merged.csv')


# 特徴量とターゲットの分割
x_df = df_mp.iloc[:, 1:-1]
y_df = df_mp.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)

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
grid_search = GridSearchCV(model, param_grid, cv=kf_cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train_std, y_train)

# 最適なモデルの取得
best_model = grid_search.best_estimator_

print("Best parameters found: ", grid_search.best_params_)
# Best parameters found:  {'C': 10, 'gamma': 1}
print("Best scores found: ", grid_search.best_score_)
# Best scores found:  -0.4037237162463957

results = pd.DataFrame(grid_search.cv_results_)
for i in range(10):
    results[f'split{i}_test_score'] = -results[f'split{i}_test_score']
results['mean_test_score'] = -results['mean_test_score']
output_results = results[['params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 
                          'split3_test_score', 'split4_test_score', 'split5_test_score', 
                          'split6_test_score', 'split7_test_score', 'split8_test_score', 
                          'split9_test_score', 'mean_test_score']]
output_results.to_csv('response_letter_grid_search/svr_mp.csv', index=False)

scaler_test = StandardScaler().fit(x_test)
X_test_std = scaler_test.transform(x_test)
normalizer_test = Normalizer().fit(X_test_std)
X_test_std = pd.DataFrame(normalizer_test.transform(X_test_std))
X_test_std.columns = x_test.columns

y_test_pred = best_model.predict(X_test_std)

y_test_dif = abs(y_test_pred - y_test)

print("---------------------------")
print(sum(y_test_dif)/len(y_test_dif))
# MAE:  0.43612599663648516
