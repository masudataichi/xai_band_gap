
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pandas as pd

# データの読み込み
df_mp = pd.read_csv('../dataset/df_mp_merged.csv')


# 特徴量とターゲットの分割
x_train = df_mp.iloc[:, 1:-1]
y_train = df_mp.iloc[:,-1]


# RandomForestRegressorモデルのインスタンス作成
model = RandomForestRegressor(random_state=42)

# ハイパーパラメータの設定
# param_grid = {
#     'n_estimators': [10, 50, 100, 200, 300, 400, 500],
#     'max_features': ['sqrt', 'log2', 1, None],
#     'max_depth': [5, 10, 15, 20, 25, 30, 50]
# }

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_features': ['sqrt', 'log2', 1, None],
    'max_depth': [5, 10, 15, 20, 25, 30]
}

# GridSearchCVによるハイパーパラメータの探索
kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

# GridSearchCVによるハイパーパラメータの探索
grid_search = GridSearchCV(model, param_grid, cv=kf_cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)

# 最適なモデルの取得
best_model = grid_search.best_estimator_

# テストデータに対する予測
# y_pred = best_model.predict(X_test_std)
print("Best parameters found: ", grid_search.best_params_)

print("Best scores found: ", grid_search.best_score_)

# Best parameters found:  {'max_depth': 25, 'max_features': None, 'n_estimators': 800}
# Best scores found:  -0.4808906278870815