
from sklearn.ensemble import GradientBoostingRegressor
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


# RandomForestRegressorモデルのインスタンス作成
model = GradientBoostingRegressor(random_state=42, n_estimators=500)

# ハイパーパラメータの設定
param_grid = {
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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


# Best parameters found:  {'learning_rate': 0.05, 'max_depth': 5}　n_estimators:500
# Best scores found:  -0.4701791423304467