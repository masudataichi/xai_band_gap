
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

# データの読み込み
df_mp = pd.read_csv('../dataset/df_mp_merged.csv')

x_df = df_mp.iloc[:, 1:-1]
y_df = df_mp.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)



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

# Best parameters found:  {'max_depth': 20, 'max_features': None, 'n_estimators': 200}
# Best scores found:  -0.42651438444189554

results = pd.DataFrame(grid_search.cv_results_)
for i in range(10):
    results[f'split{i}_test_score'] = -results[f'split{i}_test_score']
results['mean_test_score'] = -results['mean_test_score']
output_results = results[['params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 
                          'split3_test_score', 'split4_test_score', 'split5_test_score', 
                          'split6_test_score', 'split7_test_score', 'split8_test_score', 
                          'split9_test_score', 'mean_test_score']]
output_results.to_csv('response_letter_grid_search/rfr_mp.csv', index=False)

y_test_pred = best_model.predict(x_test)

y_test_dif = abs(y_test_pred - y_test)

print("---------------------------")
print(sum(y_test_dif)/len(y_test_dif))
# MAE:  0.4741663711299402

