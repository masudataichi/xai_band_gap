import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt

df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]

feature_columns = x_train.columns

# model = GradientBoostingRegressor(random_state=42, n_estimators=5000, learning_rate=0.05, max_depth=5)
model = GradientBoostingRegressor(random_state=42, n_estimators=3000, learning_rate=0.05, max_depth=3)


# 並び順に特徴量を格納するCSVファイルの読み込み
sorted_features_path = '../../rmse_pfi/gbr/csv/exp_sorted_best_selected_gbr_all_rev.csv'
sorted_features = pd.read_csv(sorted_features_path)
ordered_features = sorted_features['var_name'].tolist()

# 結果を保存するリスト
results_sorted = []
results_random = []

# 特徴量を順番に1つ、2つ、4つ、8つずつ減らす
num_features_to_remove = [0, 1, 2, 4, 8, 16, 32, 64]

np.random.seed(42)
total_features = x_train.shape[1]

# クロスバリデーションの設定
kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

for num_remove in num_features_to_remove:
    # 残す特徴量の数を計算
    remaining_features = ordered_features[num_remove:]
    num_features = total_features - num_remove
    
    # 特徴量が残っているか確認
    if len(remaining_features) == 0:
        break

    selected_features = np.random.choice(x_train.columns, num_features, replace=False)

    # データフレームのサブセットを作成
    X_subset_sorted = x_train[remaining_features]
    X_subset_random = x_train[selected_features]
    
    # モデルの初期化（計算時間を短縮するためにn_estimatorsを100に設定）
    
    # クロスバリデーションを実行し、平均精度を計算
    # scores_sorted = cross_val_score(model, X_subset_sorted, y_train, cv=kf_cv, scoring='neg_mean_absolute_error')
    # scores_random = cross_val_score(model, X_subset_random, y_train, cv=kf_cv, scoring='neg_mean_absolute_error')
    scores_sorted = cross_val_score(model, X_subset_sorted, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')
    scores_random = cross_val_score(model, X_subset_random, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')

    mean_score_sorted = -np.mean(scores_sorted)
    mean_score_random = -np.mean(scores_random)  # RMSEなので負の値を正に変換
    
    # 結果を保存
    results_sorted.append({
        'num_features_removed': num_remove,
        'remaining_features': len(remaining_features),
        'mean_rmse_score': mean_score_sorted
    })

    results_random.append({
        'num_features_removed': num_remove,
        'selected_features': len(selected_features),
        'mean_rmse_score': mean_score_random
    })
# 結果をデータフレームに変換して表示
results_sorted_df = pd.DataFrame(results_sorted)
results_random_df = pd.DataFrame(results_random)

# 予測精度の推移をプロット
plt.figure(figsize=(9, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(results_sorted_df['num_features_removed'], results_sorted_df['mean_rmse_score'], marker='o', color='blue', label='PFI', linewidth=2.5, markersize=6.5)
plt.plot(results_random_df['num_features_removed'], results_random_df['mean_rmse_score'], marker='o', color='red', label='Random', linewidth=2.5, markersize=6.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Number of features removed',fontsize=28)
plt.ylabel('RMSE (eV)',fontsize=28)
plt.title("GBR (experimental dataset)", fontsize=28)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("img/exp_gbr_ft_elimination.png")

# 結果をCSVファイルに保存
results_sorted_df.to_csv('csv/exp_sorted_gbr_ft_elimination.csv', index=False)
results_random_df.to_csv('csv/exp_random_gbr_ft_elimination.csv', index=False)
