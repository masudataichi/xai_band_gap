import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt

df_mp = pd.read_csv('../../dataset/df_mp_merged.csv')
x_train = df_mp.iloc[:, 1:-1]
y_train = df_mp.iloc[:,-1]

feature_columns = x_train.columns
model = SVR(C=10, gamma=1)

# scaler = StandardScaler().fit(x_train)
# X_train_std = scaler.transform(x_train)
# normalizer = Normalizer().fit(X_train_std)
# X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
# X_train_std.columns = feature_columns

# 並び順に特徴量を格納するCSVファイルの読み込み
sorted_features_path = '../../rmse_pfi/svr/csv/mp_sorted_best_selected_svr_all_rev.csv'
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
    

    ###################################################################################
    scaler_sorted = StandardScaler().fit(X_subset_sorted)
    X_train_std_sorted = scaler_sorted.transform(X_subset_sorted)
    normalizer_sorted = Normalizer().fit(X_train_std_sorted)
    X_train_std_sorted = pd.DataFrame(normalizer_sorted.transform(X_train_std_sorted))
    X_train_std_sorted.columns = remaining_features
    ###################################################################################

    ###################################################################################
    scaler_random = StandardScaler().fit(X_subset_random)
    X_train_std_random = scaler_random.transform(X_subset_random)
    normalizer_random = Normalizer().fit(X_train_std_random)
    X_train_std_random = pd.DataFrame(normalizer_random.transform(X_train_std_random))
    X_train_std_random.columns = selected_features
    ###################################################################################


    # モデルの初期化（計算時間を短縮するためにn_estimatorsを100に設定）
    
    # クロスバリデーションを実行し、平均精度を計算
    # scores_sorted = cross_val_score(model, X_subset_sorted, y_train, cv=kf_cv, scoring='neg_mean_absolute_error')
    # scores_random = cross_val_score(model, X_subset_random, y_train, cv=kf_cv, scoring='neg_mean_absolute_error')
    scores_sorted = cross_val_score(model, X_train_std_sorted, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')
    scores_random = cross_val_score(model, X_train_std_random, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')

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
plt.title("SVR (MP dataset)", fontsize=28)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("img/mp_svr_ft_elimination.png")

# 結果をCSVファイルに保存
results_sorted_df.to_csv('csv/mp_sorted_svr_ft_elimination.csv', index=False)
results_random_df.to_csv('csv/mp_random_svr_ft_elimination.csv', index=False)
