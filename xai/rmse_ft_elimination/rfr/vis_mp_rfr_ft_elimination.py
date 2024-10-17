import pandas as pd

from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt


# 結果をデータフレームに変換して表示
results_sorted_df = pd.read_csv("csv/mp_sorted_rfr_ft_elimination.csv")
results_random_df = pd.read_csv("csv/mp_random_rfr_ft_elimination.csv")

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
plt.title("RFR (MP dataset)", fontsize=28)

plt.ylim(0.545, 0.91)
plt.tight_layout()
plt.savefig("img/vis_mp_rfr_ft_elimination.png")
# 結果をCSVファイルに保存

