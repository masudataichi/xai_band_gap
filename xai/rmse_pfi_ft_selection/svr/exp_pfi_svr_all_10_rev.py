# import eli5
# from eli5.sklearn import PermutationImportance
import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt


df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]
# x_train = x_train[selected_feature]
feature_columns = x_train.columns

model = SVR(C=10, gamma=1)

#######################################################################

sorted_features_path = '../../rmse_pfi/svr/csv/exp_sorted_best_selected_svr_all_rev.csv'
sorted_features = pd.read_csv(sorted_features_path)
ordered_features = sorted_features['var_name'].tolist()
ordered_features.reverse()
num_features_to_remove = 50
remaining_features = ordered_features[num_features_to_remove:]
X_subset_sorted = x_train[remaining_features]

#######################################################################

scaler = StandardScaler().fit(X_subset_sorted)
X_train_std = scaler.transform(X_subset_sorted)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = remaining_features


model.fit(X_train_std, y_train)
# result = permutation_importance(model, X_train_std, y_train, n_repeats=10, random_state=0, scoring='neg_mean_absolute_error')
result = permutation_importance(model, X_train_std, y_train, n_repeats=10, random_state=0, scoring='neg_root_mean_squared_error')

df_pfi = pd.DataFrame(
   data={'var_name': X_train_std.columns, 'importance': result['importances_mean']}).sort_values('importance')

pd.concat([df_pfi.iloc[::-1]['var_name'],df_pfi.iloc[::-1]['importance']], axis=1).to_csv('csv/exp_sorted_best_selected_svr_all_rev.csv')
fig = plt.figure(figsize=(12.5, 4.8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
left, bottom, width, height = 0.5, 0.15, 0.45, 0.75  # 位置とサイズをFigureの幅と高さに対する比率で指定
ax = fig.add_axes([left, bottom, width, height])
plt.barh(df_pfi['var_name'].iloc[::-1][0:10].iloc[::-1], df_pfi['importance'].iloc[::-1][0:10].iloc[::-1])
# plt.title('Permutation difference', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel('Permutation importance (eV)',fontsize=23)
# plt.grid()
plt.xlim(0, 1.0)
plt.title("SVR (experimental dataset)", fontsize=25)

plt.savefig("img/exp_pfi_svr_all_10_rev.png")
