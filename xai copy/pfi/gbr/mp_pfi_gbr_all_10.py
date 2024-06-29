# import eli5
# from eli5.sklearn import PermutationImportance
import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt


df_mp = pd.read_csv('../../dataset/df_mp_merged.csv')
x_train = df_mp.iloc[:, 1:-1]
y_train = df_mp.iloc[:,-1]
# x_train = x_train[selected_feature]
feature_columns = x_train.columns

model = GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.07, max_depth=5)


model.fit(x_train, y_train)
result = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=0, scoring='neg_mean_absolute_error')

df_pfi = pd.DataFrame(
   data={'var_name': x_train.columns, 'importance': result['importances_mean']}).sort_values('importance')

pd.concat([df_pfi.iloc[::-1]['var_name'],df_pfi.iloc[::-1]['importance']], axis=1).to_csv('csv/mp_sorted_best_selected_gbr_all.csv')

fig = plt.figure(figsize=(12, 4.8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.barh(df_pfi['var_name'].iloc[::-1][0:10].iloc[::-1], df_pfi['importance'].iloc[::-1][0:10].iloc[::-1])
# plt.title('Permutation difference', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('score',fontsize=20)
# plt.grid()
plt.tight_layout()
plt.savefig("img/mp_pfi_gbr_all_10.png")
