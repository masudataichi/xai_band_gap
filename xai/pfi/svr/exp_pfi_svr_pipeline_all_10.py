# import eli5
# from eli5.sklearn import PermutationImportance
import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]
# x_train = x_train[selected_feature]
feature_columns = x_train.columns



pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 標準化
    ('normalizer', Normalizer()),  # 正規化
    ('model', SVR(C=10, gamma=1))
])

pipeline.fit(x_train, y_train)
result = permutation_importance(pipeline, x_train, y_train, n_repeats=10, random_state=0, scoring='neg_mean_absolute_error')

df_pfi = pd.DataFrame(
   data={'var_name': feature_columns, 'importance': result['importances_mean']}).sort_values('importance')

pd.concat([df_pfi.iloc[::-1]['var_name'],df_pfi.iloc[::-1]['importance']], axis=1).to_csv('csv/exp_sorted_best_selected_svr_pipeline_all.csv')
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
plt.savefig("img/exp_pfi_svr_pipeline_all_10.png")
