# import eli5
# from eli5.sklearn import PermutationImportance
import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt


selected_feature = pd.read_csv("best_selected_svr.csv").iloc[:,1].tolist()
df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]


model = SVR(C=10, gamma=1)

# scaler = StandardScaler().fit(x_train)
# X_train_std = scaler.transform(x_train)
# normalizer = Normalizer().fit(X_train_std)
# X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
# X_train_std.columns = selected_feature
model.fit(x_train, y_train)
result = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=0, scoring='neg_mean_absolute_error')
df_pfi = pd.DataFrame(
   data={'var_name': x_train.columns, 'importance': result['importances_mean']}).sort_values('importance')

print(result)
df_pfi.iloc[::-1]['var_name'].to_csv('sorted_best_selected_svr_not_std.csv')

plt.figure(figsize=(30, 10))
plt.barh(df_pfi['var_name'], df_pfi['importance'])
plt.title('Permutation difference', fontsize=20)
plt.xlabel('defference', fontsize=8)
plt.grid()
plt.savefig("pfi_svr_not_std.png")
