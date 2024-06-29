import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

selected_feature = pd.read_csv("best_selected_rfr.csv").iloc[:,1].tolist()
df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]

model = RandomForestRegressor(n_estimators=500, max_features='sqrt') 

scaler = StandardScaler().fit(x_train)
X_train_std = scaler.transform(x_train)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = selected_feature
model.fit(X_train_std, y_train)
result = permutation_importance(model, X_train_std, y_train, n_repeats=10, random_state=0, scoring='neg_mean_absolute_error')
df_pfi = pd.DataFrame(
   data={'var_name': X_train_std.columns, 'importance': result['importances_mean']}).sort_values('importance')
df_pfi.iloc[::-1]['var_name'].to_csv('sorted_best_selected_rfr.csv')

plt.figure(figsize=(30, 10))
plt.barh(df_pfi['var_name'], df_pfi['importance'])
plt.title('Permutation difference', fontsize=20)
plt.xlabel('defference', fontsize=8)
plt.grid()
plt.savefig("pfi_rfr.png")
plt.show()