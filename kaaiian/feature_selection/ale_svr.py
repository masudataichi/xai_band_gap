from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np




selected_feature = pd.read_csv("sorted_best_selected_svr_all.csv").iloc[:,1].tolist()


df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
# x_train = x_train[selected_feature]

model = SVR(C=10, gamma=1)

scaler = StandardScaler().fit(x_train)
X_train_std = pd.DataFrame(scaler.transform(x_train))
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
# X_train_std.columns = selected_feature
X_train_std.columns = x_train.columns
model.fit(X_train_std, y_train)

n_hold = 500

feature_values = X_train_std[["range_metallic_valence"]]
min_val, max_val = np.min(feature_values), np.max(feature_values)
thresholds = np.linspace(min_val, max_val, n_hold)
ale_effects = np.zeros(n_hold - 1)
for i in range(n_hold - 1):  
    lower, upper = thresholds[i], thresholds[i + 1]
    # in_hold = X_train_std.query('avg_Density_(g/mL) <= upper and avg_Density_(g/mL) >= lower')
    in_bin = (feature_values >= lower) & (feature_values < upper)
    print(type(in_bin))
    print(in_bin)
    sum_in_bin = np.sum(in_bin).iloc[0]
    if sum_in_bin > 0:
        inbin_true_index = in_bin[in_bin['range_metallic_valence'] == True].index
        X_train_std_index = X_train_std.loc[inbin_true_index]
        X_lower, X_upper = X_train_std_index.copy(), X_train_std_index.copy()
        X_lower.loc[:, ["range_metallic_valence"]] = lower
        X_upper.loc[:, ["range_metallic_valence"]] = upper
        # print(X_lower)
        preds_lower = model.predict(X_lower)
        preds_upper = model.predict(X_upper)
        ale_effects[i] = np.mean(preds_upper - preds_lower)
        #ここまでおけ

accumulated_effects = np.cumsum(ale_effects)



plt.scatter(thresholds[:-1], accumulated_effects)
plt.xlabel('Feature')
plt.ylabel('Accumulated Local Effects')
plt.title('ALE Plot')
plt.show()


    
