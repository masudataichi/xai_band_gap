from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np


selected_feature = pd.read_csv("../../pfi/svr/csv/exp_sorted_best_selected_svr_all.csv").iloc[:,1].tolist()


df_train = pd.read_csv("../../dataset/df_exp_merged.csv")
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

n_hold = 50

for j in range(10):
    
    feature_values = x_train[[selected_feature[j]]]
    min_val, max_val = np.min(feature_values), np.max(feature_values)
    thresholds = np.linspace(min_val, max_val, n_hold)
    ale_effects = np.zeros(n_hold - 1)
    for i in range(n_hold - 1):  
        lower, upper = thresholds[i], thresholds[i + 1]
   
        # in_hold = X_train_std.query('avg_Density_(g/mL) <= upper and avg_Density_(g/mL) >= lower')
        in_bin = (feature_values >= lower) & (feature_values < upper)
        sum_in_bin = np.sum(in_bin).iloc[0]
        if sum_in_bin > 0:
            inbin_true_index = in_bin[in_bin[selected_feature[j]] == True].index
            X_train_std_index = x_train.loc[inbin_true_index]
            X_lower, X_upper = X_train_std_index.copy(), X_train_std_index.copy()
            X_lower.loc[:, [selected_feature[j]]] = lower
            X_upper.loc[:, [selected_feature[j]]] = upper
            # print(X_lower)
           
            X_lower = scaler.transform(X_lower)
            X_lower = pd.DataFrame(normalizer.transform(X_lower))
            X_upper = scaler.transform(X_upper)
            X_upper = pd.DataFrame(normalizer.transform(X_upper))
            X_lower.columns = x_train.columns
            X_upper.columns = x_train.columns
            preds_lower = model.predict(X_lower)
            preds_upper = model.predict(X_upper)
            ale_effects[i] = np.mean(preds_upper - preds_lower)
            #ここまでおけ

    accumulated_effects = np.cumsum(ale_effects)
    plt.figure(figsize=(7, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.ylim(-1.8, 0.7)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.plot(thresholds[:-1], accumulated_effects, linewidth=4)
    plt.xlabel(selected_feature[j],fontsize=28)
    plt.ylabel('ALE plot of band gap (eV)',fontsize=28)
    plt.title("SVR (experimental dataset)", fontsize=28)
    plt.tight_layout()
    plt.savefig("img/exp/exp_ale_svr" + str(j) + ".png")

