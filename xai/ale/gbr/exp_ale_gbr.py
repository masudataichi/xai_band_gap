from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


selected_feature = pd.read_csv("../../pfi/gbr/csv/exp_sorted_best_selected_gbr_all.csv").iloc[:,1].tolist()


df_train = pd.read_csv("../../dataset/df_exp_merged.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
# x_train = x_train[selected_feature]

model = GradientBoostingRegressor(random_state=42, n_estimators=5000, learning_rate=0.05, max_depth=5)


model.fit(x_train, y_train)

n_hold = 50

for j in range(10):
    
    feature_values = x_train[[selected_feature[j]]]
    min_val, max_val = np.min(feature_values), np.max(feature_values)
    thresholds = np.linspace(min_val, max_val, n_hold)
    ale_effects = np.zeros(n_hold - 1)
    for i in range(n_hold - 1):  
        lower, upper = thresholds[i], thresholds[i + 1]
   
        in_bin = (feature_values >= lower) & (feature_values < upper)
        sum_in_bin = np.sum(in_bin).iloc[0]
        if sum_in_bin > 0:
            inbin_true_index = in_bin[in_bin[selected_feature[j]] == True].index
            X_train_index = x_train.loc[inbin_true_index]
            X_lower, X_upper = X_train_index.copy(), X_train_index.copy()
            X_lower.loc[:, [selected_feature[j]]] = lower
            X_upper.loc[:, [selected_feature[j]]] = upper
            # print(X_lower)
           

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
    plt.title("GBR (experimental dataset)", fontsize=28)
    plt.tight_layout()
    plt.savefig("img/exp/exp_ale_gbr" + str(j) + ".png")


    
