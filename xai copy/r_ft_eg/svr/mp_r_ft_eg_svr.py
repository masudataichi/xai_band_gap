import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

df_train = pd.read_csv("../../dataset/df_mp_merged.csv")
mp_eg = df_train.iloc[:, -1].to_numpy()
selected_feature = pd.read_csv("../../pfi/svr/csv/mp_sorted_best_selected_svr_all.csv").iloc[0:10,1].tolist()

x_line = [-3, 20]
y_line = [-3,20]

for i in range(len(selected_feature)):
    feature_value = df_train[selected_feature[i]]

    mp_eg_reshape = np.array(mp_eg).reshape(-1,1)
    feature_value_reshape = np.array(feature_value).reshape(-1,1)
    reg = LinearRegression()
    reg.fit(feature_value_reshape, mp_eg_reshape)

    corr_coef, _ = pearsonr(feature_value, mp_eg)

    print('ft_eg')
    print('a=', reg.coef_)
    print('b=', reg.intercept_)
    y_line_reg = reg.coef_[0] * x_line + reg.intercept_[0]
    plt.figure(figsize=(7, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(feature_value_reshape, mp_eg_reshape, color="#1f77b4")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel(selected_feature[i],fontsize=30)
    plt.ylabel("MP band gap (eV)",fontsize=30)
    plt.plot(x_line, y_line_reg, color = "#1f77b4", label="Regression line ")
    plt.plot(x_line, y_line, color="#1f77b4", linestyle="dashed", dashes=[7,8], label="Identity line")

    plt.text(0.05, 0.95, f'Slope: {reg.coef_[0][0]:.2f}\nIntercept: {reg.intercept_[0]:.2f}\nCorrelation: {corr_coef:.2f}', 
             fontsize=18, transform=plt.gca().transAxes, verticalalignment='top')

    # plt.legend(fontsize=18, frameon=False)
    # plt.xlim(-1, 13)
    plt.ylim(-1, 13)
    plt.tight_layout()
    plt.savefig("img/mp/ft_eg" + str(i) + ".png")