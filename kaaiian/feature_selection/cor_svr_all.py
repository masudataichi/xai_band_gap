from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


selected_feature = pd.read_csv("sorted_best_selected_svr_all.csv").iloc[:,1].tolist()
# part_selected_feature = []
# for i in range(1):
#     part_selected_feature.append(selected_feature[i])

df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]
cor = x_train.corr()
plt.figure(figsize=(100, 50))
sns.heatmap(cor, annot=True)
plt.savefig("cor_svr_all" + ".png")
