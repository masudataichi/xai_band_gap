import matplotlib.pyplot as plt
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


selected_feature = pd.read_csv("../../pfi/gbr/csv/mp_sorted_best_selected_gbr_all.csv").iloc[0:10,1].tolist()


df_train = pd.read_csv("../../dataset/df_mp_merged.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]
cor = x_train.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(cor, annot=True, square=True, cmap='RdBu_r')
plt.xticks(fontsize=22)  
plt.yticks(fontsize=22)
plt.tight_layout()
plt.savefig("img/mp_r_ft_gbr_10.png")
