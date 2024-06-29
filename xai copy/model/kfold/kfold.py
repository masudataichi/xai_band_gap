from sklearn.model_selection import KFold
import pandas as pd

kf = KFold(n_splits=10, random_state=1, shuffle=True)

df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
df_mp = pd.read_csv('../../dataset/df_mp_merged.csv')



xy_exp = df_exp
xy_mp = df_mp
i=0
j=0

for train_index, test_index in kf.split(xy_exp):
    i = i + 1
    xy_exp_train, xy_exp_test = xy_exp.iloc[train_index], xy_exp.iloc[test_index]
    xy_exp_train.to_csv("exp/train/train_" + str(i)+ ".csv")
    xy_exp_test.to_csv("exp/test/test_" + str(i)+ ".csv")


for train_index, test_index in kf.split(xy_mp):
    j = j + 1
    xy_mp_train, xy_mp_test = xy_mp.iloc[train_index], xy_mp.iloc[test_index]
    xy_mp_train.to_csv("mp/train/train_" + str(j)+ ".csv")
    xy_mp_test.to_csv("mp/test/test_" + str(j)+ ".csv")