from sklearn.model_selection import KFold
import pandas as pd

kf = KFold(n_splits=10, random_state=1, shuffle=True)

df = pd.read_csv('train.csv')
i=0
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
xy = df

for train_index, test_index in kf.split(y):
    i = i + 1
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xy_train, xy_test = xy.iloc[train_index], xy.iloc[test_index]
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    xy_train.to_csv("train_" + str(i)+ ".csv")
    xy_test.to_csv("test_" + str(i)+ ".csv")
    print(xy_train)