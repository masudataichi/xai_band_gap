import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd


df_train = pd.read_csv("df_exp_train.csv")
df_test = pd.read_csv("df_exp_test.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:,-1]
scaler = StandardScaler().fit(x_train)
X_train_std = scaler.transform(x_train)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = x_train.columns

X_test_std = scaler.transform(x_test)
X_test_std = pd.DataFrame(normalizer.transform(X_test_std))
X_test_std.columns = x_test.columns

model = SVR(C=10, gamma=1)
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)
print(np.mean(abs(y_pred-y_test)))

x_line = [-3, 20]
y_line = [-3,20]
plt.scatter(y_test, y_pred)
plt.plot(x_line, y_line)
plt.xlim(-1, 13)
plt.ylim(-1, 13)
plt.savefig("svr_test" + ".png")