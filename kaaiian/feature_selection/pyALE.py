from PyALE import ale
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
X_train_std.columns = selected_feature
model.fit(X_train_std, y_train)
print(type(X_train_std))
print(X_train_std)
ale_eff = ale(
    X=X_train_std, model=model, feature=["avg_Density_(g/mL)"], grid_size=500, include_CI=False
)
print(ale_eff)

plt.scatter(ale_eff.index, ale_eff.loc[:,"eff"])
plt.xlabel('Feature')
plt.ylabel('Accumulated Local Effects')
plt.title('ALE Plot')
plt.show()
