import shap
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

selected_feature = pd.read_csv("sorted_best_selected_svr.csv").iloc[:,1].tolist()
part_selected_feature = []
for i in range(1):
    part_selected_feature.append(selected_feature[i])

df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]
X1000 = shap.utils.sample(x_train, 100)
model = SVR(C=10, gamma=1)

scaler = StandardScaler().fit(x_train)
X_train_std = scaler.transform(x_train)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = part_selected_feature
model.fit(x_train, y_train)

explainer_ebm = shap.Explainer(model.predict, X1000)
shap_values_ebm = explainer_ebm(x_train)

shap.plots.beeswarm(shap_values_ebm)