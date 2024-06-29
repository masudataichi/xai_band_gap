from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv("test.csv")
x_train = df_train.iloc[:]


scaler = StandardScaler().fit(x_train)
X_train_std = pd.DataFrame(scaler.transform(x_train))
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
print("========================std-nor===========================")
print(X_train_std)
normalizer_nor =  Normalizer().fit(x_train)
X_train_nor = pd.DataFrame(normalizer_nor.transform(x_train))
print("=========================std==================-")
print(X_train_nor)