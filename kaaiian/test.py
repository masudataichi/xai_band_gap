from sklearn.svm import SVR
import pandas as pd

svr = SVR(C=150, gamma=0.003)

model = svr
d = {'svr':[0.434641, 0.747355, 2.257211, 2.423701, 1.095716],
     'gbr': [0.663085, 1.004261, 2.190738, 2.596623, 1.080490],
     'rf': [0.575086, 1.081021, 1.992988, 2.611760, 1.436220]}
list1 = [0.5, 0.8, 2.3, 2.5, 1.1]

y_train = pd.Series(data=list1)
X_train = pd.DataFrame(d)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_train)

print(y_pred)