from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np

df_train = pd.read_csv('df_exp_train.csv')
df_test = pd.read_csv('df_exp_test.csv')
X_exp_train = df_train.iloc[:,1:-1]
y_exp_train = df_train.iloc[:,-1]
X_exp_test = df_test.iloc[:,1:-1]
y_exp_test = df_test.iloc[:,-1]

X_exp_train_index = X_exp_train.index
X_exp_train_columns = X_exp_train.columns.values
scaler = StandardScaler()
X_exp_train = scaler.fit_transform(X_exp_train)
normalizer = Normalizer()
X_exp_train = pd.DataFrame(normalizer.fit_transform(X_exp_train), index=X_exp_train_index, columns=X_exp_train_columns)
y_exp_train = y_exp_train[X_exp_train_index]

X_exp_test_index = X_exp_test.index
X_exp_test_columns = X_exp_test.columns.values
scaler = StandardScaler()
X_exp_test = scaler.fit_transform(X_exp_test)
normalizer = Normalizer()
X_exp_test = pd.DataFrame(normalizer.fit_transform(X_exp_test), index=X_exp_test_index, columns=X_exp_test_columns)
y_exp_test = y_exp_test[X_exp_test_index]


model = Sequential()
model.add(Dense(10000, input_dim=129, kernel_regularizer=regularizers.l2(0.00000), kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5000,  kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0),  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
adm = optimizers.Adam(lr=0.005, decay=5e-4)
model.compile(loss='mean_squared_error', optimizer=adm, metrics=['accuracy'])
model.fit(X_exp_train, y_exp_train, validation_split=0.2, epochs=10, batch_size=64, verbose=0)

y_pred_test = model.predict(X_exp_test)
pred_list = []
for i in y_pred_test:
    for j in i:
        pred_list.append(j)
y_pred_test = np.array(pred_list)
print("-----------y_pred_test---------------")
print(y_pred_test)
print('------------------------------------')
y_exp_test_list = y_exp_test.values.tolist()
print("-----------y_exp_test_list---------------")
print(y_exp_test)
print('------------------------------------')
ae = abs(y_pred_test - y_exp_test)
mae = sum(ae)/len(ae)
print("MAE")
print(mae)
rmse = np.sqrt(sum(np.square(ae))/len(ae))
print('RMSE')
print(rmse)
y_pred_test.to_csv('NN_pred.csv', index=False)
