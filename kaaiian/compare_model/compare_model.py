import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



# create objects from custom code



def calc_mape(y_act, y_pred):
    ape = []
    for act, pred in zip(y_act, y_pred):
        if act != 0:
            val = abs(act-pred)/act+0.001 * 100
            ape.append(val)
        else:
            print('error')
    mape = sum(ape)/len(ape)
#    plt.plot(y_act, ape, 'ro')
#    plt.show()
    return mape

def return_metrics(y_true, y_pred):


    scores = [
            # r2_score(y_true, X_ensemble['svr']),
            # r2_score(y_true, X_ensemble['svr']),
#            r2_score(y_true, X_ensemble['gbr']),
#            r2_score(y_true, X_ensemble['rf']),
            r2_score(y_true, y_pred)]

    rmse = [
            # np.sqrt(mean_squared_error(y_true, X_ensemble['svr'])),
            # np.sqrt(mean_squared_error(y_true, X_ensemble['svr'])),
#            np.sqrt(mean_squared_error(y_true, X_ensemble['gbr'])),
#            np.sqrt(mean_squared_error(y_true, X_ensemble['rf'])),
            np.sqrt(mean_squared_error(y_true, y_pred))]

    mape = [
            # calc_mape(y_true, X_ensemble['svr']),
            # calc_mape(y_true, X_ensemble['svr']),
#            calc_mape(y_true, X_ensemble['gbr']),
#            calc_mape(y_true, X_ensemble['rf']),
            calc_mape(y_true, y_pred)]
    

    mae = [
    np.mean(abs(y_pred-y_true.to_numpy()))
    ]
        # np.mean(abs(X_ensemble['svr'].to_numpy()-y_true.to_numpy())),
    ae = abs(y_pred-y_true.to_numpy())
    
#    print(scores)
#    print(rmse)
#    print(mape)
    return scores, rmse, mape, mae, ae


# read in train and test data
df_exp_train = pd.read_csv('df_exp_train.csv')
df_exp_test = pd.read_csv('df_exp_test.csv')

y_exp_train = df_exp_train.iloc[:, -1]
y_exp_test = df_exp_test.iloc[:, -1]


# read in experimental predictions
svr_train = pd.read_csv('SupportVectorRegression/predictions/svr_train.csv', header=None)
svr_test = pd.read_csv('SupportVectorRegression/predictions/svr_test.csv', header=None)
gbr_train = pd.read_csv('GradientBoostingRegression/predictions/gbr_train.csv', header=None)
gbr_test = pd.read_csv('GradientBoostingRegression/predictions/gbr_test.csv', header=None)

rf_train = pd.read_csv('RandomForestRegression/predictions/rf_train.csv', header=None)
rf_test = pd.read_csv('RandomForestRegression/predictions/rf_test.csv', header=None)

lr_train = pd.read_csv('LinearRegression/predictions/lr_train.csv', header=None)
lr_test = pd.read_csv('LinearRegression/predictions/lr_test.csv', header=None)

# read in DFT predictions'
aflow_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN aflow Band Gap.csv')
aflow_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN aflow Band Gap.csv')

mp_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN mp Band Gap.csv')
mp_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN mp Band Gap.csv')

combined_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN combined Band Gap.csv')
combined_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN combined Band Gap.csv')

roost_train = pd.read_csv('Roost/predictions/roost_train.csv')
roost_test = pd.read_csv('Roost/predictions/roost_test.csv')

combined_train_correct = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN combined Band Gap_correct.csv')
combined_test_correct = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN combined Band Gap_correct.csv')

gan_train = pd.read_csv('GAN/predictions/result_gan_train_combined.csv')
gan_test = pd.read_csv('GAN/predictions/result_gan_test_modified.csv')

X_ensemble_train = pd.DataFrame(index=svr_train.index)
feature = "avg_metallic_valence"
X_ensemble_train[feature] = df_exp_train[feature]
X_ensemble_train["svr"] = svr_train
X_ensemble_train["gbr"] =  gbr_train
X_ensemble_train["rf"] =  rf_train
X_ensemble_train["lr"] = lr_train
X_ensemble_train["combined"] =  combined_train
X_ensemble_train["roost"] =  roost_train
X_ensemble_train["gan"] =  gan_train
X_ensemble_train["exp"] = df_exp_train["target"]

X_ensemble_test = pd.DataFrame(index=svr_test.index)
X_ensemble_test[feature] = df_exp_test[feature]
X_ensemble_test["svr"] = svr_test
X_ensemble_test["gbr"] =  gbr_test
X_ensemble_test["rf"] =  rf_test
X_ensemble_test["lr"] = lr_test
X_ensemble_test["combined"] =  combined_test
X_ensemble_test["roost"] =  roost_test
X_ensemble_test["gan"] =  gan_test
X_ensemble_test["exp"] = df_exp_test["target"]


columns_list = ["svr", "gbr", "rf", "lr", "combined", "roost", "gan"]
for columns in columns_list:
    train_r2, train_rmse, train_mape, train_mae, train_ae = return_metrics(y_exp_train, X_ensemble_train[columns])
    test_r2, test_rmse, test_mape, test_mae, test_ae = return_metrics(y_exp_test, X_ensemble_test[columns])
    print("==================================")
    print(columns)
    print(test_r2)
    print(test_mae)
    print(test_rmse)
    print("==================================")
    X_ensemble_train[columns + "_ae"] = train_ae
    X_ensemble_test[columns + "_ae"] = test_ae

    plt.figure(figsize=(6, 4))
    plt.ylim(-0.5, 3)
    plt.scatter(X_ensemble_train[feature], X_ensemble_train[columns + "_ae"], s =2)
    plt.savefig("result/" + feature + "/"+ columns + "_mae.png")



