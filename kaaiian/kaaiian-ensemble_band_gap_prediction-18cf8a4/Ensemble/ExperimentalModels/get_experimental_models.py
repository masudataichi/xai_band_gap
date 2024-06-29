# =============================================================================
# # YOU NEED TO SET THE PATH TO MATCH THE LOCATION OF THE Ensemble FOLDER
# =============================================================================
import sys
sys.path.append("../MachineLearningFunctions")

# read in custom functions and classes
from MSE_ML_functions import CrossValidate
from MSE_ML_functions import DisplayData

# import code from the standard library 
import numpy as np
import pandas as pd
# read in machine learning code
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt

df_exp_test = pd.read_csv('/Users/masudataichi/Academic/tanabe_cheme_lab/undergrad_research/graduation_thesis/kaaiian/kaaiian-ensemble_band_gap_prediction-18cf8a4/Ensemble/ExperimentalData/df_exp_test.csv')
X_exp_test = df_exp_test.iloc[:,1:-1]
y_exp_test = df_exp_test.iloc[:,-1]
base_path = '/Users/masudataichi/Academic/tanabe_cheme_lab/undergrad_research/graduation_thesis/kaaiian/kaaiian-ensemble_band_gap_prediction-18cf8a4/Ensemble/'
# create objects from custom code
def train_models(X_exp_test):
    cv = CrossValidate()
    display = DisplayData()
    display.alpha = 1
    display.markersize = 8
    display.mfc='w'
    # read in the training split for experimental data
    df_exp_train = pd.read_csv(base_path+'ExperimentalData/df_exp_train.csv')
    X_exp_train = df_exp_train.iloc[:,1:-1]
    y_exp_train = df_exp_train.iloc[:,-1]
    
    # read in the test split for experimental data


    svr = SVR(C=10, gamma=1)  # r2, mse: 0.815124277636 0.396226799328
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=3)  # r2, mse: 0.826690242764 0.371438550849
    rf = RandomForestRegressor(n_estimators=500, max_features='sqrt') 
    lr = LinearRegression() 



    models = [gbr, svr, lr, rf]
    names = ['gbr', 'svr', 'lr', 'rf']

    recorded_cv = []
    scaler = StandardScaler().fit(X_exp_train)
    X_train = pd.DataFrame(scaler.transform(X_exp_train))
    normalizer = Normalizer().fit(X_train)
    X_train = pd.DataFrame(normalizer.transform(X_train))
    X_exp_test_ = X_exp_test
    for model, name in zip(models, names):

        X_exp_test = X_exp_test_

        if name == 'svr':
            path = 'ExperimentalModels/SupportVectorRegression/'
            X_exp_test = scaler.transform(X_exp_test)
            X_exp_test = pd.DataFrame(normalizer.transform(X_exp_test))
            y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_train, y_exp_train, model, N=10, random_state=1)
            y_predicted.sort_index(inplace=True)  
            y_predicted.to_csv("../../../compare_model/SupportVectorRegression/predictions/svr_train_calc.csv", index = False, header = False)


            print(y_predicted)
            print(y_exp_train)
            print(type(y_predicted))
            print(metrics)
            
            print("avg_mae")
            print(np.mean(metrics.loc["test_MAE"]))
            print("avg_rmse")
            print(np.mean(metrics.loc["test_rmse"]))
            print("avg_MAPE")
            print(np.mean(metrics.loc["test_MAPE"]))
            print("avg_score")
            print(np.mean(metrics.loc["test_score"]))

            # print("=========================================================================")
            # print(X_train)
            # print("=========================================================================")
            # print("-------------------------------------------------------------------------")
            # print(y_exp_train)
            # print("-------------------------------------------------------------------------")
            model.fit(X_train, y_exp_train)
            
        elif name == 'gbr':
            print("name")
            print(name)
            path = 'ExperimentalModels/GradientBoostingRegression/'
            X_exp_test = scaler.transform(X_exp_test)
            X_exp_test = pd.DataFrame(normalizer.transform(X_exp_test))
            y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_train, y_exp_train, model, N=10, random_state=1)
            y_predicted.sort_index(inplace=True)  
            y_predicted.to_csv("../../../compare_model/GradientBoostingRegression/predictions/gbr_train_calc_non_standard.csv", index = False, header = False)
            model.fit(X_train, y_exp_train)
            print(metrics)
            print("avg_mae")
            print(np.mean(metrics.loc["test_MAE"]))
            print("avg_rmse")
            print(np.mean(metrics.loc["test_rmse"]))
            print("avg_MAPE")
            print(np.mean(metrics.loc["test_MAPE"]))
            print("avg_score")
            print(np.mean(metrics.loc["test_score"]))
        elif name == 'rf':
            path = 'ExperimentalModels/RandomForestRegression/'
            y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_train, y_exp_train, model, N=10, random_state=1)
            y_predicted.sort_index(inplace=True)  
            y_predicted.to_csv("../../../compare_model/RandomForestRegression/predictions/rf_train_calc.csv", index = False, header = False)
            model.fit(X_exp_train, y_exp_train)
            print(metrics)
            print("avg_mae")
            print(np.mean(metrics.loc["test_MAE"]))
            print("avg_rmse")
            print(np.mean(metrics.loc["test_rmse"]))
            print("avg_MAPE")
            print(np.mean(metrics.loc["test_MAPE"]))
            print("avg_score")
            print(np.mean(metrics.loc["test_score"]))
        elif name == 'lr':
            path = 'ExperimentalModels/LinearRegression/'
            X_exp_test = scaler.transform(X_exp_test)
            X_exp_test = pd.DataFrame(normalizer.transform(X_exp_test))
            y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_train, y_exp_train, model, N=10, random_state=1)
            y_predicted.sort_index(inplace=True)  
            y_predicted.to_csv("../../../compare_model/LinearRegression/predictions/lr_train_calc.csv", index = False, header = False)
            model.fit(X_train, y_exp_train)
            print(metrics)
            print("avg_mae")
            print(np.mean(metrics.loc["test_MAE"]))
            print("avg_rmse")
            print(np.mean(metrics.loc["test_rmse"]))
            print("avg_MAPE")
            print(np.mean(metrics.loc["test_MAPE"]))
            print("avg_score")
            print(np.mean(metrics.loc["test_score"]))          
        else:
            print('error!')


        y_test_prediction = pd.Series(model.predict(X_exp_test))
        y_test_prediction.to_csv("../../../compare_model/GradientBoostingRegression/predictions/gbr_test_non_standard.csv", index = False, header = False)
        display.actual_vs_predicted(y_actual, y_predicted, data_label= name + ' prediction', save=True, save_name=base_path + path + 'figures/' + name)

        y_pre_index = pd.DataFrame(y_predicted.index)
        y_pre_index.to_csv(base_path + path + 'predictions/' + '_train.csv', index=False)
        # print("-------------y_predictedindex----------------")
        # y_predicted.index)
        # print("-------------y_predictedindex----------------")
        y_predicted.sort_index(inplace=True)   
        y_predicted.to_csv(base_path + path + 'predictions/' + name + '_train.csv', index=False)
        y_test_prediction.to_csv(base_path + path + 'predictions/' + name + '_test.csv', index=False)
        joblib.dump(model, base_path + path + 'model/' + name + '.pkl') 
        recorded_cv.append(metrics) 
    
    writer = pd.ExcelWriter(base_path + 'ExperimentalModels/model_metrics.xlsx')
    for metric, name in zip(recorded_cv, names):
        metric.to_excel(writer, sheet_name=name)
    return recorded_cv

def run():
    recorded_cv = train_models(X_exp_test)

run()