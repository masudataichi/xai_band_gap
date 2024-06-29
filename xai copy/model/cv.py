
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer
cross_validate(X_ensemble_train, y_exp_train, model, N=10, random_state=7, scale_data=False)
(X_train, y_exp_train, model, N=10, random_state=1)
def cross_validate(self, X, y, model, N=5, random_state=None, scale_data=False):
    
    '''
    Parameters
    ----------
    X: Pandas.DataFrame()
        matrix of features MxN, N feature, M rows
    y: Pandas.Series()
        vector of target values Mx1, , M rows
    model: sklearn object
        any machine learning model from sklearn library
    N: int, default=5
        number of cross validations you would like to perform
    random_state: int, default=None
        integer value greater than 0, determines cross validation split
    scale_data: boolean, default = False
        scale the features to have a mean of zero and variance of unity

    Return
    ----------
    y_actual: pd.Series()
        vector containing target values
    y_predicted: pd.Series()
        vector containing predicted target values on same index values as y_actual
    metrics: pd.DataFrame()
        metrics for each validation split
    data_index:
        returns the original index of return data
    '''

    kf = KFold(n_splits=N, random_state=random_state, shuffle=True)

    y_actual = []
    y_predicted = []
    metrics = {}
    split = 0
    max_value = 0
    data_index = []


    if len(self.nn_dict) >= 1:
        initial_weights = model.get_weights()

    for train_index, test_index in kf.split(y):

        data_index += list(test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        split += 1
        metrics[split] = {}



        if scale_data is True:



            # The data is is scaled to have zero mean and unit variance. This is because
            # many algorithms in the SKLEARN package behave poorly with 'wild' features
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train),
                                            index=X_train.index.values,
                                            columns=X_train.columns.values)
            # scaler = StandardScaler().fit(X_test)
            X_test = pd.DataFrame(scaler.transform(X_test),
                                            index=X_test.index.values,
                                            columns=X_test.columns.values)

            normalizer = Normalizer().fit(X_train)
            X_train = normalizer.transform(X_train)
            # normalizer = Normalizer().fit(X_test)
            X_test = normalizer.transform(X_test)


        if len(self.nn_dict) >= 1:    
            model.set_weights(initial_weights)
        model.fit(X_train, y_train, **self.nn_dict)
        predicted_test = model.predict(X_test)

        metrics[split]['test_rmse'] = np.sqrt(mean_squared_error(y_test, predicted_test))
        metrics[split]['test_MAPE'] = np.mean(np.abs(np.array(y_test)-np.array(predicted_test))/np.array(y_test)*100)
        metrics[split]['test_score'] = r2_score(y_test, predicted_test)
        metrics[split]['test_MAE'] = np.mean(np.abs(np.array(y_test)-np.array(predicted_test)))
        

        y_actual += list(y_test)
        y_predicted += list(predicted_test)

    metrics = pd.DataFrame(metrics)
    y_actual = pd.Series(y_actual)
    y_predicted = pd.Series(y_predicted, index = data_index, name='predicted')
    return y_actual, y_predicted, metrics, data_index