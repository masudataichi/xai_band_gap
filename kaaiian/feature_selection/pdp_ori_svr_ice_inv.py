import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
x_train = x_train[selected_feature]
svr = SVR(C=10, gamma=1)
scaler = StandardScaler().fit(x_train)
X_train_std = pd.DataFrame(scaler.transform(x_train))


norms = np.linalg.norm(X_train_std, axis=1, keepdims=True)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = selected_feature
svr.fit(X_train_std, y_train)


class PartialDependence:


    def __init__(self, estimator: any, X: np.ndarray, var_names: list[str]):
        self.estimator = estimator
        self.X = X
        self.var_names = var_names
        
    def _counterfactual_prediciton(
            self,
            idx_to_replace: int,
            value_to_replace: float
    ) -> np.ndarray:
        X_replaced = self.X.copy()
        X_replaced.iloc[:, idx_to_replace] = value_to_replace
        X_replaced = pd.DataFrame(scaler.transform(X_replaced))
        X_replaced = pd.DataFrame(normalizer.transform(X_replaced))
        X_replaced.columns = selected_feature
        y_pred = self.estimator.predict(X_replaced)

        return y_pred
    
    def partial_dependence(
            self,
            var_name: str,
            n_grid: int = 50
    ) -> None:
        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)

        value_range = np.linspace(
            self.X.iloc[:, var_index].min(),
            self.X.iloc[:, var_index].max(),
            num = n_grid
        )

        individual_prediction = np.array([
            self._counterfactual_prediciton(var_index, x)
            for x in value_range
        ])
        averaga_prediction = np.array([
            self._counterfactual_prediciton(var_index, x).mean()
            for x in value_range
        ])

        self.df_partial_dependence = {var_name: value_range, "avg_pred": averaga_prediction, "ind_pred": individual_prediction}
        

    def plot(self, i, ylim: list[float] | None = None) -> None:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel(self.target_var_name,fontsize=30)
        plt.ylabel("Average Prediction",fontsize=30)
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
            color = "C0", 
            linewidth = 3
        )

        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["ind_pred"],
            color = "C0", 
            alpha = 0.02
        )
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "Predicted Band gap energy",
            ylim = (0, 5)
            # ylim = (1.2, 3.5)
        )
        plt.tight_layout()
        

        fig.savefig("pdp_ori_svr_ice_inv" + str(i))



pdp = PartialDependence(svr, x_train, selected_feature)

# pdp.partial_dependence(selected_feature[0], n_grid=50)
# pdp.plot(0)

for i in range(10):
    pdp.partial_dependence(selected_feature[i], n_grid=50)
    pdp.plot(i)