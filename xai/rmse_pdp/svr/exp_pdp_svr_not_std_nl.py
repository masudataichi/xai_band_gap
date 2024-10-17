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



selected_feature = pd.read_csv("../../pfi/svr/csv/exp_sorted_best_selected_svr_not_std_nl_all.csv").iloc[:,1].tolist()

df_exp = pd.read_csv("../../dataset/df_exp_merged.csv")
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]
x_train = x_train[selected_feature]


svr = SVR(C=10, gamma=1)



svr.fit(x_train, y_train)


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
   
        

        averaga_prediction = np.array([
            self._counterfactual_prediciton(var_index, x).mean()
            for x in value_range
        ])

        self.df_partial_dependence = pd.DataFrame(
            data = {var_name: value_range, "avg_pred": averaga_prediction}
        )

    def plot(self, i, ylim: list[float] | None = None) -> None:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel(self.target_var_name,fontsize=27)
        plt.ylabel("Predicted Band gap energy",fontsize=30)
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
            linewidth=3,
        )

        
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "Predicted Band gap energy",
            ylim = ylim,
        )
        # fig.suptitle(f"Partial Dependence PLot ({self.target_var_name})")
        plt.tight_layout()

        fig.savefig("img/exp/not_std_nl/exp_pdp_svr" + str(i))



pdp = PartialDependence(svr, x_train, selected_feature)
for i in range(10):
    pdp.partial_dependence(selected_feature[i], n_grid=50)
    pdp.plot(i, ylim=(1.1,3.0))