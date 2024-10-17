import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np



selected_feature = pd.read_csv("../../pfi/rfr/csv/exp_sorted_best_selected_rfr_all_rev.csv").iloc[:,1].tolist()

df_train = pd.read_csv("../../dataset/df_exp_merged.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]
model = RandomForestRegressor(random_state=42, max_depth=20, max_features='sqrt', n_estimators=900)

model.fit(x_train, y_train)


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
        fig, ax = plt.subplots(figsize=(9.1, 7.8))
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel(self.target_var_name,fontsize=28)
        plt.ylabel("ICE plot of band gap (eV)",fontsize=28)
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
            color = "C0", 
            linewidth = 4
        )

        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["ind_pred"],
            color = "C0", 
            linewidth = 1.5,
            alpha = 0.05
        )
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "ICE plot of band gap (eV)",
            ylim = (0, 5)
            # ylim = (1.2, 3.5)
        )
        plt.title("RFR (experimental dataset)", fontsize=28)
        # plt.tight_layout()
        

        fig.savefig("img/exp_rev/exp_ice_rfr_rev" + str(i))



pdp = PartialDependence(model, x_train, selected_feature)

# pdp.partial_dependence(selected_feature[0], n_grid=50)
# pdp.plot(0)

for i in range(10):
    pdp.partial_dependence(selected_feature[i], n_grid=50)
    pdp.plot(i)