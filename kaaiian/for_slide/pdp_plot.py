import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, Normalizer



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
        X_replaced[:, idx_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred
    
    def partial_dependence(
            self,
            var_name: str,
            n_grid: int = 50
    ) -> None:
        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)
        print(self.var_names)
        print(type(self.var_names))
        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num = n_grid
        )

        averaga_prediction = np.array([
            self._counterfactual_prediciton(var_index, x).mean()
            for x in value_range
        ])

        self.df_partial_dependence = pd.DataFrame(
            data = {var_name: value_range, "avg_pred": averaga_prediction}
        )

    def plot(self, ylim: list[float] | None = None) -> None:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel(self.target_var_name,fontsize=30)
        plt.ylabel("Average Prediction",fontsize=30)
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
            linewidth=3
        )
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "Predicted value",
            ylim = ylim
        )
        plt.tight_layout()

        fig.savefig("pdp_plot.png")


def generate_simulation_data():
    N = 1000
    x0 = np.random.uniform(-1,1,N)
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.binomial(1,0.5,N)
    epsilon = np.random.normal(0,0.1,N)

    X = np.column_stack((x0,x1,x2))

    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon

    return train_test_split(X,y,test_size=0.2,random_state=42)

X_train, X_test, y_train, y_test = generate_simulation_data()



rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train,y_train)

pdp = PartialDependence(rf, X_test, ["X0","X1","X2"])
pdp.partial_dependence("X1")
pdp.plot(ylim=(-6,6))