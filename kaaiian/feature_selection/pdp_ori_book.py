import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



def generate_simulation_data2():
    N = 1000
    X = np.random.uniform(-np.pi * 2, np.pi * 2, [N, 2])
    epsilon = np.random.normal(0, 0.1, N)
    y = 10 * np.sin(X[:, 0]) + X[:, 1] + epsilon

    return train_test_split(X, y ,test_size=0.2, random_state=42)


X_train, X_test, y_train, y_test = generate_simulation_data2()


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
        fig, ax = plt.subplots()
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
        )
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "Average Prediction",
            ylim = ylim
        )
        fig.suptitle(f"Partial Dependence PLot ({self.target_var_name})")

        fig.savefig("pdp_ori_book")

rf = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)


pdp = PartialDependence(rf, X_test, ["X0", "X1"])
pdp.partial_dependence("X0", n_grid=50)
pdp.plot()
