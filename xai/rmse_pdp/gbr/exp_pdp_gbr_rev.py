import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# Load the sorte features from a CSV file derived by PFI
sorted_feature = pd.read_csv("../../rmse_pfi/gbr/csv/exp_sorted_best_selected_gbr_all_rev.csv").iloc[:,1].tolist()

# Load the experimental dataset
df_exp = pd.read_csv("../../dataset/df_exp_merged.csv")

# Extract features and target variable
x_train = df_exp.iloc[:, 1:-1] # Features: all columns except the first and last
y_train = df_exp.iloc[:,-1] # Target: last column
x_train = x_train[sorted_feature]

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42, n_estimators=3000, learning_rate=0.05, max_depth=3)

# Train the model
model.fit(x_train, y_train)

# Define a class for Partial Dependence analysis
class PartialDependence:


    def __init__(self, estimator: any, X: np.ndarray, var_names: list[str]):
        """
        Initialize with a trained estimator, input features, and variable names.
        """
        self.estimator = estimator
        self.X = X
        self.var_names = var_names
        
    def _counterfactual_prediciton(
            self,
            idx_to_replace: int,
            value_to_replace: float
    ) -> np.ndarray:
        """
        Generate predictions after replacing a feature's values with a given value.
        """
        X_replaced = self.X.copy()
        X_replaced.iloc[:, idx_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)
        return y_pred
    
    def partial_dependence(
            self,
            var_name: str,
            n_grid: int = 50
    ) -> None:
        """
        Calculate partial dependence for a specific variable.
        """
        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)

        # Generate a grid of values within the variable's range
        value_range = np.linspace(
            self.X.iloc[:, var_index].min(),
            self.X.iloc[:, var_index].max(),
            num = n_grid
        )
   
        # Compute average predictions for each value in the grid
        averaga_prediction = np.array([
            self._counterfactual_prediciton(var_index, x).mean()
            for x in value_range
        ])

        # Store results in a DataFrame
        self.df_partial_dependence = pd.DataFrame(
            data = {var_name: value_range, "avg_pred": averaga_prediction}
        )

    def plot(self, i, ylim: list[float] | None = None) -> None:
        """
        Plot the partial dependence curve for the specified variable.
        """
        # Customize plot appearance
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(9.1, 7.8))
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel(self.target_var_name,fontsize=28)
        plt.ylabel("PDP of band gap (eV)",fontsize=28)

        # Plot the partial dependence curve
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
            linewidth=4,
        )

        # Set plot labels and limits
        ax.set(
            xlabel = self.target_var_name,
            ylabel = "PDP of band gap (eV)",
            ylim = ylim,
        )

        # Add a title and save the plot
        plt.title("GBR (experimental dataset)", fontsize=28)
        fig.savefig("img/exp_rev/exp_pdp_gbr_rev" + str(i))

# Instantiate the PartialDependence class with the trained model and dataset
pdp = PartialDependence(model, x_train, sorted_feature)

# Generate and plot Partial Dependence for the top 10 selected features
for i in range(10):
    pdp.partial_dependence(sorted_feature[i], n_grid=50)
    pdp.plot(i, ylim=(1.1,3.6))