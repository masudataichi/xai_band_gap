import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Load sorted feature names from a precomputed CSV file derived by PFI
sorted_feature = pd.read_csv("../../rmse_pfi/gbr/csv/exp_sorted_best_selected_gbr_all_rev.csv").iloc[:,1].tolist()

# Load the training dataset
df_train = pd.read_csv("../../dataset/df_exp_merged.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42, n_estimators=3000, learning_rate=0.05, max_depth=3)

# Train the model on the training dataset
model.fit(x_train, y_train)

# Number of intervals for ALE
n_hold = 50

# Loop through the top 10 sorted features to generate ALE plots
for j in range(10):
    # Extract the feature values for the current feature
    feature_values = x_train[[sorted_feature[j]]]
    min_val, max_val = np.min(feature_values), np.max(feature_values)

    # Define bin thresholds for splitting feature values
    thresholds = np.linspace(min_val, max_val, n_hold)
    # Initialize an array to store ALE effects for each bin

    # Loop through each bin to calculate the ALE effects
    ale_effects = np.zeros(n_hold - 1)
    for i in range(n_hold - 1):  
        lower, upper = thresholds[i], thresholds[i + 1]

        # Identify the samples within the current bin
        in_bin = (feature_values >= lower) & (feature_values < upper)
        sum_in_bin = np.sum(in_bin).iloc[0] # Count samples in the bin
        if sum_in_bin > 0: # Proceed if the bin is not empty
            # Get the indices of the samples within the bin
            inbin_true_index = in_bin[in_bin[sorted_feature[j]] == True].index

            # Create lower and upper copies of the training data for bin edges
            X_train_index = x_train.loc[inbin_true_index]
            X_lower, X_upper = X_train_index.copy(), X_train_index.copy()

            # Replace feature values with the bin's lower and upper edges
            X_lower.loc[:, [sorted_feature[j]]] = lower
            X_upper.loc[:, [sorted_feature[j]]] = upper
           
            # Predict the target values for both lower and upper edge data
            preds_lower = model.predict(X_lower)
            preds_upper = model.predict(X_upper)

            # Calculate the ALE effect for the bin
            ale_effects[i] = np.mean(preds_upper - preds_lower)

    # Compute the accumulated ALE effects across bins
    accumulated_effects = np.cumsum(ale_effects)

    # Plot the ALE plot for the current feature
    plt.figure(figsize=(9.1, 7.8))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.ylim(-1.8, 0.7)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.plot(thresholds[:-1], accumulated_effects, linewidth=4)
    plt.xlabel(sorted_feature[j],fontsize=28)
    plt.ylabel('ALE plot of band gap (eV)',fontsize=28)
    plt.title("GBR (experimental dataset)", fontsize=28)
    plt.savefig("img/exp_rev/exp_ale_gbr_rev" + str(j) + ".png")


    
