import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]

feature_columns = x_train.columns

# Define the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42, n_estimators=3000, learning_rate=0.05, max_depth=3)


# Load the list of features sorted by importance from a CSV file
sorted_features_path = '../../rmse_pfi/gbr/csv/exp_sorted_best_selected_gbr_all_rev.csv'
sorted_features = pd.read_csv(sorted_features_path)
ordered_features = sorted_features['var_name'].tolist()

# Initialize lists to store results for sorted and random feature elimination
results_sorted = []
results_random = []

# Define the number of features to remove in each step
num_features_to_remove = [0, 1, 2, 4, 8, 16, 32, 64]

# Set a random seed for reproducibility
np.random.seed(42)
total_features = x_train.shape[1]

# Configure K-Fold cross-validation
kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

# Iterate over the number of features to remove
for num_remove in num_features_to_remove:
    # Calculate the remaining features based on the sorted order
    remaining_features = ordered_features[num_remove:]
    num_features = total_features - num_remove
    
    # If no features are left, break the loop
    if len(remaining_features) == 0:
        break

    # Select features randomly
    selected_features = np.random.choice(x_train.columns, num_features, replace=False)

    # Create subsets of the dataset for sorted and random feature sets
    X_subset_sorted = x_train[remaining_features]
    X_subset_random = x_train[selected_features]
    
    # Perform cross-validation to evaluate the model's performance
    scores_sorted = cross_val_score(model, X_subset_sorted, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')
    scores_random = cross_val_score(model, X_subset_random, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')

    # Compute the mean RMSE for each feature elimination strategy
    mean_score_sorted = -np.mean(scores_sorted)
    mean_score_random = -np.mean(scores_random)  # RMSEなので負の値を正に変換
    
    # Save the results for each strategy
    results_sorted.append({
        'num_features_removed': num_remove,
        'remaining_features': len(remaining_features),
        'mean_rmse_score': mean_score_sorted
    })

    results_random.append({
        'num_features_removed': num_remove,
        'selected_features': len(selected_features),
        'mean_rmse_score': mean_score_random
    })

# Convert the results to DataFrames for easier handling
results_sorted_df = pd.DataFrame(results_sorted)
results_random_df = pd.DataFrame(results_random)

# Plot the progression of RMSE as features are removed
plt.figure(figsize=(9, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(results_sorted_df['num_features_removed'], results_sorted_df['mean_rmse_score'], marker='o', color='blue', label='PFI', linewidth=2.5, markersize=6.5)
plt.plot(results_random_df['num_features_removed'], results_random_df['mean_rmse_score'], marker='o', color='red', label='Random', linewidth=2.5, markersize=6.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Number of features removed',fontsize=28)
plt.ylabel('RMSE (eV)',fontsize=28)
plt.title("GBR (experimental dataset)", fontsize=28)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("img/exp_gbr_ft_elimination.png")

# Save the results to CSV files
results_sorted_df.to_csv('csv/exp_sorted_gbr_ft_elimination.csv', index=False)
results_random_df.to_csv('csv/exp_random_gbr_ft_elimination.csv', index=False)

