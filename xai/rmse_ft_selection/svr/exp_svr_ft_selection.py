import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the dataset
df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]
feature_columns = x_train.columns

# Initialize the SVR model with specified hyperparameters
model = SVR(C=10, gamma=1)

# Load feature importance rankings from a CSV file
sorted_features_path = '../../rmse_pfi/svr/csv/exp_sorted_best_selected_svr_all_rev.csv'
sorted_features = pd.read_csv(sorted_features_path)
ordered_features = sorted_features['var_name'].tolist()
ordered_features.reverse() # Reverse the order for feature removal

# Lists to store results for sorted feature elimination
results_sorted = []

# Define the number of features to remove at each step
num_features_to_remove = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Set a random seed for reproducibility
np.random.seed(42)
total_features = x_train.shape[1] # Total number of features

# Configure K-Fold cross-validation
kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

# Loop over the specified number of features to remove
for num_remove in num_features_to_remove:
    # Calculate remaining features after removal
    remaining_features = ordered_features[num_remove:]
    num_features = total_features - num_remove
    
    # Stop if no features are left
    if len(remaining_features) == 0:
        break

    selected_features = np.random.choice(x_train.columns, num_features, replace=False)

    # Create a subset of the dataset with the remaining features
    X_subset_sorted = x_train[remaining_features]

    # Apply scaling and normalization to the subset
    scaler_sorted = StandardScaler().fit(X_subset_sorted)
    X_train_std_sorted = scaler_sorted.transform(X_subset_sorted)
    normalizer_sorted = Normalizer().fit(X_train_std_sorted)
    X_train_std_sorted = pd.DataFrame(normalizer_sorted.transform(X_train_std_sorted))
    X_train_std_sorted.columns = remaining_features

    # Perform cross-validation to evaluate the model's performance
    scores_sorted = cross_val_score(model, X_train_std_sorted, y_train, cv=kf_cv, scoring='neg_root_mean_squared_error')

    # Compute the mean RMSE for the current subset of features
    mean_score_sorted = -np.mean(scores_sorted) # Convert negative RMSE to positive
    
    # Save the results
    results_sorted.append({
        'num_features_removed': num_remove,
        'remaining_features': len(remaining_features),
        'mean_rmse_score': mean_score_sorted
    })

# Convert results to a DataFrame 
results_sorted_df = pd.DataFrame(results_sorted)

# Plot the RMSE progression as features are removed
plt.figure(figsize=(9, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(results_sorted_df['num_features_removed'], results_sorted_df['mean_rmse_score'], marker='o', color='blue', linewidth=2.5, markersize=6.5)
plt.xticks(fontsize=25)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.yticks(fontsize=25)
plt.xlabel('Number of features removed',fontsize=28)
plt.ylabel('RMSE (eV)',fontsize=28)
plt.title("SVR (experimental dataset)", fontsize=28)

plt.tight_layout()
plt.savefig("img/exp_svr_ft_elimination.png")

# Save the results to a CSV file
results_sorted_df.to_csv('csv/exp_sorted_svr_ft_elimination.csv', index=False)

