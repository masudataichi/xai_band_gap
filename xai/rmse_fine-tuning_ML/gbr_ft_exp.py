
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pandas as pd

# Load the dataset from a CSV file
df_exp = pd.read_csv('../dataset/df_exp_merged.csv')

# Split the dataset into features (x_train) and target (y_train)
x_train = df_exp.iloc[:, 1:-1] # Features: all columns except the first and last
y_train = df_exp.iloc[:,-1] # Target: the last column

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42, n_estimators=500)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [1000, 3000, 5000, 7000, 9000],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
    'max_depth': [1, 3, 5, 7, 9]
}

# Set up K-Fold cross-validation
# Split the data into 10 folds, shuffle the data, and set a random seed
kf_cv = KFold(n_splits=10, random_state=42, shuffle=True)

# Configure GridSearchCV for hyperparameter tuning
# Use negative RMSE as the scoring metric
grid_search = GridSearchCV(model, param_grid, cv=kf_cv, scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(x_train, y_train)

# Retrieve the best model found during grid search
best_model = grid_search.best_estimator_

# Print the best hyperparameters and the corresponding score
print("Best parameters found: ", grid_search.best_params_)
print("Best scores found: ", grid_search.best_score_)

# Best parameters found:  {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 3000}
# Best scores found:  -0.6715556514654288


# Convert grid search results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Convert the negative RMSE scores to positive for easier interpretation
for i in range(10):
    results[f'split{i}_test_score'] = -results[f'split{i}_test_score']

# Convert the mean score to positive as well
results['mean_test_score'] = -results['mean_test_score']

# SSave the results to a CSV file
output_results = results[['params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 
                          'split3_test_score', 'split4_test_score', 'split5_test_score', 
                          'split6_test_score', 'split7_test_score', 'split8_test_score', 
                          'split9_test_score', 'mean_test_score']]
output_results.to_csv('response_letter_grid_search/gbr_exp.csv', index=False)