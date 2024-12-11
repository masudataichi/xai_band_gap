import pandas  as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df_exp = pd.read_csv('../../dataset/df_exp_merged.csv')

# Extract feature columns (input variables) and target column (output variable)

x_train = df_exp.iloc[:, 1:-1] # Features: all columns except the first and last
y_train = df_exp.iloc[:,-1] # Target: last column
feature_columns = x_train.columns # Store feature column names for later use

# Initialize the Gradient Boosting Regressor model with specified hyperparameters
model = GradientBoostingRegressor(random_state=42, n_estimators=3000, learning_rate=0.05, max_depth=3)

# Train the Gradient Boosting Regressor model using the training data
model.fit(x_train, y_train)

# Perform permutation importance to evaluate feature importance
# n_repeats specifies the number of shuffles; scoring uses negative RMSE
result = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=0, scoring='neg_root_mean_squared_error')

# Create a DataFrame for feature importance results, sorted by importance
df_pfi = pd.DataFrame(
   data={'var_name': x_train.columns, 'importance': result['importances_mean']}).sort_values('importance')

# Save the reversed order of sorted feature importances to a CSV file
pd.concat([df_pfi.iloc[::-1]['var_name'],df_pfi.iloc[::-1]['importance']], axis=1).to_csv('csv/exp_sorted_best_selected_gbr_all_rev.csv')

fig = plt.figure(figsize=(12.5, 4.8))

# Customize font and tick styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Set the position and size of the plot
left, bottom, width, height = 0.5, 0.15, 0.45, 0.75  
ax = fig.add_axes([left, bottom, width, height])

# Plot the bar chart for the top 10 features in descending order of importance
plt.barh(df_pfi['var_name'].iloc[::-1][0:10].iloc[::-1], df_pfi['importance'].iloc[::-1][0:10].iloc[::-1])

# Set font sizes for ticks, labels, and title
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Permutation importance (eV)',fontsize=23)

# Limit x-axis range to a maximum value of 1.0
plt.xlim(0, 1.0)
plt.title("GBR (experimental dataset)", fontsize=25)

# Save the plot as a PNG image
plt.savefig("img/exp_pfi_gbr_all_10_rev.png")
