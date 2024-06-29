import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Load the data
df_exp = pd.read_csv("../../dataset/df_exp_merged.csv")

# Prepare the feature matrix (X) and target vector (y)
x_train = df_exp.iloc[:, 1:-1]
y_train = df_exp.iloc[:,-1]

# Train the RandomForest model
model = GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=5)

model.fit(x_train, y_train)

# Extract feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importances
feature_names = x_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance
sorted_importance_df = importance_df.sort_values(by='Importance', ascending=False)

sorted_importance_df.to_csv('csv/exp_gbr_sorted_importance.csv')

fig = plt.figure(figsize=(12, 4.8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.barh(sorted_importance_df['Feature'][0:10].iloc[::-1], sorted_importance_df['Importance'][0:10].iloc[::-1])
# plt.title('Permutation difference', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('score',fontsize=20)
# plt.grid()
plt.tight_layout()
plt.savefig("img/exp_gbr_fi_10.png")