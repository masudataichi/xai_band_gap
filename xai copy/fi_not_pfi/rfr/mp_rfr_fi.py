import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the data
df_mp = pd.read_csv("../../dataset/df_mp_merged.csv")

# Prepare the feature matrix (X) and target vector (y)
x_train = df_mp.iloc[:, 1:-1]
y_train = df_mp.iloc[:,-1]

# Train the RandomForest model
model = RandomForestRegressor(random_state=42, max_depth=25, max_features=None, n_estimators=800)

model.fit(x_train, y_train)

# Extract feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importances
feature_names = x_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance
sorted_importance_df = importance_df.sort_values(by='Importance', ascending=False)

sorted_importance_df.to_csv('csv/mp_rfr_sorted_importance.csv')

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
plt.savefig("img/mp_rfr_fi_10.png")