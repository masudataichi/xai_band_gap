import pandas as pd

df = pd.read_csv("df_mp_merged.csv")
filtered_df = df[df['target'] >= 0.005]

filtered_df.to_csv("df_mp_merged_nonzero.csv", index=False)
compound_list = df.iloc[:, 0]
mp_list = df.iloc[:, -1]
data_list = [compound_list, mp_list]
columns_list = ["compound", "mp"]
filtered_df_sorted = pd.DataFrame(data=data_list, index=columns_list).transpose().sort_values('mp', ascending=True)

filtered_df_sorted.to_csv("df_mp_merged_nonzero_sorted.csv")