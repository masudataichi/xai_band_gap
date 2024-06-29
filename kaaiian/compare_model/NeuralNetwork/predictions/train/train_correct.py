import pandas as pd

df = pd.read_csv('y_exp_train_predicted NN combined Band Gap.csv')
df_exp = 1.1546 * df + 0.3908

df_exp.to_csv("y_exp_train_predicted NN combined Band Gap_correct.csv", index=False)