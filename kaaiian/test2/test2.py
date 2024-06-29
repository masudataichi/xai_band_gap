import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

a_list = pd.Series([1,3,56,3,56,3,2,6])
b_list = pd.Series([12,6,9,11,22,33,6,7])
c_list  = pd.Series([5,6,92,4,4,5,2,3])
true_a_list = pd.Series([6,4,44,6,46,8,4,9])
true_b_list = pd.Series([22,10,5,14,29,20,10,11])
true_c_list = pd.Series([1,10,88,19,3,5,11,4])

print("MAE")
print(np.mean([np.mean(abs(true_a_list-a_list)),np.mean(abs(true_b_list-b_list)),np.mean(abs(true_c_list-c_list))]))
print("R2")
print(np.mean([r2_score(true_a_list, a_list),r2_score(true_b_list, b_list),r2_score(true_c_list, c_list)]))
print("RMSE")
print(np.mean([np.sqrt(mean_squared_error(true_a_list, a_list)),np.sqrt(mean_squared_error(true_b_list, b_list)),np.sqrt(mean_squared_error(true_c_list, c_list))]))
print("MAPE")
print(np.mean([np.mean((np.abs(np.array(true_a_list)-np.array(a_list))/np.array(true_a_list)*100)),np.mean((np.abs(np.array(true_b_list)-np.array(b_list))/np.array(true_b_list)*100)),np.mean((np.abs(np.array(true_c_list)-np.array(c_list))/np.array(true_c_list)*100))]))


all_list = pd.Series([1,3,56,3,56,3,2,6,12,6,9,11,22,33,6,7,5,6,92,4,4,5,2,3])
true_list = pd.Series([6,4,44,6,46,8,4,9,22,10,5,14,29,20,10,11,1,10,88,19,3,5,11,4])

print("MAE")
print(np.mean(abs(all_list-true_list)))
print("R2")
print(r2_score(true_list, all_list))
print("RMSE")
print(np.sqrt(mean_squared_error(true_list, all_list)))
print("MAPE")
print(np.mean((np.abs(np.array(true_list)-np.array(all_list))/np.array(true_list)*100)))

