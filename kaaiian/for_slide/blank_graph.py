

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import Callable
import scipy as sp
x_list = [0.45, 0.31, 0.5]
y_list = [0.25, 0.37, 0.5]
plt.figure(figsize=(7, 6))
# plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
plt.tick_params(bottom=False, left = False, right = False, top=False)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.xlim((-0.05,1.05))
plt.ylim((-0.05,1.05))
plt.scatter(x_list,y_list,s=200)
plt.tight_layout()
plt.savefig("blank_graph.png",transparent=True)

