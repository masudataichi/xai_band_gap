import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def generate_simulation_data():
    N = 1000
    x0 = np.random.uniform(-1,1,N)
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.binomial(1,0.5,N)
    epsilon = np.random.normal(0,0.1,N)

    X = np.column_stack((x0,x1,x2))

    y = x0 - 5 * x1 + 10 * x1 * x2 + epsilon

    return train_test_split(X,y,test_size=0.2,random_state=42)

X_train, X_test, y_train, y_test = generate_simulation_data()

def plot_scatter(x,y,title=None,xlabel=None,ylabel=None):
    plt.figure(figsize=(7, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(x,y,alpha=0.3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.tight_layout()
    plt.savefig("interaction_plot.png")

plot_scatter(X_train[:,1],y_train,xlabel="X1",ylabel="Y")