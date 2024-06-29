

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import Callable
import scipy as sp

def f(X: np.ndarray) -> np.ndarray:
    """真の関数形：f(X0, X1) = X0 + X1^2"""
    return X[:, 0] + X[:, 1] ** 2

def generate_simulation_data(
    N: int = 10000,
    rho: float = 0.0,
    sigma_e: float = 0.01,
    f: Callable[[np.ndarray], np.ndarray] = f,
) -> tuple[np.ndarray, np.ndarray]:
    """シミュレーションデータを生成する関数"""

    # 区間[0, 1]の一様分布に従って、しかも相関する特徴量が作りたい
    # 正規分布を使って相関する特徴量を作る
    mu = np.array([0, 0])
    Sigma = np.array([[1, rho], [rho, 1]])
    X = np.random.multivariate_normal(mu, Sigma, N)
    for j in range(2):
        # 正規分布の確率密度関数を噛ませると区間[0, 1]の一様分布に従うようになる
        X[:, j] = sp.stats.norm().cdf(X[:, j])

    # ノイズ
    e = np.random.normal(0, sigma_e, N)

    # 目的変数
    y = f(X) + e

    return X, y


X, y = generate_simulation_data(rho=0.99)

plt.figure(figsize=(7, 6))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'
plt.scatter(X[:,0],X[:,1], alpha=0.3)
plt.xlabel("X1",fontsize=18)
plt.ylabel("X2",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("ale_cohe.png")

