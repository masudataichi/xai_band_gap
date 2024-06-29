import pandas as pd
import scipy.stats as stats

# CSVファイルの読み込み（必要に応じてファイルパスを変更してください）
df_exp = pd.read_csv('df_exp_merged.csv')
df_mp = pd.read_csv('df_mp_merged.csv')

# 歪度の計算
skewness_exp = stats.skew(df_exp['target'])
skewness_mp = stats.skew(df_mp['target'])

# 尖度の計算
kurtosis_exp = stats.kurtosis(df_exp['target'])
kurtosis_mp = stats.kurtosis(df_mp['target'])

print(f"df_exp の歪度: {skewness_exp}, 尖度: {kurtosis_exp}")
print(f"df_mp の歪度: {skewness_mp}, 尖度: {kurtosis_mp}")

# df_exp の歪度: 1.6507469876176224, 尖度: 4.512323498513649
# df_mp の歪度: 1.2810053677968554, 尖度: 2.220096277345835
