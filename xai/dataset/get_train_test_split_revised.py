from feature_assign import VectorizeFormulaMinimal
import pandas as pd

from sklearn.model_selection import train_test_split

vf = VectorizeFormulaMinimal()

# read in band gap data from Zhou et al. publication:
# J. Phys. Chem. Lett., 2018, 9 (7), pp 1668–1673
#DOI: 10.1021/acs.jpclett.8b00124
#Publication Date (Web): March 13, 2018
df_band_gap = pd.read_csv('band_gap_results_cleaned.csv') # excel sheet was edited to fix formula "GaAs0.1P0.9G1128" to "GaAs0.1P0.9"
df_band_gap = df_band_gap.drop(columns=['Material ID'])

# take the average of duplicte composition entries
df_band_gap_exp = df_band_gap.groupby('Formula').mean().reset_index().drop(columns=['Band Gap (eV)', 'Formation Energy (eV/atom)'])
df_band_gap_mp = df_band_gap.groupby('Formula').mean().reset_index().drop(columns=['Experimental Band Gap (eV)', 'Formation Energy (eV/atom)'])

# separate the metal and non-metal compounds
# df_band_gap_non_metal = df_band_gap[df_band_gap['Eg (eV)'] > 0]
# df_band_gap_metal = df_band_gap[df_band_gap['Eg (eV)'] == 0]

# randomly select train and test split
df_train_exp = df_band_gap_exp.sample(frac=0.8, random_state=256)
df_train_mp = df_band_gap_mp.sample(frac=0.8, random_state=256)
df_test_exp = df_band_gap_exp.iloc[~df_band_gap_exp.index.isin(df_train_exp.index.values)]
df_test_mp = df_band_gap_mp.iloc[~df_band_gap_mp.index.isin(df_train_mp.index.values)]

# rename columns for use with feature generation
df_band_gap_exp.columns = ['formula', 'target']
df_band_gap_mp.columns = ['formula', 'target']
df_train_exp.columns = ['formula', 'target']
df_train_mp.columns = ['formula', 'target']
df_test_exp.columns = ['formula', 'target']
df_test_mp.columns = ['formula', 'target']

# save the formula and band gap for train and test splits
df_band_gap_exp.to_csv('test/merged_data_exp_2.csv')
df_band_gap_mp.to_csv('test/merged_data_mp_2.csv')
df_train_exp.to_csv('test/train_exp_2.csv')
df_train_mp.to_csv('test/train_mp_2.csv')
df_test_exp.to_csv('test/test_exp_2.csv')
df_test_mp.to_csv('test/test_mp_2.csv')

# generate features for both train and test split
X_merged_exp, y_merged_exp, formula_merged_exp = vf.generate_features(df_band_gap_exp)
X_merged_mp, y_merged_mp, formula_merged_mp = vf.generate_features(df_band_gap_mp)
X_train_exp, y_train_exp, formula_train_exp = vf.generate_features(df_train_exp)
X_train_mp, y_train_mp, formula_train_mp = vf.generate_features(df_train_mp)
X_test_exp, y_test_exp, formula_test_exp = vf.generate_features(df_test_exp)
X_test_mp, y_test_mp, formula_test_mp = vf.generate_features(df_test_mp)


# save the vectorized train and test data
df_exp_band_gap = pd.concat([formula_merged_exp, X_merged_exp, y_merged_exp], axis=1)
df_mp_band_gap = pd.concat([formula_merged_mp, X_merged_mp, y_merged_mp], axis=1)
df_exp_train = pd.concat([formula_train_exp, X_train_exp, y_train_exp], axis=1)
df_mp_train = pd.concat([formula_train_mp, X_train_mp, y_train_mp], axis=1)
df_exp_test = pd.concat([formula_test_exp, X_test_exp, y_test_exp], axis=1)
df_mp_test = pd.concat([formula_test_mp, X_test_mp, y_test_mp], axis=1)

df_exp_band_gap.to_csv('test/df_exp_merged_2.csv', index=False)
df_mp_band_gap.to_csv('test/df_mp_merged_2.csv', index=False)
df_exp_train.to_csv('test/df_exp_train_2.csv', index=False)
df_mp_train.to_csv('test/df_mp_train_2.csv', index=False)
df_exp_test.to_csv('test/df_exp_test_2.csv', index=False)
df_mp_test.to_csv('test/df_mp_test_2.csv', index=False)

