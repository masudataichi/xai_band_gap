#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:16:58 2018

@author: steven
"""
# =============================================================================
# # YOU NEED TO SET THE PATH TO MATCH THE LOCATION OF THE Ensemble FOLDER
# =============================================================================
# import sys
# ## base_path = r'location of the folder Esemble'
# base_path = r'/home/steven/Research/PhD/DFT Ensemble Models/publication code/Ensemble/'
# sys.path.insert(0, base_path)

import sys
sys.path.append("../../kaaiian/kaaiian-ensemble_band_gap_prediction-18cf8a4/Ensemble/MachineLearningFunctions")
print(sys.executable)
from MSE_ML_functions import VectorizeFormulaMinimal
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
df_band_gap_exp.to_csv('merged_data_exp.csv')
df_band_gap_mp.to_csv('merged_data_mp.csv')
df_train_exp.to_csv('train_exp.csv')
df_train_mp.to_csv('train_mp.csv')
df_test_exp.to_csv('test_exp.csv')
df_test_mp.to_csv('test_mp.csv')

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

df_exp_band_gap.to_csv('df_exp_merged.csv', index=False)
df_mp_band_gap.to_csv('df_mp_merged.csv', index=False)
df_exp_train.to_csv('df_exp_train.csv', index=False)
df_mp_train.to_csv('df_mp_train.csv', index=False)
df_exp_test.to_csv('df_exp_test.csv', index=False)
df_mp_test.to_csv('df_mp_test.csv', index=False)

