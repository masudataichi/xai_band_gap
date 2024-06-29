import pymatgen as mg
import pandas as pd
import pymatgen.core
from pymatgen.ext.matproj import MPRester


print(pymatgen.core.__version__)

df_train = pd.read_csv("../dataset/df_exp_merged.csv")

formulaes = df_train.iloc[:, 0]

for formula in formulaes:
    fractional_composition = mg.core.composition.Composition(formula).fractional_composition.as_dict()
    element_composition = mg.core.composition.Composition(formula).element_composition.as_dict()
    print("element_composition")
    print(element_composition)