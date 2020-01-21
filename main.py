"""
https://www.youtube.com/watch?v=CPqOCI0ahss&t=496s
"""
from pprint import *
import pandas as pd

data = pd.read_csv('data_sets/car_ucna.csv')
data.head()
clss = data.columns[len(data.columns) - 1]
"""
:var data: data-frame iz csv datoteke
:var clss: ime stolpca kjer naj bi bil class
"""

data.sort_values(by=[clss], inplace=True)
col_vals = {}
total = data.shape[0]
cols = data.columns
"""
:var col_vals: slovar, kjer so kljuci imena stolpcev data-frama, vrednosti pa arrayi vseh edinstvenih
    vrednosti tistega stolpca
:var total: stevilo vrstic
:var cols: imena vseh stolpcev
"""

for item in data.columns:
    col_vals[item] = data[item].unique()

model = {}
"""
:var model: slovar, kjer bodo prestete vse vrednosti vsakega atributa glede na klasifikacijo
"""

for val in col_vals[clss]:
    model[val] = {"count": (data[clss] == val).sum()}
    for col in cols:
        if col != clss:
            model[val][col] = {}

classifier = col_vals[clss]
"""
:var classifier = je array vseh edinstvenih vrednosti klasifikacijskega stolpca
"""


col_vals.pop(clss)

for cl in classifier:
    for key in col_vals.keys():
        for val in col_vals[key]:
            sort = data.loc[data[clss] == cl]
            sort1 = sort.loc[sort[key] == val]
            model[cl][key][val] = (sort1.shape[0]) / (model[cl]['count'])



