"""
https://www.youtube.com/watch?v=CPqOCI0ahss&t=496s
"""
import pandas as pd
from pprint import pprint

df_learn = pd.read_csv('data_sets/car_ucna.csv')
dvar = df_learn.columns[len(df_learn.columns) - 1]
"""
:var df_learn: data-frame iz csv datoteke
:var dvar: ime stolpca kjer naj bi bil class
"""

df_learn.sort_values(by=[dvar], inplace=True)
col_vals = {}
total = df_learn.shape[0]
cols = df_learn.columns
"""
:var col_vals: slovar, kjer so kljuci imena stolpcev data-frama, vrednosti pa arrayi vseh edinstvenih
    vrednosti tistega stolpca
:var total: stevilo vrstic
:var cols: imena vseh stolpcev
"""

for item in df_learn.columns:
    col_vals[item] = df_learn[item].unique()

model = {}
"""
:var model: slovar, kjer bodo prestete vse vrednosti vsakega atributa glede na klasifikacijo
"""

for val in col_vals[dvar]:
    model[val] = {"count": (df_learn[dvar] == val).sum()}
    for col in cols:
        if col != dvar:
            model[val][col] = {}

classifier = col_vals[dvar]
"""
:var classifier = je array vseh edinstvenih vrednosti klasifikacijskega stolpca
"""

col_vals.pop(dvar)

for cl in classifier:
    for key in col_vals.keys():
        for val in col_vals[key]:
            sort = df_learn.loc[df_learn[dvar] == cl]
            sort1 = sort.loc[sort[key] == val]
            model[cl][key][val] = (sort1.shape[0]) / (model[cl]['count'])

val_probs = {}
"""
:var val_probs: splosna verjetnost za vsako mozno vrednost v vaskem stolpcu
"""

for col in col_vals:
    sums = df_learn[col].value_counts()
    val_probs[col] = {}
    for index in sums.index:
        val_probs[col][index] = sums[index] / total

df_test = pd.read_csv('data_sets/car_testna.csv')
comp_class = df_test[dvar]
"""
:var df_test: dataframe testnega csv-ja
:var comp_class: klasifikacije testnega data-seta pred napovedjo
"""

for key in model.keys():
    model[key]['count'] /= total

guess = []
for index, row in df_test.iterrows():
    select = []
    for clss in classifier:
        in_prob = 1
        for col in col_vals:
            value = row[col]
            in_prob *= model[clss][col][value]
        select.append(in_prob)

    for i in range(len(select)):
        if select[i] == max(select):
            guess.append(classifier[i])
            break

df_test['guess'] = guess

tocnost = 0
for index, row in df_test.iterrows():
    if row[dvar] == row['guess']:
        tocnost+=1

tocnost /= df_test.shape[0]

matrix = {}
for cl in classifier:
    matrix[cl] = {}
    for cl1 in classifier:
        matrix[cl][cl1] = 0

for index, row in df_test.iterrows():
    real = row[dvar]
    pred = row['guess']
    if real == pred:
        matrix[real][real] += 1
    else:
        matrix[real][pred] += 1

pprint(matrix)

