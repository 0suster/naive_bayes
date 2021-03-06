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
        norm = 1
        for col in col_vals:
            value = row[col]
            in_prob *= model[clss][col][value]
            norm *= val_probs[col][value]

        select.append(in_prob/norm)


    for i in range(len(select)):
        if select[i] == max(select):
            guess.append(classifier[i])
            break

df_test['guess'] = guess

accuracy = 0
for index, row in df_test.iterrows():
    if row[dvar] == row['guess']:
        accuracy += 1


accuracy /= df_test.shape[0]
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

tf_matrix = {}
for cl in classifier:
    tf_matrix[cl] = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }

for mkey in matrix.keys():
    for key in matrix:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for skey in matrix[key]:
            val = matrix[key][skey]
            if mkey == key and key == skey:
                tp += val
            elif mkey == key and key != skey:
                fp += val
            elif mkey != key and key == skey:
                tn += val
            elif mkey != key and key != skey:
                fn += val

        tf_matrix[mkey]['tp'] += tp
        tf_matrix[mkey]['fp'] += fp
        tf_matrix[mkey]['tn'] += tn
        tf_matrix[mkey]['fn'] += fn

metrike = {
    "recall" : 0,
    "prec" : 0,
    "fscore" : 0
}

for tf in tf_matrix:
    val = tf_matrix[tf]
    recall = val['tp']/((val['tp']+val['fn']))
    prec = val['tp']/((val['tp']+val['fp']))
    fscore = 2 * (recall * prec) / ((recall+prec))
    metrike[tf] = {
        "recall" : recall,
        "prec" : prec,
        "fscore" : fscore
    }

srecall = 0
sprec = 0
sfscore = 0
for cl in model:
    srecall += (model[cl]["count"] * metrike[cl]["recall"])
    sprec += (model[cl]["count"] * metrike[cl]["prec"])
    sfscore += (model[cl]["count"] * metrike[cl]["fscore"])

metrike["recall"] = srecall
metrike["prec"] = sprec
metrike["fscore"] = sfscore



def izpis():
    #pprint(model)
    print()
    pprint(matrix)
    print()
    pprint(tf_matrix)
    pprint(metrike)

izpis()
