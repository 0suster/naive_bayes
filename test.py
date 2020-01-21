import pandas as pd

data = pd.read_csv('data_sets/car_ucna.csv')
data.head()
clss = data.columns[len(data.columns) - 1]

a = data.loc[data[clss] == 'good']
b = a.loc[a['buying'] == 'high']

print(data.loc[data[clss] == "acc"].shape[0])
