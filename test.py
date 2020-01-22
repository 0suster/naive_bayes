import pandas as pd

data = pd.read_csv('data_sets/car_ucna.csv')
data.head()
clss = data.columns[len(data.columns) - 1]
