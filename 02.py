import pandas as pd

dataset = pd.read_csv('02.csv', encoding = 'latin-1')
dataset.head()
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
dataset.head()
dataset.rename(columns = {'v1': 'status', 'v2': 'message'}, inplace = True)
dataset.head()
dataset['status'].value_counts()