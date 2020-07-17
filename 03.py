import pandas as pd
import numpy as np

dataset = pd.read_csv('03.csv', encoding = 'latin-1')
dataset.head()
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
dataset.head()
dataset.rename(columns = {'v1': 'status', 'v2': 'message'}, inplace = True)
dataset.head()
dataset['status'].value_counts()
dataset['status'] = dataset['status'].map({'ham': 0, 'spam': 1})
dataset.head()
totalDatas = 4825 + 747
trainIndex, testIndex = list(), list()
for i in range(dataset.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = dataset.loc[trainIndex]
testData = dataset.loc[testIndex]
trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
trainData.head()
trainData['status'].value_counts()
testData['status'].value_counts()