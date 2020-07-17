
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np

dataset = pd.read_csv('spam.csv', encoding = 'latin-1')
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

spam_words = ' '.join(list(dataset[dataset['status'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

ham_words = ' '.join(list(dataset[dataset['status'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()