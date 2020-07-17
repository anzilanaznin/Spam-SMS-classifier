
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log
from wordcloud import WordCloud
import pandas as pd
import numpy as np

dataset = pd.read_csv('spam.csv', encoding = 'latin-1')
dataset.head()
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
dataset.head()
dataset.rename(columns = {'v1': 'statuses', 'v2': 'message'}, inplace = True)
dataset.head()
dataset['statuses'].value_counts()
dataset['status'] = dataset['statuses'].map({'ham': 0, 'spam': 1})
dataset.drop(['statuses'], axis = 1, inplace = True)
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

testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)
testData.head()

trainData['status'].value_counts()
testData['status'].value_counts()
print("------------------------------------------")
print("SPAM WORDS")
spam_words = ' '.join(list(dataset[dataset['status'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
print("------------------------------------------")
print("HAM WORDS")
ham_words = ' '.join(list(dataset[dataset['status'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

class SpamClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.messages, self.statuses = trainData['message'], trainData['status']
        self.method = method
    
    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_BoW()

    def calc_BoW(self):
        self.prob_spam = dict()
        self.prob_ham  = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + len(list(self.tf_ham.keys())))
            
            self.prob_spam_message, self.prob_ham_message = self.spam_messages / self.total_messages, self.ham_messages / self.total_messages 

    def calc_TF_and_IDF(self):
        noOfMessages = self.messages.shape[0]
        self.spam_messages, self.ham_messages = self.statuses.value_counts()[1], self.statuses.value_counts()[0]
        self.total_messages = self.spam_messages + self.ham_messages
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.messages[i])
            count = list() #To keep track of whether the word has ocured in the message or not.
                            #For IDF
            for word in message_processed:
                if self.statuses[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.statuses[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_messages + self.ham_messages) / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_messages + self.ham_messages)/ (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
            
    
        self.prob_spam_message, self.prob_ham_message = self.spam_messages / self.total_messages, self.ham_messages / self.total_messages 
    
    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:                
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                if self.method == 'tf-idf':
                    pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                if self.method == 'tf-idf':
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))) 
                else:
                    pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_message)
            pHam += log(self.prob_ham_message)
        return pSpam >= pHam
     
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result     
    
   
    
def metrics(status, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(status)):
        true_pos += int(status[i] == 1 and predictions[i] == 1)
        true_neg += int(status[i] == 0 and predictions[i] == 0)
        false_pos += int(status[i] == 0 and predictions[i] == 1)
        false_neg += int(status[i] == 1 and predictions[i] == 0)       
    recall = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    Fscore = 2 * precision * recall / (precision + recall)
       
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)
    
print("TF-IDF")    
sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['message'])
metrics(testData['status'], preds_tf_idf)    
print("-------------------------------------") 
print("Bag of Words") 
sc_bow = SpamClassifier(trainData, 'bow')
sc_bow.train()
preds_bow = sc_bow.predict(testData['message'])
metrics(testData['status'], preds_bow)

while True:
    msg=input("Enter the message: ")
    if(msg=='q'):
        break
    pm = process_message(msg)
    result=sc_tf_idf.classify(pm)
    if(result==1):
        print("Spam")
    else:
        print("Ham")
    #print(result)

        
        
        
        
        
        
        
        
        