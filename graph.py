import matplotlib.pyplot as plt 
 
x = ['Precision','Recall','F-Score','Accuracy'] 
y = [0.8782051282051282, 0.7025641025641025, 0.7806267806267807,0.9446043165467626]
plt.plot(x,y,label = 'TF-IDF')

x1 = ['Precision','Recall','F-Score','Accuracy']
y1 = [0.8702290076335878,0.5846153846153846,0.7181208053691275, 0.9386861313868613]
plt.plot(x1,y1,label = 'Bag of Words') 
plt.xlabel('x - axis') 
plt.ylabel('y - axis')  
plt.title('Method Comparision')   
plt.legend()
plt.show() 

