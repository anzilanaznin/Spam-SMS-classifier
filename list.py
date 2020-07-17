# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:28:46 2019

@author: abinash boruah
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([3,8,9,13,3,6,11,21,1,16])
y = np.array([30,57,64,72,36,43,59,90,20,83])

numer =sum((np.mean(x)-x)*(np.mean(y)-(y)))
deno =  sum((np.mean(x)-x)**2)

w1 = numer/deno

wo = np.mean(y)-(np.mean(x)*w1)

x_new = 10
y_new = wo+(w1*x_new)

 
 x1 = min(x)
 y1 = min(y)
 x2 = max(x)
 y2 = max(y)
 X = [x1,x2]
 Y = [y1,y2]
 plt.plot(X,Y) 
 
