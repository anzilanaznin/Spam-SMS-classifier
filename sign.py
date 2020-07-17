# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:06:08 2019

@author: abinash boruah
"""

import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    l=[]
    for i in x:
        if i>0:
            l1=1
            l.append(l1)
        else:
            l2=0
            l.append(l2)
    return l
x = np.arange(-8,8,1)
y = sign(x)

plt.plot(x,y)
