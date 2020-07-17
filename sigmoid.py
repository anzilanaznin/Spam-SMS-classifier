# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:06:08 2019

@author: abinash boruah
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-8,8,1)
y = sigmoid(x)

plt.plot(x,y)