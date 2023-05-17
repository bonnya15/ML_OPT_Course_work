# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:41:49 2023

@author: shiuli Subhra Ghosh
"""


import numpy as np
import matplotlib.pyplot as plt

def relu(x):
	return(max(0.0, x))


def s(x):
    return(relu(x + 1) - relu(x))

x = np.arange(-2,1,0.2)

y = np.zeros(x.shape).reshape(-1)

for i in range(len(x)):
    y[i] = s(x[i])
    
plt.plot(x,y)
plt.show()


def s_2(x):
    return(relu(relu(x+1) - 2 * relu(x)))

x = np.arange(-2,2,0.2)

y = np.zeros(x.shape).reshape(-1)

for i in range(len(x)):
    y[i] = s_2(x[i])
    
plt.plot(x,y)
plt.show()


def both(x):
    s1 = s(x)
    s2 = s(-x)
    return(relu(s1 + s2))


x = np.arange(-2,2,0.2)

y = np.zeros(x.shape).reshape(-1)

for i in range(len(x)):
    y[i] = both(x[i])
    
plt.plot(x,y)
plt.show()

