# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 21:17:49 2023

@author: shiuli Subhra Ghosh
"""

import numpy as np
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement


mem = Memory("./mycache")

@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]

X, y = get_data("a9a.txt")
X_test, y_test = get_data("a9a_t.txt")
X = X[:,:122].toarray()
X_test = X_test.toarray()


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)


def sgdmethod(X, Y, subgradloss, subgradreg, regparam, w1, T, a, m):
    out = []
    w,b = w1
    out.append((w,b))
    subgradloss_w = np.zeros([X.shape[1]])
    subgradloss_b = 0
    loss = 0
    objfun = np.zeros(T)
    p,n = X.shape
    for t in range(0, T):
        sample_index = sample_without_replacement(p,m) ## Sampling the random index from minibatch 
        learning_rate = 1/(1 + a * t)                  ## Defining the learning rate
        x = X[sample_index,:] 
        y = Y[sample_index]
        for i in range(m):
            loss_i, sub_w , sub_b = subgradloss(x[i,:], y[i], (w,b))  ## calculating subgrad for a single sample given parameters w
            loss += loss_i
            subgradloss_w += sub_w
            subgradloss_b += sub_b
        reg, subgrad_w, subgrad_b = subgradreg(regparam, (w,b)) ## Adding the regularizer term 
        objfun[t] = hingelossobj(x, y, (w,b), regparam)
        w = w - learning_rate * (subgradloss_w / m + subgrad_w)
        b = b - learning_rate * (subgrad_b /m + subgrad_b)
        out.append((w,b))
    return(out,objfun)

def subgradloss(x, y, w):
    W,b = w
    f_x = np.dot(x,W) + b
    loss_val = 1 - y * f_x
    if loss_val >= 0:
        subgrad_w = - y * x
        subgrad_b = -1 * y
    else:
        loss_val = 0 
        subgrad_w = 0 * x
        subgrad_b = 0
    return((loss_val, subgrad_w, subgrad_b))

def subgradreg(regparam, w):
    W,b = w
    reg = regparam / 2 * np.dot(W,W)
    subgrad_w_reg = regparam * W
    subgrad_b_reg = 0
    return(reg,  subgrad_w_reg, subgrad_b_reg)

def hingelossobj(x, y, w, lam):
    W, b = w
    b = np.ones(x.shape[0])*b
    dis = 1 - y * (np.dot(x, W) + b)
    dis[dis < 0] = 0 
    objfun = lam/2 * np.dot(W, W) + np.sum(dis) / x.shape[0]
    return objfun


n_train = X.shape[0]
regparam = 1/n_train
w = np.ones((X.shape[1]))
b = 1
w1 = (w,b)
T = 1000

import matplotlib.pyplot as plt

# parameter governing step size
a = 1  
m = n_train
# solve sgdmethod
w, objfun = sgdmethod(X, y, subgradloss, subgradreg, regparam, w1, T, a, m)
