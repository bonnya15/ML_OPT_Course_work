# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 23:15:12 2023

@author: shiuli Subhra Ghosh
"""

import time
from math import sqrt
import numpy as np
from scipy import linalg

rng = np.random.RandomState(42)
m, n = 15, 20

# random design
A = rng.randn(m, n)  # random design

x0 = rng.rand(n)
x0[x0 < 0.9] = 0
b = np.dot(A, x0)
l = 0.5  # regularization parameter


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A) ** 2  # Lipschitz constant
    time0 = time.time()
    for _ in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A) ** 2
    time0 = time.time()
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times

def softShrink(x0, alpha):
    x0 = x0.reshape(len(x0),1)
    x = np.zeros((len(x0),1))
    x[x0 > alpha] = x0[x0 > alpha] - alpha
    x[x0 < -alpha] = x0[x0 < -alpha] + alpha
    return(x)

def istaLasso(A, b, lambda_reg, x0, alpha, T):
    objhist = [f(A,x0,b)+g(lambda_reg,x0)]
    X = [x0]
    for t in range(T):
        x0_softShrink = x0 - alpha*subgrad(A,x0,b);
        alpha_softShrink = alpha*lambda_reg;
        x = softShrink(x0_softShrink, alpha_softShrink).reshape(-1);
        X.append(x)
        objhist.append(f(A,x0,b)+g(lambda_reg,x0))
        x0 = x
    xT = X[-1]
    return(xT, objhist)

def f(A,x,b):
    return(0.5 * np.linalg.norm(A @ x - b)** 2)
def g(lambda_reg, x):
    return(lambda_reg * np.linalg.norm(x,1))
def subgrad(A,x,b):
    m,n = A.shape
    ols_term = A.T @ ( A @ x - b)
    return(ols_term)



# =============================================================================
maxit = 3000
T = 3000
lambda_reg = 0.5
#alpha = 1 / (linalg.norm(A) ** 2) 
alpha = 0.003
A = rng.randn(m, n)  # random design
xo = rng.rand(n)
xo[xo < 0.9] = 0
b = np.dot(A, x0)
x0 = np.zeros(A.shape[1])
# 
# =============================================================================
xT, objhist = istaLasso(A, b, lambda_reg, x0, alpha, T)



x_ista, pobj_ista, times_ista = ista(A, b, l, maxit)

x_fista, pobj_fista, times_fista = fista(A, b, l, maxit)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure()
plt.stem(x0, markerfmt='go')
plt.stem(x_ista, markerfmt='bo')
plt.stem(x_fista, markerfmt='ro')

plt.figure()
plt.plot(times_ista, pobj_ista, label='ista')
plt.plot(times_fista, pobj_fista, label='fista')
plt.xlabel('Time')
plt.ylabel('Primal')
plt.legend()
plt.show()