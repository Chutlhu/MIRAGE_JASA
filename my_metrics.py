import numpy as np

def distance(x,x_ref,fun):
    L = x.shape[0]
    error = np.zeros(L)
    for l in range(L):
        error[l] = fun(x[l,:], x_ref[l,:])
    return np.mean(error)

def nrmse(x, x_ref):
    return np.sqrt(np.sum((x - x_ref)**2, axis = 0))\
           /np.sqrt(np.sum((x_ref - np.mean(x_ref, axis = 0))**2, axis = 0))
