import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt

nfft = 1024
hop  = 512

def stft(x):
    return sp.signal.stft(x, fs = 16000, nperseg = nfft, noverlap = hop)

def gcc(x, y, tau_grid, mode = None, weights = None,
        time_pooling = 'max', freq_pooling = 'sum', normalize = False):

    n_taus = len(tau_grid)

    f, t, X = stft(x)
    f, t, Y = stft(y)

    F, T = X.shape

    P = X*np.conj(Y)

    if mode is 'phat':
        P = P/np.abs(P)
    if mode is 'weights':
        P = P/weights

    spec = np.zeros([F,T,n_taus])

    for idx, tau in enumerate(tau_grid):
        exp = np.exp(-1j*2*np.pi*tau*f)[:,None]
        spec[:,:,idx] = np.real(P)*np.real(exp) - np.imag(P)*np.imag(exp)

    # aggreate frequencies
    if freq_pooling is 'sum':
        spec = np.sum(spec, axis = 0)

    # Pooling: aggreate times
    if time_pooling is 'max':
        spec = np.max(spec, axis = 0)
    if time_pooling is 'sum':
        spec = np.max(spec, axis = 0)

    # normalize if requested
    if normalize:
        spec = spec/np.max(np.abs(spec))

    return spec
