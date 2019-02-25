import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt

Fs = 16000
nfft = 1024
hop  = 512

def stft(x):
    return sp.signal.stft(x, fs = Fs, nperseg = nfft, noverlap = hop, nfft = 2*nfft)

def istft(x):
    return sp.signal.istft(x, fs = Fs, nperseg = nfft, noverlap = hop, nfft = 2*nfft)

def cc(x, y, tau_grid = None, Fs = Fs, normalize = False):
    r = np.correlate(y, x, "full")
    if normalize:
        r = r/np.max(np.abs(r))
    if tau_grid is None:
        max_lag = int((len(r)-1)/2)
        lag = np.concatenate((-np.arange(1,max_lag+1)[::-1], np.arange(max_lag+1))) / float(Fs)
    else:
        max_lag = int(tau_grid[-1]*Fs)
        min_lag = int(-tau_grid[0]*Fs)
        if tau_grid[0]*Fs > 0:
            print('Not Implemented') #TODO implement negative min lag
        lag = np.concatenate((-np.arange(1,min_lag+1)[::-1], np.arange(max_lag+1))) / float(Fs)
        r = r[int((len(r)-1)/2)-min_lag:int((len(r)-1)/2)+1+max_lag]
        r = np.interp(tau_grid, lag, r)
        lag = tau_grid
    return r

def gcc(x, y, tau_grid, mode = None, weights = None,
        time_pooling = 'sum', freq_pooling = 'sum', normalize = False):

    n_taus = len(tau_grid)

    f, t, X = stft(x)
    f, t, Y = stft(y)

    F, T = X.shape

    P = X*np.conj(Y)

    if mode is 'phat':
        P = P/np.abs(P)
    if mode is 'weights':
        P = P*weights

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

def awgn(x, snr_db):
    L = max(x.shape)
    snr_lin = 10**(snr_db/10)   # SNR to linear scale
    Ex = np.sum(np.abs(x)**2)/L # Energy of the signal
    N = Ex/snr_lin              # find the noise variance
    n = np.sqrt(N)              # standard deviation for AWGN noise
    return x + np.random.normal(0, n, L)
