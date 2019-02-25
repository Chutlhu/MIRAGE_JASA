import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.autograd import Variable, Function

from my_signal_processing_tools import awgn, cc, gcc
from my_utils import load_from_pickle, save_to_pickle, load_matfile

DATA_DIR = './data/'

def create_tau_grid(max_tau, tau_res):
    # Create grid of time delays (tau [ms])
    max_tau = 2e-3
    min_tau = -max_tau
    tau_res  = 0.01e-3
    tau_grid = np.linspace(min_tau,max_tau,
                        num = (max_tau-min_tau)/tau_res + 1)
    n_taus = len(tau_grid)
    print('TAU GRID:',n_taus, 'samples')
    return tau_grid, n_taus

def load_or_generate(filename, params):
    n_obs_max = params['n_obs_max']
    print('Searching for', filename)
    if os.path.isfile(filename):
        print('Loading', filename)
        gcc_features = load_from_pickle(filename)
        n_obs = min(n_obs_max, gcc_features['noisy'].shape[0])
        print('Number of RIRs', n_obs)

    # if filename does not exist, create it
    else:
        print('No dataset stored, creating it.')
        rirs = { x : load_matfile(DATA_DIR + x + '_rirs.mat', var = x + '_rirs')
                for x in ['clean', 'noisy']}
        annotation = load_matfile(DATA_DIR + 'annotations_vars.mat')
        n_obs = min(n_obs_max, rirs['clean'].shape[0])
        print('Number of RIRs', n_obs)

        # create the dataset of observatio
        tau_grid, n_taus = create_tau_grid(params['max_tau'], params['tau_res'])

        gcc_features = { x : np.zeros([n_obs, n_taus])
                        for x in ['clean', 'noisy']}

        for n in range(n_obs):
            for data in ['noisy', 'clean']:

                # Source signal
                if params['source_signal'][data] is 'noise':
                    s = np.random.randn(int(1.024*FS))
                    s = s - np.mean(s)
                    s = 0.95*s/np.max(np.abs(s))
                elif params['source_signal'][data] is 'impulse':
                    s = 1
                elif params['source_signal'][data] is 'speech':
                    s, fs = sf.read('./data/wav/100.wav')
                    s = s - np.mean(s)
                    s = 0.95*s/np.max(np.abs(s))
                else:
                    break

                ## Filter signal
                if params['filter_signal'][data] is 'schimmel_clean':
                    h1 = rirs['clean'][n,0,:]
                    h2 = rirs['clean'][n,1,:]
                elif params['filter_signal'][data] is 'schimmel_reverb':
                    h1 = rirs['noisy'][n,0,:]
                    h2 = rirs['noisy'][n,1,:]
                elif params['filter_signal'][data] is 'sythetic_echoes2':
                    h1 = rirs['clean'][n,0,:].copy()
                    h2 = rirs['clean'][n,1,:].copy()
                    extra_pos  = np.random.random_integers(200,400,size=[2,2])
                    extra_coef = 0.2*np.random.rand(2,2)
                    h1[extra_pos[0,:]] = extra_coef[0,:]
                    h2[extra_pos[1,:]] = extra_coef[1,:]
                else:
                    break

                m1 = awgn(np.convolve(s, h1), params['awgn'][data])
                m2 = awgn(np.convolve(s, h2), params['awgn'][data])

                if params['correlation'][data] is 'cc':
                    corr_fun = lambda x, y : cc(x, y, tau_grid = tau_grid, normalize = True)
                elif params['correlation'][data] is 'gcc':
                    corr_fun = lambda x, y : gcc(x, y, tau_grid, normalize = True)
                elif params['correlation'][data] is 'gcc_phat':
                    corr_fun = lambda x, y : gcc(x, y, tau_grid, normalize = True, mode = 'phat')
                else:
                    break

                gcc_features[data][n,:] = corr_fun(m1, m2)

                if (data is 'noisy') and ((n % 20 == 0) or (n == n_obs - 1)):
                    print('Computing GGC features:', int(100*(n+1)/n_obs), '%')

        save_to_pickle(gcc_features, filename)

    dataset = gcc_features
    return dataset

def dataset2pytorch(samples, targets, params):
    n_obs = samples.shape[0]
    ## DATA TO PYTORCH DATASET
    n_test  = int(np.ceil(n_obs*params['p_test']/100))
    n_train = int(n_obs - n_test)
    n_valid = int(np.ceil(n_train*params['p_valid']/100))
    n_overf = int(np.ceil(n_train*params['p_overf']/100))
    print('n_obs, n_train, n_valid, n_overf, n_test: ',n_obs, n_train, n_valid, n_overf, n_test)

    random_indeces = np.random.permutation(n_obs)
    random_subindeces = np.random.permutation(n_train-n_valid)
    indeces = { 'Train' : random_indeces[0:n_train-n_valid],
                'Valid' : random_indeces[n_train-n_valid:n_train],
                'Overf' : random_indeces[random_subindeces[0:n_overf]],
                'Test ' : random_indeces[n_train:n_obs]}
    n_train = indeces['Train'].shape[0]
    phases = indeces.keys()
    # Training set
    return { x : { 'samples' : Variable(torch.from_numpy(samples[indeces[x],:])).float(),
                    'targets' : Variable(torch.from_numpy(targets[indeces[x],:])).float()
                      }
                for x in phases}

def get_torch_dataset(filename, params):
    dataset = load_or_generate(filename, params)
    return dataset2pytorch(dataset['noisy'], dataset['clean'], params)
