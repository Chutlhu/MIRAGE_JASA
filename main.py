### CODE FOR
##  MIRAGE: 2D Source Localization using Microphone Array Augmentation with Echoes
#   By Diego Di Carlo, Antoine Deleforge and Nancy Bertin
#   Read for @ JAS 2019
#

import os.path
from termcolor import colored
import numpy  as np
import pickle as pkl
import scipy  as sp
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import hashlib
import json
import copy

from my_utils import load_from_pickle, save_to_pickle, load_matfile
from my_metrics import distance, nrmse
from my_signal_processing_tools import gcc, cc
from my_torch_utils import EarlyStopping, WassersteinLossStab
from my_network import AutoEncoder
from my_dataset import get_torch_dataset

## DIRECTORIES
RESULTS_DIR = "./results/"
DATA_DIR  = "./data/"
SAVED_MODELS = './saved_models'

# DATASET CONSTANT
params_data = {
    'n_obs_max' : 500,
    'FS' : 16000,
    'max_tau' : 2e-3,
    'tau_res' : 1e-5,
    'source_signal' : {'samples' : 'noise', # 'impulse', 'noise', 'speech'
                       'targets' : 'noise'},
    'filter_signal' : {'samples' : 'schimmel_reverb', # 'schimmel_clean', 'schimmel_reverb', 'sythetic_echoes2'
                       'targets' : 'schimmel_reverb'},
    'awgn' : {'samples' : 0,
              'targets' : 30},
    'features' : ['ILD', 'rIPD', 'iILD'],
    'fft_bins' : 513,
    'p_test'   : 10,  # dimension of test set
    'p_valid'  : 10,
    'p_overf'  : 50
    }
# NETWORK HYPER-PARAMS
params_dnn = {
    'n_epochs' : 10000,  # or whatever
    'batch_size' : 300, # or whatever
    'patience' : 100,
    'n_classes': 100,
    'n_hidden_in': 1000,
    'n_hidden_out': 50,
    'n_hidden_ratio':  4,
    'dropout'       : 0.3
    'loss_type'     : 'wasserstein', # or 'MSE'
    'accuracy_type' : 'wasserstein', # 'euclidean', 'wasserstein', 'correlation', 'cosine'
    'do_l1_regularization' : True,
    'l1_regularization_patience' : 100,
    'l1_regularization_lambda' : 0.1,
    }

data_suffix = hashlib.sha1(json.dumps(params_data, sort_keys=True).encode('utf-8')).hexdigest()
filename   = './data/obs_' + data_suffix
datasets = get_torch_dataset(filename + '.pkl', params_data)
params = {**params_data, **params_dnn}
exp_suffix = hashlib.sha1(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

phases = datasets.keys()
for key in phases:
    print(key, datasets[key].keys())
print('done.\n')

n_train, n_feat = datasets['Train']['samples'].shape

# define the network
model = AutoEncoder(input_size = n_feat,
                    layer_size = [n_feat//4, n_feat//8, n_feat//16])
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
stopping  = EarlyStopping(patience = params['patience'])
loss_func_sparse = torch.nn.L1Loss()  # this is for regression mean squared loss
Lambda = params['l1_regularization_lambda']
def get_distance_function(n):
    x = np.arange(n)
    M = (x[:,None]- x[None,:])**2
    return M / M.max()
M = get_distance_function(n_feat)
if params['loss_type'] is 'wasserstein':
    loss_func = WassersteinLossStab(cost = torch.from_numpy(M).float())
if params['loss_type'] is 'MSE':
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

batch_size = params['batch_size']
if batch_size < n_train:
    indeces = [(i,i+batch_size) for i in range(n_train) if i%batch_size==0 and i < n_train]
    indeces[-1] = (indeces[-1][0], n_train)
else:
    indeces = [(0, n_train)]

performance = { x : { 'error' : [],
                      'loss'  : [],
                      'sample': [],
                      'target': [],
                      'output': [],
                      'l1_kickin' : 0}
               for x in phases}
results = {'best_err' : np.infty,
           'best_model' : None,
           'best_epoch' : None,}
distances = { 'wasserstein' : lambda pred, ref : sp.stats.wasserstein_distance(pred, ref),
              'euclidean'   : lambda pred, ref : sp.spatial.distance.euclidean(pred, ref),
              'cosine'      : lambda pred, ref : sp.spatial.distance.cosine(pred, ref),
              'correlation' : lambda pred, ref : sp.spatial.distance.correlation(pred, ref),
            }

converged = False
add_l1_regularization = False
print_colored_if = lambda cond_text : colored(cond_text, 'green' if cond_text else 'red')

## TRAINING
n_epochs = params['n_epochs']
for it in range(n_epochs):

    if converged:
        break

    for phase in ['Train', 'Valid', 'Overf']:

        if phase is 'Train':
            model.train()
            batch_indeces = indeces
        else:
            model.eval()
            batch_indeces = [(0, 10000)]

        # Batch learning for training
        for i, (start, end) in enumerate(batch_indeces):
            print(phase, i, (start, end))
            samples = datasets[phase]['samples'][start:end,:]
            targets = datasets[phase]['targets'][start:end,:]
            outputs = model(samples)     # Batch x Taus
            outputs = outputs*samples

            loss = loss_func(outputs+1, targets+1)     # must be (1. nn output, 2. target)
            if params['do_l1_regularization'] and add_l1_regularization:
                loss_spr = loss_func_sparse(outputs, torch.tensor(0.))
                loss += Lambda*loss_spr

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            # error/accuracy (wasserstein distance)
            errors = { x : distance(outputs.data.numpy(),
                                   targets.data.numpy(),
                                   distances[x]) for x in distances.keys()}
            epoch_err = errors[params['accuracy_type']]

            # Track the accuracy
            if (i) % 20 == 0 or i == len(batch_indeces) - 1:
                print('{} >> Epoch [{:03d}/{:03d}], Batch [{:02d}/{:02d}], Loss: {:.4f}, Error: {:.2f}' \
                      .format(phase, it + 1, n_epochs, i+1, len(batch_indeces), loss.item(), epoch_err))


            if phase == "Valid":
                converged = stopping.step(epoch_err)
                print(print_colored_if(converged), print_colored_if(add_l1_regularization))
                if converged and params['do_l1_regularization'] and not add_l1_regularization:
                    print('Start traking L1 regularization')
                    performance[phase]['l1_kickin'] = it
                    converged = False
                    add_l1_regularization = True
                    stopping  = EarlyStopping(patience=params['patience']//2)

                if epoch_err < results['best_err']:
                    print(" ")
                    results['best_err'] = epoch_err
                    results['best_model'] = copy.deepcopy(model.state_dict())
                    results['best_epoch'] = it
                    print("--- Best error: ", results['best_err'])
                    print(" ")


        performance[phase]['error'].append(errors)
        performance[phase]['loss' ].append(loss.item())
        performance[phase]['sample'].append(samples[0,:].data.numpy())
        performance[phase]['target'].append(targets[0,:].data.numpy())
        performance[phase]['output'].append(outputs[0,:].data.numpy())

print('training ends.\n')

### !!! TEST !!! ###
phase = 'Test '
# load best model weights
model.load_state_dict(results['best_model'])
# load test data
samples = datasets[phase]['samples']
targets = datasets[phase]['targets']
# predict
outputs = model(samples)
# performance metrics
errors = { x : distance(outputs.data.numpy(),
                       targets.data.numpy(),
                       distances[x]) for x in distances.keys()}
performance[phase]['error'].append(errors)
performance[phase]['sample'].append(samples[0,:].data.numpy())
performance[phase]['target'].append(targets[0,:].data.numpy())
performance[phase]['output'].append(outputs[0,:].data.numpy())

### Save results
exp_obj = {
    'results' :     results,
    'performance' : performance,
}
save_to_pickle(exp_obj, './results/results_' + exp_suffix + '.pkl')
