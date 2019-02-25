import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import hashlib
import json

from my_utils import load_from_pickle
params_data = {
    'n_obs_max' : 500,
    'FS' : 16000,
    'max_tau' : 2e-3,
    'tau_res' : 1e-5,
    'source_signal' : {'noisy' : 'impulse', # 'impulse', 'noise', 'speech'
                       'clean' : 'impulse'},
    'filter_signal' : {'noisy' : 'schimmel_clean', # 'schimmel_clean', 'schimmel_reverb', 'sythetic_echoes2'
                       'clean' : 'schimmel_clean'},
    'awgn' : {'noisy' : 30,
              'clean' : np.infty},
    'correlation'  : { 'noisy' : 'cc',
                       'clean' : 'cc'},
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
    'loss_type'     : 'wasserstein', # or 'MSE'
    'accuracy_type' : 'wasserstein', # 'euclidean', 'wasserstein', 'correlation', 'cosine'
    'do_l1_regularization' : True,
    'l1_regularization_patience' : 100,
    'l1_regularization_lambda' : 0.1,
    }

params = {**params_data, **params_dnn}
exp_suffix = hashlib.sha1(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

filename = 'results_' + exp_suffix
f =  load_from_pickle('./results/' + filename + '.pkl')
performance = f['performance']
results = f['results']

metrics = performance['Train']['error'][0].keys()
n_metrics = len(metrics)
n_epochs = len(performance['Train']['error'])
print(metrics)

## PLOT LEARNING EVOLUTION
phases = ['Train', 'Valid', 'Overf', 'Test ']
n_phases = len(phases)

f, axarr = plt.subplots(max(n_phases, n_metrics),3, sharex=True)
for p, phase in enumerate(phases):
    if phase is 'Test ':
        continue
    for i, metric in enumerate(metrics):
        m = np.zeros(n_epochs)
        for n in range(n_epochs):
            m[n] = performance[phase]['error'][n][metric]
        m /= np.max(np.abs(m))
        axarr[p,0].plot(m, alpha = 0.7)
        axarr[p,1].plot(np.gradient(m), alpha = 0.7)

        axarr[i,2].plot(m, alpha = 0.7)
        axarr[i,2].legend(phases)
        axarr[i,2].axvline(performance['Valid']['l1_kickin'], c = 'r', ls = '--')
        axarr[i,2].set_title(metric)
        axarr[i,2].set_xlabel('Metrics')
        axarr[i,2].set_ylabel('Error / Distance')

    axarr[p,0].axvline(performance['Valid']['l1_kickin'], c = 'r', ls = '--')
    axarr[p,1].axvline(performance['Valid']['l1_kickin'], c = 'r', ls = '--')
    axarr[p,0].legend(metrics)
    axarr[p,1].legend(metrics)
    axarr[p,0].set_title(phase)
    axarr[p,1].set_title(phase)
    axarr[p,0].set_xlabel('Epochs')
    axarr[p,1].set_xlabel('Epochs')
    axarr[p,0].set_ylabel('Error / Distance')
    axarr[p,1].set_ylabel('Error / Distance')
plt.show()

frames = []
folder = './results/video/'
c = 1
step = max(n_epochs//100,1)
for ep in range(n_epochs):
    if ep%step == 0 or ep == n_epochs-1:
        phase = 'Valid'
        plt.plot(performance[phase]['sample'][ep], alpha = 0.6, ls = '-.')
        plt.plot(performance[phase]['output'][ep], alpha = 0.6, ls = '-')
        plt.plot(performance[phase]['target'][ep], alpha = 0.6, ls = '--')
        plt.legend(['samples', 'outputs','targets'])
        plt.title(str(ep))
        plt.savefig(folder + "file%03d.png" % c)
        plt.close()
        if ep > performance['Valid']['l1_kickin']:
            ax = plt.gca()
            ax.set_facecolor('xkcd:black')
        c += 1
        print('Saving: ', int(100*ep/n_epochs), '%')

subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', folder + 'file%03d.png',
    'video_' + filename + '.mp4'
])
for file_name in glob.glob(folder + "*.png"):
    os.remove(file_name)
