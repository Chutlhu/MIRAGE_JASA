import os.path
from termcolor import colored
import numpy  as np
import pickle as pkl
import scipy  as sp
import scipy.io
import matplotlib.pyplot as plt
import soundfile as sf

from my_utils import load_from_pickle, save_to_pickle, load_matfile
from signal_processing_tools import gcc, cc, awgn

DATA_DIR = './data/'
obs_save_as = DATA_DIR + 'plot_data.pkl'
if not os.path.isfile(obs_save_as):
    rirs = { x : load_matfile(DATA_DIR + x + '_rirs.mat', var = x + '_rirs')[0:10,:,:]
            for x in ['clean', 'noisy']}
    annotation = load_matfile(DATA_DIR + 'annotations_vars.mat')

    for key in rirs:
        print(key, rirs[key].shape)

    save_to_pickle(rirs, obs_save_as)
else:
    rirs = load_from_pickle(obs_save_as)

n_obs = rirs['clean'].shape[0]
print('Number of RIRs', n_obs)

d = 0.1
angle_resultion = 1
angle_grid = np.linspace(0, 180, num = 180/angle_resultion + 1)[::-1]
tau_grid_physic = d*np.cos(np.deg2rad(angle_grid))/343
max_tau_physic = tau_grid_physic[-1]
min_tau_physic = tau_grid_physic[0]

max_tau = 1e-3
min_tau = -max_tau
tau_step  = 0.001e-3
tau_grid = np.linspace(min_tau,max_tau,
                       num = (max_tau-min_tau)/tau_step + 1)

f, axarr = plt.subplots(6,3, sharex=True, sharey=True)

for s, source in enumerate(['impulse', 'white_noise', 'speech']):
    for f, filtr in enumerate(['syn_1k', 'schl_cln', 'schl_rev']):
        for r, snr in enumerate([15, np.infty]):
            for c, correlation in enumerate(['cc', 'gcc', 'gcc_phat']):

                n = 2

                ## Source signal
                if source is 'impulse':
                    src = 1
                elif source is 'white_noise':
                    src = np.random.randn(int(1.024*16000))
                    src = src - np.mean(src)
                    src = 0.95*src/np.max(np.abs(src))
                elif source is 'speech':
                    src, fs = sf.read('./data/wav/100.wav')
                    src = src - np.mean(src)
                    src = 0.95*src/np.max(np.abs(src))

                else:
                    pass

                ## Filter signal
                if filtr is 'schl_cln':
                    h1 = rirs['clean'][n,0,:]
                    h2 = rirs['clean'][n,1,:]
                elif filtr is 'schl_rev':
                    h1 = rirs['noisy'][n,0,:]
                    h2 = rirs['noisy'][n,1,:]
                elif filtr is 'syn_1k':
                    h1 = rirs['clean'][n,0,:].copy()
                    h2 = rirs['clean'][n,1,:].copy()
                    extra_pos  = np.random.random_integers(200,400,size=[2,2])
                    extra_coef = 0.2*np.random.rand(2,2)
                    h1[extra_pos[0,:]] = extra_coef[0,:]
                    h2[extra_pos[1,:]] = extra_coef[1,:]
                else:
                    pass

                ## generate microphone signal and awgn
                m1 = awgn(np.convolve(src, h1), snr)
                m2 = awgn(np.convolve(src, h2), snr)

                if correlation is 'cc':
                    corr_fun = lambda x, y : cc(x, y, tau_grid = tau_grid, normalize = True)
                elif correlation is 'gcc':
                    corr_fun = lambda x, y : gcc(x, y, tau_grid, normalize = True)
                elif correlation is 'gcc_phat':
                    corr_fun = lambda x, y : gcc(x, y, tau_grid, normalize = True, mode = 'phat')
                else:
                     pass

                gcc_features = corr_fun(m1, m2)

                ## the actual plots
                axarr[f*2+r,s].plot(1e4*tau_grid, gcc_features)
                # axarr[f*2+r,s].axvline(1e4*max_tau_physic, c = 'r', ls = '--')
                # axarr[f*2+r,s].axvline(1e4*min_tau_physic, c = 'r', ls = '--')
                # axarr[f*2+r,s].set_title('Signal: ' + source \
                #                      + '\nFilter: ' + filtr \
                #                      + '\nAWGN: ' + str(snr))
                if s == 0:
                    axarr[f*2+r,s].set_ylabel(filtr + ' + ' + str(snr) + ' dB')
                if f*2+r == 5:
                    axarr[f*2+r,s].set_xlabel(source)

            axarr[f*2+r,s].legend(['cc', 'gcc', 'gcc_phat'])
plt.tight_layout()
plt.show()
