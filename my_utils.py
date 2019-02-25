import numpy as np
import pickle as pkl
import h5py
import hdf5storage
def save_to_pickle(obj, filename ):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def load_matfile(filename, var = None):
    print('Loading: ' + filename + ' ...')
    mat = hdf5storage.loadmat(filename)
    if not var is None:
        mat = mat[var]
    print('done')
    return mat
    # with h5py.File(filename, 'r') as file:
    #     if not var is None:
    #         a = list(file[var])
    #     else:
    #         a = list(file)
    #     print('done.')
    #     return a
