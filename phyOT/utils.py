import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn


def convolve_avg(array, window):
    kernel = jnp.ones(window)
    new_array = convolve(array, kernel, mode='same') / convolve(jnp.ones_like(array), kernel, mode='same')
    return new_array
    
activation_functions = {
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'silu':nn.silu,
    'elu':nn.elu,
    'sin':jnp.sin
    # Add more activation functions as needed
}

#########################################################
# Plotting Utilities
#########################################################

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'axes.labelsize':   16,
    'axes.titlesize':   16,
    'xtick.labelsize' : 12,
    'ytick.labelsize' : 12,
          })
# latex font definition
plt.rc('legend',fontsize=14)
plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
    
def array_to_batch_list(data, size_batch=128):
    '''
        function that takes in [n_data_pts, dim_data] and splits
        it into a list of batches of size size_batch with the last
        batch containg the rest of the data.
    '''
    dataset = []
    num_whole_batches = int( data.shape[0] / size_batch)
    for i in range( num_whole_batches ):
        dataset.append( data[i * size_batch: (i+1) * size_batch] )
    if data.shape[0] % size_batch != 0: 
        dataset.append(data[num_whole_batches * size_batch:])

    print('num_batches = ', len(dataset))
    n = data.shape[0] - num_whole_batches * size_batch
    if n >= 0:
        print(' items in uneven batch: ', n)
    else: print('data.shape[0] < size_batch')

    return dataset


def get_keys_and_rng(key, num=1):
    '''
        pass rng and n
        get keys and rng
    '''
    key, rng = jax.random.split(key)
    keys = jax.random.split(key, num=num)
    return keys, rng


def get_exp_sequence(lower, upper, n):
    return jnp.exp(jnp.linspace(jnp.log(lower), jnp.log(upper), n))
def get_linear_sequence(lower, upper, n):
    return jnp.linspace(lower, upper, n)


# @partial(jax.jit, static_argnames='dim')
def fill_lower_tri(v, dim):
    '''
        Make vector into lower-triangular matrix 
        https://github.com/google/jax/discussions/10146
    '''
    idxL = jnp.tril_indices(dim)
    L    = jnp.zeros((dim, dim), dtype=v.dtype).at[idxL].set(v) 
    idxD = jnp.diag_indices(dim)
    LD   = nn.softplus( L.at[idxD].get() )
    L    = L.at[idxD].set(LD)
    return L
