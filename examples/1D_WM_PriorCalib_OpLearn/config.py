import os
import sys
import jax
import jax.numpy as jnp
import jax.random as random
from ml_collections import config_dict
import optax
from src.poisson1DSolver import PoissonSolver
from src.WM_prior_1D import WM_Prior_1D
from src.FNO import FNO, FNO_utils1D
from phyEBM.utils import *

rng = random.key(0)
print('file direct:', os.path.dirname(os.path.realpath(__file__)))
path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
os.makedirs(path, exist_ok=True)

def make_config():
    ### Make config dict ###
    cfg = config_dict.ConfigDict()

    cfg.dim = 1
    cfg.train_iters = 5_000
    cfg.batch_size = 100
    cfg.n_z_samples_res  = 20
    cfg.n_z_samples_data = 1000

    cfg.n_projections = 1000
    cfg.resOnly = False
    cfg.resFact = 1. # 0.005
    cfg.n_fno_step = 10
    cfg.chk_iter_freq = 1_000

    cfg.Prior = config_dict.ConfigDict()
    cfg.Prior.n_basis = 20
    cfg.Prior.learning_rate = 1e-2 * 0.5
    cfg.Prior.n_decay_steps = int( cfg.train_iters / 4)
    cfg.Prior.decay_rate    = 0.5

    cfg.FNO = config_dict.ConfigDict()
    cfg.FNO.dim_v   = 64
    cfg.FNO.n_modes = 8 #32
    cfg.FNO.out_dim = 1
    cfg.FNO.activation = 'silu'
    cfg.FNO.n_layers   = 4
    cfg.FNO.learning_rate = 1e-3
    cfg.FNO.n_decay_steps = int( cfg.train_iters * cfg.n_fno_step / 4)
    cfg.FNO.decay_rate    = 0.5
    cfg.FNO.opt_type   = 'adam' # adam || amsgrad


    cfg.Mesh = config_dict.ConfigDict()
    cfg.Mesh.nx = 100
    cfg.Mesh.forcing_const_val = 10.

    cfg.Observation = config_dict.ConfigDict()
    cfg.Observation.n_data = 1000
    cfg.Observation.n_locations = 50
    cfg.Observation.sigma = 0.01

    return cfg

cfg = make_config()

solver = PoissonSolver(cfg.Mesh.nx, verbose=True)
solver.solverChoice = 'linear'

prior = WM_Prior_1D(cfg.Prior.n_basis)

lr_prior = optax.exponential_decay(
    cfg.Prior.learning_rate,
    cfg.Prior.n_decay_steps,
    cfg.Prior.decay_rate,
    staircase=True
)
prior.opt = optax.adam(lr_prior)


init_params_prior = {'sigma_val': jnp.log( 1. ), 
                     'ell_val': jnp.log( jnp.exp(0.5) - 1. ),
                     'nu_val': jnp.log( 0.75 - 0.5) }

# regularizer_vals = {'sigma_sigma': 2., 'sigma_mu': jnp.log(2.), 
#                     'ell_sigma': 2., 'ell_mu': jnp.log(1.), 
#                     'nu_sigma': 2., 'nu_mu': jnp.log(7./2.), }
regularizer_vals = None

def value_reset_prior(params_prior):
    #! reset prior parameter
    # params_prior['nu_val'] = init_params_prior['nu_val']
    params_prior['sigma_val'] = init_params_prior['sigma_val']
    # params_prior['ell_val'] = init_params_prior['ell_val']
    return params_prior

fno = FNO(cfg.FNO, FNO_utils1D)
import types
def forward(self, params, z ):
    zX = jnp.concatenate((z[..., None], solver.grid[ ..., None]), -1)
    u = self.apply(params, zX).reshape(-1,) #select 1D sln
    u = solver.boundary_func(u, solver.grid)
    return u
fno.forward = types.MethodType(forward, fno)

def forcing_function(grid):
    return grid ** 0. * cfg.Mesh.forcing_const_val # CONSTANT VALUE 1

f_field = forcing_function(solver.grid)

def observation_map(key, u, x):
    x = x.reshape(-1,)
    y = solver.inter1d_wrapper(u, x)
    y += cfg.Observation.sigma * random.normal(key, shape=y.shape)
    return y

Gamma = jnp.ones((cfg.Observation.n_locations,)) * cfg.Observation.sigma ** 2.

from pathlib import Path
file_path = os.path.dirname(os.path.realpath(__file__)) + '/config.py'
config_txt = Path(file_path).read_text()