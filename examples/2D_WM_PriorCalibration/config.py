import os
import sys
# sys.path.append('../')
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.95'
import jax
from jax import config
config.update("jax_debug_nans", True)
import jax.numpy as jnp
import jax.random as random
import optax
from ml_collections import config_dict
from phyOT.poisson2DSolver import Poisson2DSolver
from phyOT.WM_prior_2D import WM_Prior_2D

rng = random.key(1)
print('file direct:', os.path.dirname(os.path.realpath(__file__)))
path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
os.makedirs(path, exist_ok=True)

def make_config():
    ### Make config dict ###
    cfg = config_dict.ConfigDict()

    cfg.dim = 2
    cfg.train_iters = 5_000 #20000 for data2
    cfg.batch_size = 100
    cfg.n_z_samples = 100
    cfg.n_projections = 1000

    cfg.Prior = config_dict.ConfigDict()
    cfg.Prior.n_basis = 20
    cfg.Prior.learning_rate = 1e-2
    cfg.Prior.n_decay_steps = int( cfg.train_iters / 4) #6 for data2
    cfg.Prior.decay_rate    = 0.5

    cfg.Mesh = config_dict.ConfigDict()
    cfg.Mesh.nx = 100
    cfg.Mesh.forcing_const_val = 10.

    cfg.Observation = config_dict.ConfigDict()
    cfg.Observation.n_data = 1000
    cfg.Observation.n_locations = 50
    cfg.Observation.sigma = 0.01 

    return cfg

cfg = make_config()

solver = Poisson2DSolver(cfg.Mesh.nx)
solver.solverChoice = 'cg' # 'cg' || ''bicgstab ||'gmres'

# # for direct observaiton of z field
#     # data gen
# # solver.solve = lambda u, z, f: (z.reshape(cfg.Mesh.nx, cfg.Mesh.nx), None, None)
#     # training
# solver.solve = lambda u, z, f: z.reshape(cfg.Mesh.nx, cfg.Mesh.nx)

prior = WM_Prior_2D(cfg.Prior.n_basis, cfg.Prior.n_basis)

lr_prior = optax.exponential_decay(
    cfg.Prior.learning_rate,
    cfg.Prior.n_decay_steps,
    cfg.Prior.decay_rate,
    staircase=True
)
prior.opt = optax.adam(lr_prior)


#! data1
init_params_prior = {'sigma_val': jnp.log( 1. ), 
                     'ell_val': jnp.log( 0.5 ),
                     'nu_val': jnp.log( 1.5 ) }


#! data2
# init_params_prior = {'sigma_val': jnp.log( 1. ), 
#                      'ell_val': jnp.log( 0.5 ),
#                      'nu_val': jnp.log( 4. ) }

regularizer_vals = {'sigma_sigma': 2., 'sigmamu': jnp.log(1.), 
                    'ell_sigma': 2., 'ell_mu': jnp.log(1.), 
                    'nu_sigma': 2., 'nu_mu': jnp.log(7./2.) }

def value_reset_prior(params_prior):
    #! reset prior parameter
    # params_prior['nu_val'] = init_params_prior['nu_val']
    params_prior['sigma_val'] = init_params_prior['sigma_val']
    # params_prior['ell_val'] = init_params_prior['ell_val']
    return params_prior

regularizer_vals = None

def forcing_function(grid):
    return grid[0] ** 0. * cfg.Mesh.forcing_const_val

f_field = forcing_function(solver.grid)

def observation_map(key, u, x):
    y = solver.interp2d_wrapped(u, x)
    y += cfg.Observation.sigma * random.normal(key, shape=y.shape)
    return y

Gamma = jnp.ones((cfg.Observation.n_locations,)) * cfg.Observation.sigma ** 2.

from pathlib import Path
file_path = os.path.dirname(os.path.realpath(__file__)) + '/config.py'
config_txt = Path(file_path).read_text()