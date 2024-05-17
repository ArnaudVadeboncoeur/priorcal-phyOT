import os
import sys
import jax
import jax.numpy as jnp
import jax.random as random
from ml_collections import config_dict
import optax
from phyOT.poisson1DSolver import PoissonSolver
from phyOT.level_set_prior_1D import Level_Set_Prior_1D
from phyOT.utils import *

rng = random.key(0)
print('file direct:', os.path.dirname(os.path.realpath(__file__)))
path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
os.makedirs(path, exist_ok=True)

def make_config():
    ### Make config dict ###
    cfg = config_dict.ConfigDict()

    cfg.dim = 1
    cfg.train_iters = 1000
    cfg.batch_size = 1000
    cfg.n_z_samples = 1000
    cfg.n_projections = 1000 # 10_000

    cfg.Prior = config_dict.ConfigDict()
    cfg.Prior.n_basis = 20
    cfg.Prior.learning_rate = 1e-2
    cfg.Prior.n_decay_steps = int( cfg.train_iters / 4)
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

solver = PoissonSolver(cfg.Mesh.nx, verbose=True)
solver.solverChoice = 'linear'
solver.residuals_info = False

prior = Level_Set_Prior_1D(cfg.Prior.n_basis)
prior.ell = 10
lr_prior = optax.exponential_decay(
    cfg.Prior.learning_rate,
    cfg.Prior.n_decay_steps,
    cfg.Prior.decay_rate,
    staircase=True
)
prior.opt = optax.adam(lr_prior)

init_params_prior = {'lambda_val': jnp.log(8.),
                     'kappas':jnp.array([jnp.log(1.), jnp.log(2.)])}
regularizer_vals = None

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