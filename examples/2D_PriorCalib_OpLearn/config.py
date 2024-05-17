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
from ml_collections import config_dict
import optax
from phyOT.poisson2DSolver import Poisson2DSolver
from phyOT.level_set_prior_2D import Level_Set_Prior_2D
from phyOT.FNO import FNO, FNO_utils2D

rng = random.key(0)
print('file direct:', os.path.dirname(os.path.realpath(__file__)))
path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
os.makedirs(path, exist_ok=True)


def make_config():
    ### Make config dict ###
    cfg = config_dict.ConfigDict()

    cfg.dim = 2
    cfg.train_iters = 5_000
    cfg.batch_size = 100
    cfg.n_z_samples_res  = 20
    cfg.n_z_samples_data = cfg.batch_size
    cfg.n_projections = 1000
    cfg.resOnly = False
    cfg.resFact = 1.
    cfg.n_fno_step = 10
    cfg.chk_iter_freq = 5_000
    
    cfg.FNO = config_dict.ConfigDict()
    cfg.FNO.dim_v   = 64
    cfg.FNO.n_modes = 8
    cfg.FNO.out_dim = 1
    cfg.FNO.activation = 'silu'
    cfg.FNO.n_layers   = 4
    cfg.FNO.learning_rate = 1e-3 
    cfg.FNO.n_decay_steps = int( cfg.train_iters * cfg.n_fno_step / 4)
    cfg.FNO.decay_rate    = 0.5
    cfg.FNO.opt_type   = 'adam' # adam || amsgrad


    cfg.Prior = config_dict.ConfigDict( )
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

solver = Poisson2DSolver(cfg.Mesh.nx, verbose=True)
solver.solverChoice = 'cg'

prior = Level_Set_Prior_2D(cfg.Prior.n_basis, cfg.Prior.n_basis)
prior.ell = 5
lr_prior = optax.exponential_decay(
    cfg.Prior.learning_rate,
    cfg.Prior.n_decay_steps,
    cfg.Prior.decay_rate,
    staircase=True
)

prior.opt = optax.adam(lr_prior)

init_params_prior = {'lambda_val': jnp.log(5.),
                     'kappas':jnp.array([jnp.log(1.), jnp.log(2.)])}

regularizer_vals = {'lambda_sigma': 2., 'lambda_mu': jnp.log(10.), 
                    'kappas_sigma': jnp.array([2., 2.]), 'kappas_mu': jnp.array([jnp.log(3.), jnp.log(3.)])}
# regularizer_vals = None

fno = FNO(cfg.FNO, FNO_utils2D)
import types
def forward(self, params, z ):
    zX = jnp.concatenate((z[None, ...], solver.grid), 0)
    zX = jnp.transpose(zX, (1, 2, 0))
    u = self.apply(params, zX)
    u = u[..., 0]
    u = solver.boundary_func(u, solver.grid)
    return u
fno.forward = types.MethodType(forward, fno)

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