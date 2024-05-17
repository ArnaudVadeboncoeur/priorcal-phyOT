import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import optax
import flax.linen as nn

from phyOT.utils import  get_keys_and_rng


class WM_Prior_1D():
    
    def __init__(self, N):
        self.dim = 1.
        self.N = N
        self.basis_indices = jnp.arange(1, stop=self.N+1)
        
        self.opt = optax.adam(1e-2)

        # Parameters are sigma, ell, nu


    def sample_smooth_field(self, rng, params, grid):
        
        sigma_val = jnp.exp(params['sigma_val'] )
        ell_val   = jnp.exp(params['ell_val'] ) 
        nu_val    = jnp.exp(params['nu_val'] )
        
        gamma_val = sigma_val**2. * 2.**self.dim * jnp.pi**(self.dim/2.) * jax.scipy.special.gamma(nu_val + self.dim/2.) / jax.scipy.special.gamma(nu_val)
        
        print('sigma_val', sigma_val)
        print('ell_val', ell_val)
        print('nu_val', nu_val)
        print('gamma_val', gamma_val)


        def add_basis(field_rng, j):
            
            key, field_rng[1] = random.split(field_rng[1])
            
            field_rng[0] += jnp.sqrt( gamma_val * ell_val**self.dim * (ell_val**2. * (j**2.) * jnp.pi**2. + 1.)**(-nu_val - self.dim/2) ) \
                            * random.normal(key) * jnp.cos(jnp.pi*j*grid)
                            
            return field_rng, None
        
        a_field_rng, stack = jax.lax.scan(add_basis, [grid*0., rng], self.basis_indices)
        a_field = a_field_rng[0]
        return a_field
    

    
    def sample_smooth_z(self, key, params, grid):

        a_field = self.sample_smooth_field(key, params, grid)

        z_field = jnp.exp(a_field)
        
        return z_field, a_field

    def init_optimizer(self, params):

        print('using optimizer:', self.opt)
        opt_state  = self.opt.init(params)

        return params, opt_state
    
    def update(self, grads, params, opt_state):
        print('UPDATE PRIOR') 
        updates, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    

if __name__ == '__main__':

    WM_prior = WM_Prior_1D(20)

    params = {'sigma_val': jnp.log( 1. ), 
              'ell_val': jnp.log( jnp.exp(0.5) - 1. ),
              'nu_val': jnp.log( 0.75 - 0.5) }
    
    params, opt_state = WM_prior.init_optimizer(params)

    rng  = random.key(42)
    grid = jnp.linspace(0., 1., 500)

    n = 10
    m = 4
    nu_vals = jnp.linspace(jnp.log(0.51 - 0.5), jnp.log(10. - 0.5), n)

    os.makedirs('tmp_WM_1D', exist_ok=True)

    for i in range(n):

        for j in range(m):

            key, rng = random.split(rng)
            params['nu_val'] = nu_vals[i]

            z_field, a_field = WM_prior.sample_smooth_z(key, params, grid)

            plt.plot(grid, z_field, label='z')
            plt.plot(grid, a_field, label='a')
            plt.legend()
            plt.grid()
            plt.savefig(f'tmp_WM_1D/z_sample_{jnp.exp(nu_vals[i])+0.5:.1f}_{j}.png')
            plt.close()

