import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import optax
import flax.linen as nn

from phyOT.utils import  get_keys_and_rng


class Level_Set_Prior_2D():
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.basis_indices = jnp.stack(jnp.meshgrid( jnp.arange(1, stop=self.N+1),
                                                     jnp.arange(1, stop=self.M+1), indexing='ij'))
        self.basis_indices_list = self.basis_indices.reshape(2, -1).T
        # print('self.basis_indices_list:', self.basis_indices_list)
        self.beta = 4
        
        self.norm = 'L2' # infty || L2
        # self.norm = 'infty' # infty || L2
        self.ell = 10
        self.sample_smooth_field = self.sample_smooth_field_laxscan
        self.opt = optax.adam(1e-2)



    def sample_smooth_field_laxscan(self, rng, params, grid):
        lambda_val = jnp.exp(params['lambda_val'])
        print('lambda_val', lambda_val)
        beta = self.beta

        def add_basis(field_rng, jk):
            key, field_rng[1] = random.split(field_rng[1])
            field_rng[0] += ((jk[0]**2.+jk[1]**2.)*jnp.pi**2. + lambda_val**2.)**(-beta/2) \
                            * random.normal(key) * jnp.cos(jnp.pi*jk[0]*grid[0]) *  jnp.cos(jnp.pi*jk[1]*grid[1])
            return field_rng, None
        
        a_field_rng, stack = jax.lax.scan(add_basis, [grid[0]*0., rng], self.basis_indices_list)
        a_field = a_field_rng[0]
        return a_field
    

    def sample_smooth_field_vmap(self, key, params, grid):

        lambda_val = jnp.exp(params['lambda_val'])
        print('lambda_val', lambda_val)
        beta = self.beta

        rand_basis = lambda key_val, n, m: ((n**2.+m**2.)*jnp.pi**2. + lambda_val**2.)**(-beta/2) \
                * random.normal(key_val) * jnp.cos(jnp.pi*n*grid[0]) *  jnp.cos(jnp.pi*m*grid[1])

        keys, rng = get_keys_and_rng(key, num=self.N * self.M)  
        a_field = jnp.sum(jax.vmap(rand_basis, in_axes=(0,0,0))(keys, 
                    self.basis_indices[0].reshape(-1,), self.basis_indices[1].reshape(-1,)), axis=0)

        return a_field
    

    def sample_lvl_set(self, key, params, grid):

        a_field = self.sample_smooth_field(key, params, grid)

        kappas = params['kappas']

        level_set_field = jnp.where(a_field < 0, jnp.exp(kappas[0]), jnp.exp(kappas[1]))
        # level_set_field = a_field

        return level_set_field, a_field
    
    def sample_smooth_z(self, key, params, grid):

        a_field = self.sample_smooth_field(key, params, grid)

        kappas = params['kappas']
        kappa_1 = jnp.exp(kappas[0])
        kappa_2 = jnp.exp(kappas[1])

        a_infty = jnp.max(jnp.abs(a_field)) # infinity norm

        dx = 1./( grid[0].shape[0] - 1)
        print('dx:', dx)

        a_L2 = jnp.sqrt( jnp.sum( a_field**2. * dx**2. ) )

        print(f'a_infty: {a_infty}, a_L2: {a_L2}')

        if self.norm == 'infty':
            a_norm = a_infty
        elif self.norm == 'L2':
            a_norm = a_L2

        a_bar_field = a_field / a_norm

        z_field = jnp.tanh((a_bar_field) * self.ell ) * 0.5 * (kappa_2 - kappa_1) + kappa_1 + 0.5 * (kappa_2 - kappa_1)
    
        return z_field, a_bar_field

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

    level_set_prior = Level_Set_Prior_2D(20, 20)

    params = {'lambda_val': jnp.log(10.),
              'kappas':jnp.array([jnp.log(1.), jnp.log(2.)])}
    
    params, opt_state = level_set_prior.init_optimizer(params, 1e-3)

    rng  = random.key(42)
    x1 = jnp.linspace(0., 1., 500)
    x2 = x1
    grid = jnp.stack(jnp.meshgrid(x1,x2, indexing='ij'))

    n = 10
    m = 4
    lambda_vals = jnp.linspace(jnp.log(0.5), jnp.log(25.), n)

    os.makedirs('tmp_smooth_lvlset', exist_ok=True)

    for i in range(n):

        for j in range(m):

            key, rng = random.split(rng)
            params['lambda_val'] = lambda_vals[i]
            # lvl_set, a_field = level_set_prior.sample_lvl_set(key, params, grid)
            lvl_set, a_field = level_set_prior.sample_smooth_lvl_set(key, params, grid)



            plt.contourf(grid[0], grid[1], lvl_set, 50)
            plt.colorbar()
            plt.savefig(f'tmp_smooth_lvlset/lvl_set_sample_{jnp.exp(lambda_vals[i]):.1f}_{j}.png')
            plt.close()

            plt.contourf(grid[0], grid[1], a_field, 50)
            plt.colorbar()
            plt.savefig(f'tmp_smooth_lvlset/a_field_sample_{jnp.exp(lambda_vals[i]):.1f}_{j}.png')
            plt.close()

