import jax
import jax.numpy as jnp
import jax.random as random
import time
import pickle

from phyEBM.utils import *

class DataSet_Gen():
    def __init__(self, prior, solver, observation_map, smooth):
        self.prior = prior
        self.solver = solver
        self.observation_map = observation_map
        self.dim = 1
        self.smooth = smooth

    def gen_dataset(self, key, cfg, path, params, f_func, n_samples, n_obs_locs):
        '''
            key,
            params dict for prior
            forcing function
            grid for u_sln
            n_sample in dataset
        '''

        keys, rng = get_keys_and_rng(key, num=n_samples)
        if self.smooth == 'False':
            print('SHARP LEVELSET DATA GEN')
            dataZs, a_fields = jax.vmap(self.prior.sample_lvl_set, in_axes=(0, None, None))(keys, params, self.solver.grid)
        elif self.smooth == 'True':
            print('SMOOTH LEVELSET DATA GEN')
            dataZs, a_fields = jax.vmap(self.prior.sample_smooth_z, in_axes=(0, None, None))(keys, params, self.solver.grid)

        print('dataZs.shape', dataZs.shape)

        f_field = f_func(self.solver.grid)
        if self.dim == 1:
            u       = jnp.zeros(self.solver.sln_shape)
        elif self.dim == 2:
            u       = jnp.zeros(self.solver.sln_shape[1:])

        start_time = time.time()
        dataUs = []
        for i in range(n_samples):
            u_sln, _, res_slns = self.solver.solve(
                u.reshape(-1,), dataZs[i].reshape(-1,), f_field.reshape(-1,))
            dataUs.append(u_sln)
            print(f'solution {i}, res = {res_slns}')
        dataUs = jnp.stack(dataUs)
        print(f'time taken = {time.time() - start_time:.3f}s')

        key, rng = random.split(rng)
        obs_locs  = random.uniform(key, shape=(n_obs_locs, self.dim), minval=0., maxval=1.) 

        keys, rng = get_keys_and_rng(rng, num=dataUs.shape[0])
        dataYs = jax.vmap(self.observation_map, in_axes=(0, 0, None))(keys, dataUs, obs_locs)

        dataSet = {'dataYs':dataYs, 'dataUs':dataUs, 'dataZs':dataZs, 'dataAs':a_fields, 'dataFs':f_field,
                    'obs_locs':obs_locs, 'grid':self.solver.grid, 'cfg':cfg, 'params':params, 'smooth':self.smooth}
        
        with open(path, 'wb') as handle:
            pickle.dump(dataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)