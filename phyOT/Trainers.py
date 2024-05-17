import sys
sys.path.append('../')
import jax
import jax.numpy as jnp
import jax.random as random
from config import *
import argparse
import pickle
import time
from phyOT.utils import get_keys_and_rng
from phyOT.jx_pot import sliced_wasserstein_distance_CDiag, sliced_wasserstein_distance
from functools import partial


def regularizer_lvlset(key, params_prior):
    lambda_val = params_prior['lambda_val']
    kappas_val = params_prior['kappas']
    reg_loss   =  1./(2. * regularizer_vals['lambda_sigma']**2.) * ( lambda_val - regularizer_vals['lambda_mu'] )** 2.
    reg_loss   +=  jnp.sum( 1./(2. * regularizer_vals['kappas_sigma']**2.) * ( kappas_val - regularizer_vals['kappas_mu'] )** 2. )
    return reg_loss

def regularizer_wm(key, params_prior):
    sigma_val = params_prior['sigma_val']
    ell_val   = params_prior['ell_val']
    nu_val    = params_prior['nu_val']

    reg_loss    =  1./(2. * regularizer_vals['sigma_sigma']**2.) * ( sigma_val - regularizer_vals['sigma_mu'] )** 2.
    reg_loss   +=  1./(2. * regularizer_vals['ell_sigma']**2.) * ( ell_val - regularizer_vals['ell_mu'] )** 2.
    reg_loss   +=  1./(2. * regularizer_vals['nu_sigma']**2.) * ( nu_val - regularizer_vals['nu_mu'] )** 2.

    return reg_loss


class Trainer_PriorCalib():

    def __init__(self, cfg, prior, solver):
        self.cfg = cfg
        self.dim = cfg.dim
        self.prior = prior
        self.solver = solver 
        self.prior_type = 'lvlset'

    def loss(self, rng, params_prior, y_b, obs_locs):

        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples)

        Zs, _ = jax.vmap(prior.sample_smooth_z, in_axes=(0, None, None))(keys, params_prior, solver.grid)

        Us    = jax.vmap(solver.solve, in_axes=(None, 0, None))(solver.u_init, Zs, f_field)

        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples)
        ys    = jax.vmap(observation_map, in_axes=(0,0,None))(keys, Us, obs_locs)

        key, rng = random.split(rng)
        # sw_y = sliced_wasserstein_distance(key, y_b, ys, p=2., n_projections=self.cfg.n_projections) ** 2.
        sw_y = y_b.shape[1] / 2 * sliced_wasserstein_distance_CDiag(key, y_b, ys, Gamma, p=2., n_projections=self.cfg.n_projections) ** 2.

        if regularizer_vals is not None:
            if self.prior_type == 'lvlset':
                reg_loss = regularizer_lvlset(None, params_prior)
            elif self.prior_type == 'wm':
                reg_loss = regularizer_wm(None, params_prior)
        else: reg_loss = 0.
        loss =  sw_y + reg_loss
        return loss, (sw_y, reg_loss)
    

    def update(self, rng, params_prior, opt_state_prior, 
                y_b, obs_locs):
        
        loss_grad_func = jax.value_and_grad(self.loss, argnums=(1), has_aux=True)

        loss_grad_aux_vals  = loss_grad_func(rng, params_prior, y_b, obs_locs)

        loss = loss_grad_aux_vals[0][0]
        aux  = loss_grad_aux_vals[0][1]
        grads_prior  = loss_grad_aux_vals[1]

        params_prior, opt_state_prior   = self.prior.update(grads_prior, params_prior, opt_state_prior)

        return loss, aux, params_prior, grads_prior, opt_state_prior
    

    def train(self, rng, params_prior, opt_state_prior,
                n_iters, dataYs, obs_locs):
        
        update_jit = jax.jit(self.update)

        idx = jnp.arange(0, dataYs.shape[0])
        loss_list = []
        aux_list  = []
        kappa_list = []
        lambda_vals_list = []
        for i in range(n_iters):

            key, rng = random.split(rng)
            y_b_idx = random.choice(key, idx, shape=(cfg.batch_size,), replace=True)
            y_b     = dataYs[y_b_idx]

            key, rng = random.split(rng)
            loss, aux,\
                  params_prior, grads_prior, opt_state_prior = update_jit(key, params_prior, opt_state_prior, 
                                                                            y_b, obs_locs)
            # print('params_prior:', params_prior)
            # print('grads_prior:', grads_prior)

            loss_list.append(loss)
            aux_list.append(aux)
            kappa_list.append(params_prior['kappas'])
            lambda_vals_list.append(params_prior['lambda_val'])


            if i % 10 == 0:
                print(f'iter: {i} | loss: {loss:.7f} | data_loss: {aux[0]:.7f} | res: {aux[1]:.7f}')
                print('params_prior:', params_prior)

        loss_list = jnp.array(loss_list)
        aux_list  = jnp.array(aux_list) 
        kappa_list = jnp.array(kappa_list)
        lambda_vals_list = jnp.array(lambda_vals_list)
        aux_dict = {'aux': aux_list, 'kappas':kappa_list, 'lambdas':lambda_vals_list}

        return loss_list, aux_dict, params_prior, opt_state_prior


    def train_wm(self, rng, params_prior, opt_state_prior,
                n_iters, dataYs, obs_locs):
        
        update_jit = jax.jit(self.update)

        idx = jnp.arange(0, dataYs.shape[0])
        loss_list = []
        aux_list  = []
        
        nu_list = []
        ell_list = []
        sigma_list = []
        
        for i in range(n_iters):

            key, rng = random.split(rng)
            y_b_idx = random.choice(key, idx, shape=(cfg.batch_size,), replace=True)
            y_b     = dataYs[y_b_idx]

            key, rng = random.split(rng)
            loss, aux,\
                  params_prior, grads_prior, opt_state_prior = update_jit(key, params_prior, opt_state_prior, 
                                                                            y_b, obs_locs)
            # print('params_prior:', params_prior)
            # print('grads_prior:', grads_prior)
            
            # reset prior parameter
            # params_prior['nu_val'] = init_params_prior['nu_val']
            # params_prior['sigma_val'] = init_params_prior['sigma_val']
            # params_prior['ell_val'] = init_params_prior['ell_val']
    
            params_prior = value_reset_prior(params_prior)



            loss_list.append(loss)
            aux_list.append(aux)
            nu_list.append(params_prior['nu_val'])
            sigma_list.append(params_prior['sigma_val'])
            ell_list.append(params_prior['ell_val'])

            if i % 10 == 0:
                print(f'iter: {i} | loss: {loss:.7f} | data_loss: {aux[0]:.7f} | res: {aux[1]:.7f}')
                print('params_prior:', params_prior)

        loss_list = jnp.array(loss_list)
        aux_list  = jnp.array(aux_list) 
        
        nu_list_jnp = jnp.array(nu_list)
        sigma_list_jnp = jnp.array(sigma_list)
        ell_list_jnp = jnp.array(ell_list)
        
        aux_dict = {'aux': aux_list, 'nus':nu_list_jnp, 'sigmas':sigma_list_jnp, 'ells':ell_list_jnp}

        return loss_list, aux_dict, params_prior, opt_state_prior
    
    

class Trainer_PriorCalib_OpLearn():

    def __init__(self, cfg, prior, fno, solver, chk_path):
        self.cfg = cfg
        self.dim = cfg.dim
        self.prior = prior
        self.fno   = fno
        self.solver = solver 
        self.chk_path = chk_path
        self.chk_iter_freq = 10_000
        self.prior_type = 'lvlset'


    def loss_data(self, rng, params_prior, params_fno, y_b, obs_locs):
        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples_data)
        Zs, _ = jax.vmap(prior.sample_smooth_z, in_axes=(0, None, None))(keys, params_prior, solver.grid)
        Us    = jax.vmap(self.fno.forward, in_axes=(None, 0))(params_fno, Zs)
        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples_data)
        ys    = jax.vmap(observation_map, in_axes=(0,0,None))(keys, Us, obs_locs)
        key, rng = random.split(rng)
        sw_y = y_b.shape[1] / 2 * sliced_wasserstein_distance_CDiag(key, y_b, ys, Gamma, p=2., n_projections=self.cfg.n_projections) ** 2.
        # sw_y = sliced_wasserstein_distance(key, y_b, ys, p=2., n_projections=self.cfg.n_projections) ** 2.
        if regularizer_vals is not None:
            if self.prior_type == 'lvlset':
                reg_loss = regularizer_lvlset(None, params_prior)
            elif self.prior_type == 'wm':
                reg_loss = regularizer_wm(None, params_prior)
        else: reg_loss = 0.
        loss =  sw_y + reg_loss
        return loss, (sw_y, reg_loss)
    

    def loss_res(self, rng, params_prior, params_fno):
        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples_res)
        Zs, _ = jax.vmap(prior.sample_smooth_z, in_axes=(0, None, None))(keys, params_prior, solver.grid)
        Us    = jax.vmap(self.fno.forward, in_axes=(None, 0))(params_fno, Zs)
        keys, rng = get_keys_and_rng(rng, num=cfg.n_z_samples_res)
        Rs    = jax.vmap(solver.residual, in_axes=(0, 0, None, None, None))(Us, Zs, f_field, False, None)
        w_r   = jnp.mean( jnp.sum(Rs.reshape(Rs.shape[0], -1)**2., axis=-1) ,axis=0)
        return w_r
    
    @partial(jax.jit, static_argnums=0)
    def update_operator(self, rng, params_prior, params_fno, opt_state_fno):
        
        print('### Using multistep decouple Op Learn ###')

        key, rng = random.split(rng)
        loss_grad_func_res = jax.value_and_grad(self.loss_res, argnums=(2), has_aux=False)
        loss_grad_aux_vals_res = loss_grad_func_res(key, params_prior, params_fno)

        loss_res = loss_grad_aux_vals_res[0]
        grads_fno = loss_grad_aux_vals_res[1]

        params_fno, opt_state_fno = self.fno.update(grads_fno, params_fno, opt_state_fno)

        return loss_res, params_fno, opt_state_fno, grads_fno
    
    @partial(jax.jit, static_argnums=0)
    def update_prior(self, rng, params_prior, opt_state_prior, params_fno, y_b, obs_locs):
        
        print('### Using multistep decouple Prior Learn ###')

        key, rng = random.split(rng)
        loss_grad_func_data = jax.value_and_grad(self.loss_data, argnums=(1), has_aux=True)
        loss_grad_aux_vals_data  = loss_grad_func_data(key, params_prior, params_fno, y_b, obs_locs)
        
        loss_data = loss_grad_aux_vals_data[0][0]
        aux  = loss_grad_aux_vals_data[0][1]
        grads_prior  = loss_grad_aux_vals_data[1]

        params_prior, opt_state_prior   = self.prior.update(grads_prior, params_prior, opt_state_prior)

        return loss_data, aux, params_prior, grads_prior, opt_state_prior
    
#############################################################################################
############## Multi Step Operator Decouple BiLevel Learning ################################
    # @partial(jax.jit, static_argnums=0)
    def loss_data_update_fno(self, rng, params_prior, opt_state_fno, params_fno, y_b, obs_locs):

        for _ in range(cfg.n_fno_step):
            key, rng = random.split(rng)
            loss_res, params_fno, opt_state_fno, grads_fno = self.update_operator(key, params_prior, params_fno, opt_state_fno)

        loss_data, aux = self.loss_data(key, params_prior, params_fno, y_b, obs_locs)

        return loss_data, (*aux, loss_res, params_fno, opt_state_fno, grads_fno)

    @partial(jax.jit, static_argnums=0)
    def update_decouple_bilvel_opt(self, rng, params_prior, opt_state_prior, params_fno, opt_state_fno, y_b, obs_locs):
        
        print('### Using multistep Decouple BiLevel Learning ###')
        
        if cfg.resOnly == True:
            print('### using res only training ###')
            key, rng = random.split(rng)
            loss_res, params_fno, opt_state_fno, grads_fno = self.update_operator(key, params_prior, params_fno, opt_state_fno)
            aux = (0., 0., loss_res)
            return 0., aux, params_prior, None, opt_state_prior, params_fno, opt_state_fno, grads_fno
        
        key, rng = random.split(rng)
        loss_grad_func_data = jax.value_and_grad(self.loss_data_update_fno, argnums=(1), has_aux=True)
        loss_grad_aux_vals_data  = loss_grad_func_data(key, params_prior, opt_state_fno, params_fno, y_b, obs_locs)
        
        loss_data = loss_grad_aux_vals_data[0][0]
        aux  = loss_grad_aux_vals_data[0][1]
        grads_prior  = loss_grad_aux_vals_data[1]

        params_prior, opt_state_prior   = self.prior.update(grads_prior, params_prior, opt_state_prior)

        loss = loss_data
        loss_res, params_fno, opt_state_fno, grads_fno = aux[2], aux[3], aux[4], aux[5]
        aux = (aux[0], aux[1], aux[2])

        return loss, aux, params_prior, grads_prior, opt_state_prior, params_fno, opt_state_fno, grads_fno

#####################################################################################

    def train(self, key, params_prior, opt_state_prior,
               params_fno, opt_state_fno,
                n_iters, dataYs, obs_locs):
        
        start_time = time.time()

        update_jit = self.update_decouple_bilvel_opt

        idx = jnp.arange(0, dataYs.shape[0])
        loss_list = []
        aux_list  = []
        kappa_list = []
        lambda_vals_list = []
        for i in range(n_iters):

            key, rng = random.split(key)
            y_b_idx = random.choice(key, idx, shape=(cfg.batch_size,), replace=True)
            y_b     = dataYs[y_b_idx]

            key, rng = random.split(rng)
            loss, aux,\
                  params_prior,grads_prior, opt_state_prior, \
                  params_fno, opt_state_fno, grads_fno = update_jit(key, params_prior, opt_state_prior, 
                                                params_fno, opt_state_fno,
                                                y_b, obs_locs)
            # print('params_prior:', params_prior)
            # print('grads_prior:', grads_prior)

            loss_list.append(loss)
            aux_list.append(aux)
            kappa_list.append(params_prior['kappas'])
            lambda_vals_list.append(params_prior['lambda_val'])

            if i % 100 == 0:
                print(f'iter: {i} | loss: {loss:.7f} | data_loss: {aux[0]:.7f}  | reg: {aux[1]:.7f} | res: {aux[2]:.7f}')
                print('params_prior:', params_prior)

            if i % self.chk_iter_freq == 0 or i == n_iters - 1:
                print('\n\n CHECK POINTING\n\n')
                loss_list_jnp = jnp.array(loss_list)
                aux_list_jnp  = jnp.array(aux_list) 
                kappa_list_jnp = jnp.array(kappa_list)
                lambda_vals_list_jnp = jnp.array(lambda_vals_list)
                aux_dict = {'aux': aux_list_jnp, 'kappas':kappa_list_jnp, 'lambdas':lambda_vals_list_jnp}

                D = {'iter': [i, n_iters],
                     'loss':loss_list_jnp, 'aux':aux_dict, 
                     'params_prior':params_prior, 'opt_state_prior':opt_state_prior,
                     'params_fno':params_fno, 'opt_state_fno':opt_state_fno,
                     'time':time.time() - start_time, 
                     'cfg':cfg, 'config_txt':config_txt}

                with open(self.chk_path, 'wb') as handle:
                    pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
        
    
    def train_wm(self, key, params_prior, opt_state_prior,
               params_fno, opt_state_fno,
                n_iters, dataYs, obs_locs):
        
        start_time = time.time()

        update_jit = self.update_decouple_bilvel_opt

        idx = jnp.arange(0, dataYs.shape[0])
        loss_list = []
        aux_list  = []
        
        sigma_list = []
        nu_list = []
        ell_list = []
        
        for i in range(n_iters):

            key, rng = random.split(key)
            y_b_idx = random.choice(key, idx, shape=(cfg.batch_size,), replace=True)
            y_b     = dataYs[y_b_idx]

            key, rng = random.split(rng)
            loss, aux,\
                  params_prior, grads_prior, opt_state_prior, \
                  params_fno, opt_state_fno, grads_fno = update_jit(key, params_prior, opt_state_prior, 
                                                params_fno, opt_state_fno,
                                                y_b, obs_locs)
            # print('params_prior:', params_prior)
            # print('grads_prior:', grads_prior)
            
            params_prior = value_reset_prior(params_prior)

            loss_list.append(loss)
            aux_list.append(aux)
            nu_list.append(params_prior['nu_val'])
            sigma_list.append(params_prior['sigma_val'])
            ell_list.append(params_prior['ell_val'])

            if i % 100 == 0:
                print(f'iter: {i} | loss: {loss:.7f} | data_loss: {aux[0]:.7f}  | reg: {aux[1]:.7f} | res: {aux[2]:.7f}')
                print('params_prior:', params_prior)

            if i % self.chk_iter_freq == 0 or i == n_iters - 1:
                print('\n\n CHECK POINTING\n\n')
                loss_list_jnp = jnp.array(loss_list)
                aux_list_jnp  = jnp.array(aux_list) 
                
                nu_list_jnp = jnp.array(nu_list)
                ell_list_jnp = jnp.array(ell_list)
                sigma_list_jnp = jnp.array(sigma_list)

                aux_dict = {'aux': aux_list_jnp, 'nus':nu_list_jnp, 'ells':ell_list_jnp, 'sigmas':sigma_list_jnp}

                D = {'iter': [i, n_iters],
                     'loss':loss_list_jnp, 'aux':aux_dict, 
                     'params_prior':params_prior, 'opt_state_prior':opt_state_prior,
                     'params_fno':params_fno, 'opt_state_fno':opt_state_fno,
                     'time':time.time() - start_time, 
                     'cfg':cfg, 'config_txt':config_txt}

                with open(self.chk_path, 'wb') as handle:
                    pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
