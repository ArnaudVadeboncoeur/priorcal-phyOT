import sys
sys.path.append('../')
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'

import jax
import jax.numpy as jnp
import jax.random as random
from config import *
import argparse
import pickle
import time
from phyOT.Trainers import Trainer_PriorCalib

rng = random.key(1)


def main_multi_trainer(rng, data):

    batch_list   = [1, 10, 100, 1_000, 10_000]
    n_data_list  = [1, 10, 100, 1_000, 10_000]

    # batch_list   = [1, 10]
    # n_data_list  = [10, 100]

    trial = range(100)

    D = {'batch_list': batch_list, 'n_data_list':n_data_list, 'trial':trial}

    for i, nd in enumerate(n_data_list):
        for j, nb in enumerate(batch_list):
            for t in trial:

                print('trial:', t)
                print('nd, nb:', nd, nb)
                cfg.batch_size = nb
                key, rng = random.split(rng)

                dataYs   = data['dataYs'][:nd, :]
                obs_locs = data['obs_locs']
                key1, key2, key3, rng = random.split(rng, num=4)
                params_prior = {'lambda_val': random.uniform(key1, minval=jnp.log(0.5), maxval=jnp.log(4.)),
                                'kappas':jnp.array([random.uniform(key2, minval=jnp.log(0.5), maxval=jnp.log(4.))
                                                    ,random.uniform(key3, minval=jnp.log(6.), maxval=jnp.log(10.))])}

                params_prior, opt_state_prior = prior.init_optimizer(params_prior)
                start_time = time.time()
                solver.device='cpu'
                trainer = Trainer_PriorCalib(cfg, prior, solver)
                loss, aux, \
                    params_prior, opt_state_prior = trainer.train(key, params_prior, opt_state_prior, 
                            cfg.train_iters, dataYs, obs_locs)
                time_taken = time.time() - start_time
                D[f'run_{i}_{j}_{t}'] = {'loss':loss, 'aux':aux, 'cfg':cfg,
                    'params_prior':params_prior, 'opt_state_prior':opt_state_prior, 'time':time_taken, 
                    'n_data_pts': nd, 'n_data_batch':nb}

    return D

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--filename_dataset", help="Database name", default='data3_10k.pkl')
    parser.add_argument("-ft", "--filename_train", help="Database name", default='models_mt_adam.pkl')
    args = parser.parse_args()
    print('\n\n\n')
    print('##########')
    print(f"USING DATA FILE: {args.filename_dataset}")
    print(f"DUMP TO: {args.filename_train}")
    print('##########')
    print('\n\n\n')

    with open(path + args.filename_dataset, 'rb') as f:
        data = pickle.load(f)

    trained_dicts = main_multi_trainer(rng, data)

    with open(path + args.filename_train, 'wb') as handle:
        pickle.dump(trained_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Training info dumped to file {args.filename_train}')