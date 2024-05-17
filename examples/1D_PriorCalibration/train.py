import sys
sys.path.append('../')
import jax
import jax.numpy as jnp
import jax.random as random
from config import *
import argparse
import pickle
from phyOT.Trainers import Trainer_PriorCalib
import time
from phyOT.utils import get_keys_and_rng
from phyOT.jx_pot import sliced_wasserstein_distance_CDiag, sliced_wasserstein_distance

rng = random.key(1)


def main(rng, data):

    dataYs   = data['dataYs']
    obs_locs = data['obs_locs']
    params_prior = {'lambda_val': 0.1,
                     'kappas':jnp.array([jnp.log(5.), jnp.log(10.)])}
    
    params_prior, opt_state_prior = prior.init_optimizer(params_prior)
    start_time = time.time()
    trainer = Trainer_PriorCalib(cfg, prior, solver)
    loss, aux, \
        params_prior, opt_state_prior = trainer.train(rng, params_prior, opt_state_prior, 
                  cfg.train_iters, dataYs, obs_locs)
    time_taken = time.time() - start_time
    return {'loss':loss, 'aux':aux, 'cfg':cfg, 'config_txt':config_txt,
            'params_prior':params_prior, 'opt_state_prior':opt_state_prior, 'time':time_taken}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--filename_dataset", help="Database name", default='data1_smooth.pkl')
    parser.add_argument("-ft", "--filename_train", help="Database name", default='default.pkl')
    args = parser.parse_args()
    print('\n\n\n')
    print('##########')
    print(f"USING DATA FILE: {args.filename_dataset}")
    print(f"DUMP TO: {args.filename_train}")
    print('##########')
    print('\n\n\n')

    with open(path + args.filename_dataset, 'rb') as f:
        data = pickle.load(f)
    
    trained_dict = main(rng, data)

    with open(path + args.filename_train, 'wb') as handle:
        pickle.dump(trained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Training info dumped to file {args.filename_train}')