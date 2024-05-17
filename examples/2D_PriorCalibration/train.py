import sys
sys.path.append('../')
import jax
import jax.numpy as jnp
import jax.random as random
from config import *
import argparse
import pickle
import time
from phyOT.Trainers import Trainer_PriorCalib
from phyOT.utils import get_keys_and_rng
from phyOT.jx_pot import sliced_wasserstein_distance_CDiag, sliced_wasserstein_distance

rng = random.key(0)

def main(rng, data):

    dataYs   = data['dataYs']
    obs_locs = data['obs_locs']

    params_prior = {'lambda_val': jnp.log(0.1),
                     'kappas':jnp.array([jnp.log(5.), jnp.log(10.)])}
    # params_prior = init_params_prior
    
    params_prior, opt_state_prior = prior.init_optimizer(params_prior)

    trainer = Trainer_PriorCalib(cfg, prior, solver)

    start_time = time.time()

    key, rng = random.split(rng)

    loss, aux, params_prior, opt_state_prior = trainer.train(key, params_prior, opt_state_prior, 
                  cfg.train_iters, dataYs, obs_locs)
    
    time_taken = time.time() - start_time
    print('time taken:', time_taken)

    return {'loss':loss, 'aux':aux, 'cfg':cfg,
            'params_prior':params_prior, 'opt_state_prior':opt_state_prior, 'time':time_taken}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--filename_dataset", help="Database name", default='data1.pkl')
    parser.add_argument("-ft", "--filename_train", help="Database name", default='model1.pkl')
    args = parser.parse_args()

    print('fd: ', args.filename_dataset)
    print('ft: ', args.filename_train)

    with open(path + args.filename_dataset, 'rb') as f:
        data = pickle.load(f)
    
    trained_dict = main(rng, data)

    with open(path + args.filename_train, 'wb') as handle:
        pickle.dump(trained_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Training info dumped to file {args.filename_train}')