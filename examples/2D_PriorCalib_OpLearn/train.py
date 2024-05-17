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
from phyOT.FNO import FNO, FNO_utils2D
from phyOT.Trainers import Trainer_PriorCalib_OpLearn
from phyOT.jx_pot import sliced_wasserstein_distance_CDiag, sliced_wasserstein_distance

rng = random.key(0)

def main(rng, data, chk_path):

    dataYs   = data['dataYs']
    obs_locs = data['obs_locs']

    zX = jnp.concatenate((solver.grid[0][None, ...], solver.grid), 0)
    zX = jnp.transpose(zX, axes=(1,2,0))
    key, rng = random.split(rng)
    params_fno, opt_state_fno = fno.init_model(key, zX)

    params_prior = {'lambda_val': jnp.log(0.1),
                     'kappas':jnp.array([jnp.log(5.), jnp.log(10.)])}
    
    params_prior, opt_state_prior = prior.init_optimizer(params_prior)

    trainer = Trainer_PriorCalib_OpLearn(cfg, prior, fno, solver, chk_path)
    trainer.chk_iter_freq = cfg.chk_iter_freq
    start_time = time.time()

    key, rng = random.split(rng)

   
    trainer.train(key, params_prior, opt_state_prior, 
                  params_fno, opt_state_fno, 
                  cfg.train_iters, dataYs, obs_locs)
    
    time_taken = time.time() - start_time
    print('time taken:', time_taken)

    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-fd", "--filename_dataset", help="Database name", default='data1.pkl')
    parser.add_argument("-ft", "--filename_train", help="Database name", default='model_adam.pkl')
    args = parser.parse_args()

    print('fd: ', args.filename_dataset)
    print('ft: ', args.filename_train)

    with open(path + args.filename_dataset, 'rb') as f:
        data = pickle.load(f)
    
    chk_path = path + args.filename_train
    
    main(rng, data, chk_path)

    print(f'Training info dumped to file {args.filename_train}')