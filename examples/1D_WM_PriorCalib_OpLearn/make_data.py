import sys
sys.path.append('../')
import jax
from config import *
from phyOT.dataset_gen import DataSet_Gen
import jax.random as random
import argparse
import pickle

def make_data(rng, filename, smooth):
    key, rng = random.split(rng)
    dataset_generator = DataSet_Gen(prior, solver, observation_map, smooth)
    
    key, rng = random.split(rng)
    dataset_generator.gen_dataset(key, cfg, path+filename, init_params_prior, forcing_function,
                                   cfg.Observation.n_data, cfg.Observation.n_locations)
    
if __name__ == '__main__':
    rng = random.key(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Database name", default='data1.pkl')
    args = parser.parse_args()

    print(args)

    make_data(rng, args.filename, smooth='True')

    with open(path + args.filename, 'rb') as f:
        d = pickle.load(f)