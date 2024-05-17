import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
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
    dataset_generator.dim = 2
    dataset_generator.solver.solverChoice='cg'
    dataset_generator.solver.device='cpu'
    dataset_generator.solver.residuals_info = True
    
    key, rng = random.split(rng)
    dataset_generator.gen_dataset(key, cfg, path+filename, init_params_prior, forcing_function,
                                   cfg.Observation.n_data, cfg.Observation.n_locations)
    
if __name__ == '__main__':
    rng = random.key(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Database name", default='data1.pkl')
    parser.add_argument("-s", "--smooth", help="smooth or sharp levelset", default='False')
    args = parser.parse_args()
    make_data(rng, args.filename, args.smooth)

    with open(path + args.filename, 'rb') as f:
        d = pickle.load(f)