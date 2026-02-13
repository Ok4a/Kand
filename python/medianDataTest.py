import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt
import time
import genLaplace
from multiprocessing import Pool


def runBiCGStab(A,b,M_inv,tol):

    _,_,k,flag = ls.BiCGSTAB(A,b,M_inv=M_inv,tol=tol)
    


    return [k, flag]







if __name__ == '__main__':

    config = {}

    config['step_range'] = 0.1
    config['train_iteration_count'] = 50
    config['num_coef'] = 20
    config['data_seed'] = None
    config['change_seed'] = None
    config['1D_laplace_size'] = 40
    config['data_set_size'] = 25
    config['data_set_param'] = [1, 0.25]
    config['CG_tol'] = 1e-10
    config['dim'] = config['1D_laplace_size'] * config['1D_laplace_size']



    rng_data = np.random.default_rng(config['data_seed'])
    training_data = genLaplace.genLaplaceData(N = config['1D_laplace_size'], param = config['data_set_param'], data_count = config['data_set_size'], seed=rng_data)

    median_list = []
    with Pool(12) as pool:


        for ii in range(100):
            print(ii,end='\r')
            test_data = genLaplace.genLaplaceData(N = config['1D_laplace_size'], param = config['data_set_param'], data_count = config['data_set_size'], seed=rng_data)

            k_list = pool.starmap(runBiCGStab,[(A, np.ones((config['dim'], 1)),None,config['CG_tol']) for A in test_data])
            median_list.append(float(np.median(k_list,axis=0)[0]))
    
    print(median_list)
    print(np.min(median_list))
    print(np.max(median_list))
    print(np.mean(median_list))
    print(np.var(median_list))
    print(np.std(median_list))
    print(np.median(median_list))