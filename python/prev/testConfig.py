import linearSolver as ls
import Precondition as prec
import numpy as np
import util
from scipy.io import mmread
import matplotlib.pyplot as plt
import time
import genLaplace
from multiprocessing import Pool
from datetime import datetime
from itertools import product
from sys import argv
from configparser import ConfigParser

def runBiCGStab(A,b,M_inv, config:ConfigParser):
    if b is None:
        b = np.ones((config.getint('Data', 'dim'),1 ))
    _,_,k,flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = config.getfloat('Solver', 'tol'), max_iter=config.getintOrNone('Solver', 'max_iteration'))
    return k, flag


def statStr(k_list):
    mean = np.mean(k_list, axis=0)
    median = np.median(k_list, axis=0)
    std = np.std(k_list, axis=0)
    prt1 = np.quantile(k_list,q=(1-0.68)/2, axis=0)
    prt2 = np.quantile(k_list,q=1-(1-0.68)/2, axis=0)

    return f'Mean: {np.round(mean,3)}, median: {np.round(median,3)}, std: {np.round(std,3)}, percent: {np.round((prt1+prt2)/2,3)}'


if __name__ == '__main__':

    if len(argv) == 1:
        file_str = 'config/config_mean_0.ini'
    else:
        file_str = argv[1]


    config = util.getConfig(file_str)



    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)

    with Pool() as pool:
        k_list, flag_list = zip(*pool.starmap(runBiCGStab,[(A, None ,None, config) for A in data]))
        print(statStr(k_list))
        print(np.sum(flag_list))

    

