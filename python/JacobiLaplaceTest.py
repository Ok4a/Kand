import linearSolver as ls
import Precondition as prec
import numpy as np
import util
import genLaplace
from multiprocessing import Pool
from sys import argv
from configparser import ConfigParser
import percentParShiftLaplace as strUtil


def runBiCGStab(A,b,pre, config:ConfigParser):
    b = np.ones((config.getint('Data', 'dim'),1 ))
    if pre:
        M_inv = prec.Jacobi(A)
    else:
        M_inv = None
    _,_,k,flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = config.getfloat('Solver', 'tol'), max_iter=config.getintOrNone('Solver', 'max_iteration'))
    return k, flag






if __name__ == '__main__':

    if len(argv) == 1:
        file_str = 'config/config_mean_0.ini'
    else:
        file_str = argv[1]
    for seed in range(7):
        config = util.getConfig(file_str)
        rng_data = np.random.default_rng(seed)
        _ = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)
        test_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)
        with open(f'testData/shift_laplace_jacobi.txt', mode = 'a') as txt_file:
            txt_file.write(f'\nJacobi: \nSeed: {seed}\n\n')
            txt_file.write('Test:\n')
            with Pool() as pool:
                non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None ,False, config) for A in test_data]))
                txt_file.write(f'No precond: {strUtil.statStr(non_precond_k_list)}, flag: {np.sum(non_pre_flag_list)} \n\t{non_precond_k_list}\n\n')

                jacobi_k_list, jacobi_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, True, config) for A in test_data]))

                sign = strUtil.betterWorse(jacobi_k_list, non_precond_k_list)

                txt_file.write(f'Last: {strUtil.statStr(jacobi_k_list)}, B: {sign[1]}, W: {sign[-1]}, flag: {np.sum(jacobi_flag_list)} \n\t{jacobi_k_list}\n')

            # print(strUtil.betterWorseStr(jacobi_k,non_k))