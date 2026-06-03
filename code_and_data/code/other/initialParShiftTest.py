import linearSolver as ls
import Precondition as prec
import numpy as np
import util
import scipy.sparse as sparse
# import matplotlib.pyplot as plt
import genLaplace
import multiprocessing as mp
from datetime import datetime
from itertools import product
from sys import argv
from configparser import ConfigParser
from time import perf_counter
from scipy.io import mmread

ebutton = mp.Event()

def runBiCGStab(A,b,M_inv, config:ConfigParser, stop_futher = True, other = None):
    np.seterr(all = 'raise')

    try:
        if ebutton.is_set() and stop_futher:
            return -1, 4
        
        # precond_type = config.get('Precondition', 'type')
        if M_inv is None:
            pass
        elif config.get('Precondition', 'diag') == 'eye':
            M_inv = sparse.eye(config.getint('Data', 'dim')) + M_inv
        elif config.get('Precondition', 'diag') == 'jacobi':
            M_inv = prec.Jacobi(A) + M_inv
        # if M_inv is None:
        #     pass
        # elif other == 'jacobi':
        #     M_inv = prec.Jacobi(A)
        # elif precond_type.lower() in ['par_shift', 'super_shift']:
        #     M_inv = sparse.eye(config.getint('Data', 'dim')) + M_inv
        # elif precond_type.lower() in ['par_shift_jacobi','super_shift_jacobi']:
        #     M_inv = prec.Jacobi(A) + M_inv
        

        if b is None:
            b = np.ones((config.getint('Data', 'dim'),1 ))


        _,_,k,flag = ls.BiCGSTAB(A, b, M_inv = M_inv, tol = config.getfloat('Solver', 'tol'), max_iter = config.getintOrNone('Solver', 'max_iteration'), extra_stop = (ebutton, stop_futher))


        if flag == 2:
            ebutton.set()
        return k, flag
    
    except FloatingPointError:
        ebutton.set()
        return -1, 3








if __name__ == '__main__':

    if len(argv) == 1:
        file_str = 'test_config.ini'
    else: 
        file_str = argv[1]
    
   


    config = util.getConfig(file_str)
    precond_num = config.getint('Data','amount')


    matrix_name = config.get('Other', 'matrix_name')
    the_matrix = mmread(f'matrixMarket/{matrix_name}.mtx.gz')
    precond_type = config.get('Precondition', 'type')


    config.set('Data', 'dim',str(np.shape(the_matrix)[0]))
    print(matrix_name, config.get('Data','dim'))

    preconds = []
    for ii in range(precond_num):
        rng = np.random.default_rng(ii)
        preconds.append(prec.diagShiftPrecond(config, rng=rng))





    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    # training_data = genLaplace.genLaplaceData(config = config, seed = rng_data)
    training_data = [the_matrix]

    

    with open(f'testData/initial_par_shift_{matrix_name}.txt', mode = 'a') as txt_file:
        txt_file.write(f'\n{datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
        txt_file.write(f'Config file: {file_str}\n')
        
        
        for key in config.sections():
            txt_file.write(f'{key}:\n')
            for op in config.options(key):
                txt_file.write(f'\t{op}: {config.get(key, op)}\n')


            

        txt_file.write(f'\n')
        txt_file.write('Train:\n')
        start = perf_counter()
        # coef_list, best_coef_list = laplaceDataML(training_data, file = txt_file, config = config, precond_class=precond_class)
        end = perf_counter()

        txt_file.write('\n')


        test_data = genLaplace.genLaplaceData(config = config, seed = rng_data)
        test_data = [the_matrix]

        txt_file.write('Test:\n')

        # No precond
        pool = mp.Pool()
        non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, None, config, False) for A in test_data]))
        txt_file.write(f'No precond: {util.statStr(non_precond_k_list)}, flag: {np.count_nonzero(non_pre_flag_list)} \n\t{non_precond_k_list+(0,)}\n\n')
        ebutton.clear()

        # last coef precond data
        pool = mp.Pool()
        # M_inv = prec.parShiftOff(config.getint('Data', 'dim'), coef_list)
        # M_inv = precond_class.makePrecond()
        start = perf_counter()
        final_k_list, final_flag_list = zip(*pool.starmap(runBiCGStab, [(the_matrix, None, precond_c.makePrecond(), config, False) for precond_c in preconds]))
        end = perf_counter()
        ebutton.clear()

        # # "best" coef precond data
        # pool = mp.Pool()
        # # M_inv = prec.parShiftOff(config.getint('Data', 'dim'), best_coef_list)
        # M_inv = precond_class.makePrecond('best')
        # best_k_list, best_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
        # ebutton.clear()
        

        # # Jacobi 
        # config.set('Precondition', 'type', 'par_shift_jacobi')
        # pool = mp.Pool()
        # M_inv = prec.parShiftOff(config.getint('Data', 'dim'), [0])
        # jacobi_k_list, jacobi_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
        # sign_JvN = util.betterWorse(jacobi_k_list, non_precond_k_list)
        # txt_file.write(f'Jacobi: {util.statStr(jacobi_k_list)}, BvN: {sign_JvN[1]}, WvN: {sign_JvN[-1]}, flag: {np.count_nonzero(jacobi_flag_list)} \n\t{jacobi_k_list}\n\n')
        # ebutton.clear()


        # Last coef write
        # sign_LvN = util.betterWorse(final_k_list, non_precond_k_list)
        # sign_LvJ = util.betterWorse(final_k_list, jacobi_k_list)
        txt_file.write(f'Last: {util.statStr(final_k_list)}, flag: {np.count_nonzero(final_flag_list)} \n\t{final_k_list}\n\t{final_flag_list}\n\n')

        # # "Best" coef write
        # sign_BvN = util.betterWorse(best_k_list, non_precond_k_list)
        # sign_BvJ = util.betterWorse(best_k_list, jacobi_k_list)
        # txt_file.write(f'Best: {util.statStr(best_k_list)}, BvN: {sign_BvN[1]}, WvN: {sign_BvN[-1]}, BvJ: {sign_BvJ[1]}, WvJ: {sign_BvJ[-1]}, flag: {np.count_nonzero(final_flag_list)} \n\t{best_k_list}\n')

            
        # txt_file.write(f'\nLast Coef List\n{precond_class.coef_dict}\n')
        # txt_file.write(f'\nBest Coef List\n{precond_class.best_coef}\n')

        txt_file.write(f'\nRun time: {end-start}\n')

        pool.close()
        pool.join()

    # n1 ,_,_ = plt.hist([non_precond_k_list,final_k_list,jacobi_k_list],bins=40, alpha = 1, label=['Non', config.get('Precondition', 'type'), 'jacobi'], color=['c', 'm', 'y'])
    # plt.legend()
    # plt.show()
