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

ebutton = mp.Event()

def runBiCGStab(A,b,M_inv, config:ConfigParser, stop_futher = True, other = None):
    np.seterr(all = 'raise')

    try:
        if ebutton.is_set() and stop_futher:
            return -1, 4
        
        precond_type = config.get('Precondition', 'type')
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
    precond_type = config.get('Precondition', 'type')
    if precond_type == 'diag_shift':
        precond_class = prec.diagShiftPrecond(config)
    elif precond_type == 'rand_entry':
        precond_class = prec.randEntryPrecond(config)
    else:
        raise Exception(f'"{precond_type}" preconditioner not defined')






    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    print('data1')

    training_data = genLaplace.genLaplaceData(config = config, seed = rng_data)
    del training_data

    

    with open(f'Data/{file_str[:-4]}_data.txt', mode = 'a') as txt_file:
        txt_file.write(f'\n{datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
        txt_file.write(f'Config file: {file_str}\n')
        
        
        for key in config.sections():
            txt_file.write(f'{key}:\n')
            for op in config.options(key):
                txt_file.write(f'\t{op}: {config.get(key, op)}\n')



        print('data2')
        test_data = genLaplace.genLaplaceData(config = config, seed = rng_data)

        txt_file.write('Test:\n')


        print('No')
        # No precond
        pool = mp.Pool()
        no_start = perf_counter()
        non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, None, config, False) for A in test_data]))
        no_end = perf_counter()
        txt_file.write(f'No precond: {util.statStr(non_precond_k_list)}, flag: {np.sum(non_pre_flag_list)}, time: {no_end-no_start} \n\t{non_precond_k_list}\n\n')
        ebutton.clear()

        print('pre')
        # last coef precond data
        pool = mp.Pool()
        M_inv = precond_class.makePrecond()
        pre_start = perf_counter()
        final_k_list, final_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
        pre_end = perf_counter()
        ebutton.clear()


        print('Jacobi')
        # Jacobi 
        config.set('Precondition', 'type', 'par_shift_jacobi')
        pool = mp.Pool()
        M_inv = prec.parShiftOff(config.getint('Data', 'dim'), [0])
        jacobi_start = perf_counter()
        jacobi_k_list, jacobi_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
        jacobi_end = perf_counter()
        sign_JvN = util.betterWorse(jacobi_k_list, non_precond_k_list)
        txt_file.write(f'Jacobi: {util.statStr(jacobi_k_list)}, BvN: {sign_JvN[1]}, WvN: {sign_JvN[-1]}, flag: {np.sum(jacobi_flag_list)}, time: {jacobi_end-jacobi_start} \n\t{jacobi_k_list}\n\n')
        ebutton.clear()


        # Last coef write
        sign_LvN = util.betterWorse(final_k_list, non_precond_k_list)
        sign_LvJ = util.betterWorse(final_k_list, jacobi_k_list)
        txt_file.write(f'Last: {util.statStr(final_k_list)}, BvN: {sign_LvN[1]}, WvN: {sign_LvN[-1]}, BvJ: {sign_LvJ[1]}, WvJ: {sign_LvJ[-1]}, flag: {np.sum(final_flag_list)}, time: {pre_end-pre_start} \n\t{final_k_list}\n\n')
            
        txt_file.write(f'\nLast Coef List\n{precond_class.coef_dict}\n')

        pool.close()
        pool.join()

    # n1 ,_,_ = plt.hist([non_precond_k_list,final_k_list,jacobi_k_list],bins=40, alpha = 1, label=['Non', config.get('Precondition', 'type'), 'jacobi'], color=['c', 'm', 'y'])
    # plt.legend()
    # plt.show()
