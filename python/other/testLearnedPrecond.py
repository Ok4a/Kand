import linearSolver as ls
import Precondition as prec
import numpy as np
import util
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import genLaplace
import multiprocessing as mp
from datetime import datetime
from itertools import product
from sys import argv
from configparser import ConfigParser
from percentParShiftLaplace import runBiCGStab
import re






def passCoef(str_list):
    coef_list = []
    for string in str_list:
        temp = re.split(' |\\[|\\]\n|\n',string)
        for each in temp:
            if len(each) > 0:
                # print(each)
                coef_list.append(float(each))

    return coef_list






if __name__ == '__main__':



    data_file = 'SaveData/shift_Jacobi_laplace_sign.txt'
    line = 224
    found_coef = False
    coef_str_list = []
    with open(data_file) as file:
        for line_no, line_str in enumerate(file):
            if line_no == line:
                config_file = re.split('\n| ', line_str)
                config = util.getConfig(config_file[2])
            elif line_no >= line and (line_str[0] == '[' or found_coef):
                if line_str[0:1] == '\n':
                    break
                found_coef = True
                coef_str_list.append(line_str)

    
    coef_list = passCoef(coef_str_list)
    # print(coef_list)

    rng_data = np.random.default_rng(config.getint('Learn', 'seed'))

    _ = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)

    test_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)


    pool = mp.Pool()
    # non_k_list, Non_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, None, config, False) for A in test_data]))
    non_k_list = [1901, 3242, 1970, 2606, 2480, 1510, 1504, 2925, 664, 2319, 856, 2220, 1363, 2319, 1325, 4827, 1313, 2096, 1194, 1774, 2190, 2501, 1375, 1792, 1349, 1398, 7071, 2478, 2014, 1657, 3825, 1849, 2918, 2994, 2696, 1641, 4176, 3921, 2067, 1633, 1859, 2163, 1576, 2795, 1368, 4006, 2575, 5949, 3558, 2260, 3505, 1931, 2125, 1802, 1314, 2499, 2342, 2010, 1292, 2769, 1077, 2388, 1857, 1336, 1918, 1719, 1875, 8409, 3822, 1417, 4913, 5185, 2121, 3704, 1111, 2289, 1722, 1619, 3505, 1889, 1874, 1287, 4732, 2901, 3298, 1277, 2007, 2789, 3169, 6286, 1411, 3915, 3148, 1288, 1650, 1295, 3210, 4860, 1849, 1443, 4030, 2698, 2834, 5245, 3253, 2309, 2224, 3390, 8573, 2377, 3236, 1841, 1978, 2624, 2162, 1987, 2943, 1167, 2541, 1232, 5631, 874, 2907, 5726, 2298, 1303, 1913, 4272, 1071, 1069, 2986, 3122, 2957, 2538, 2146, 2613, 2090, 2819, 2732, 2879, 2632, 973, 2948, 4606, 2225, 3181, 1384, 3844, 3129, 1682, 2275, 4859, 1412, 617, 927, 4918, 4417, 2912, 2782, 987, 2932, 879, 1589, 2441, 1461, 2524, 2598, 3604, 3241, 3337, 3314, 3609, 2600, 1859, 1034, 939, 5251, 6602, 5194, 1968, 1288, 1885, 1703, 1711, 2798, 3334, 8692, 1348, 1204, 1573, 1784, 3596, 2448, 2041, 2242, 2833, 4420, 3819, 5621, 1965, 3047, 1288, 3785, 3683, 2721, 2180, 2216, 1526, 6115, 2032, 3099, 3378, 5003, 2367, 2281, 4647, 3217, 3277, 2048, 1838, 2117, 2189, 1257, 9233, 1822, 4768, 2465, 1522, 956, 1906, 3059, 1013, 6424, 1769, 2714, 2593, 3989, 1273, 2700, 1967, 3061, 4424, 3642, 1276, 1907, 1480, 2681, 3400, 1626, 1822]


    # M_inv = prec.parShiftOff(config.getint('Data', 'dim'), [0])
    # j_k_list, j_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, 'None', config, False, 'jacobi') for A in test_data]))
    j_k_list = [1571, 3016, 1767, 1816, 2046, 1238, 1214, 2694, 543, 2505, 629, 1427, 778, 2288, 975, 5364, 1051, 1690, 854, 1178, 2109, 2094, 998, 1559, 1071, 1293, 8720, 2075, 1426, 1244, 3769, 1287, 1981, 2437, 2531, 1952, 4109, 3348, 1591, 1420, 1824, 1628, 1439, 2789, 946, 2708, 2039, 5179, 2357, 2445, 2679, 1425, 1851, 1623, 1129, 1934, 1898, 1274, 904, 2036, 863, 1771, 1149, 1124, 1642, 1599, 1591, 5997, 3253, 939, 4541, 4175, 1706, 4029, 922, 2048, 1408, 1279, 2797, 1472, 1586, 886, 4381, 2220, 2967, 999, 1613, 2527, 2518, 3805, 995, 4094, 2633, 917, 1214, 1008, 3165, 4532, 1589, 1148, 4295, 2174, 2203, 4878, 1970, 1723, 2057, 3021, 11046, 2048, 2644, 1584, 1634, 2613, 1883, 1777, 2048, 835, 1798, 915, 4516, 650, 3060, 3766, 1768, 1117, 1588, 3976, 1005, 702, 2350, 2442, 2181, 1884, 1474, 2385, 2070, 2309, 2146, 2840, 2566, 785, 2183, 4294, 2428, 1963, 1187, 3547, 2670, 1327, 2445, 5342, 1180, 458, 868, 3121, 2909, 2312, 2454, 755, 2358, 692, 1094, 1589, 1150, 2548, 1975, 3295, 3284, 3241, 3208, 3868, 2396, 1736, 824, 738, 4471, 6205, 6248, 1559, 1103, 1391, 1480, 1041, 2324, 2313, 7660, 1125, 802, 1285, 1374, 2743, 1809, 1534, 1727, 1980, 5490, 2297, 5322, 1769, 2999, 781, 6091, 2696, 2832, 1624, 2334, 1095, 5007, 2268, 2913, 2854, 4619, 1994, 1840, 4118, 2290, 2898, 1245, 1526, 1754, 2069, 1113, 4958, 1371, 4162, 1845, 1143, 654, 1349, 2473, 783, 4680, 1352, 1926, 2779, 2748, 1089, 2564, 2072, 2828, 3893, 3001, 1007, 1549, 1385, 1985, 3380, 1836, 1609]

    M_inv = prec.parShiftOff(config.getint('Data', 'dim'), coef_list)
    Prec_k_list, Prec_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, M_inv, config, False) for A in test_data]))
    pool.close()
    pool.join()

    print()
    print(util.statStr(non_k_list))
    print()
    print(util.statStr(j_k_list))
    print()

    
    print(util.statStr(Prec_k_list))
    print()
    print(util.betterWorseStr(Prec_k_list, non_k_list))

    print()
    print(util.betterWorseStr(Prec_k_list, j_k_list))
    
    # if len(argv) == 1:
    #     file_str = 'test_config.ini'
    # else:
    #     file_str = argv[1]


    # config = util.getConfig(file_str)




    # rng_data = np.random.default_rng(config.getint('Learn', 'seed'))
    # training_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)


    

    # with open(f'testData/{config.get('Precondition', 'type')}_laplace_{config.get("Learn", "method")}.txt', mode = 'a') as txt_file:
    #     txt_file.write(f'\n{datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}\n')
    #     txt_file.write(f'Config file: {file_str}\n')
        
        
    #     for key in config.sections():
    #         txt_file.write(f'{key}:\n')
    #         for op in config.options(key):
    #             txt_file.write(f'\t{op}: {config.get(key, op)}\n')


            

    #     txt_file.write(f'\n')
    #     txt_file.write('Train:\n')
    #     coef_list, coef_list_initial = laplaceDataML(training_data, file = txt_file, config = config)

    #     txt_file.write('\n')


    #     test_data = genLaplace.genLaplaceData(N = config.getint('Data', '1D_laplace_size'), param = config.getfloatList('Data', 'params'), data_count = config.getint('Data', 'amount'), seed = rng_data)

    #     txt_file.write('Test:\n')

    #     pool = mp.Pool()
    #     non_precond_k_list, non_pre_flag_list = zip(*pool.starmap(runBiCGStab,[(A, None, None, config, False) for A in test_data]))
    #     txt_file.write(f'No precond: {util.statStr(non_precond_k_list)}, flag: {np.sum(non_pre_flag_list)} \n\t{non_precond_k_list}\n\n')
    #     ebutton.clear()

        
    #     pool = mp.Pool()
    #     M_inv = prec.parShift(config.getint('Data', 'dim'), coef_list)
    #     final_k_list, final_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
    #     ebutton.clear()
        

    #     config.set('Precondition', 'type', 'par_shift_jacobi')
    #     pool = mp.Pool()
    #     M_inv = prec.parShift(config.getint('Data', 'dim'), [0])
    #     jacobi_k_list, jacobi_flag_list = zip(*pool.starmap(runBiCGStab, [(A, None, M_inv, config, False) for A in test_data]))
    #     ebutton.clear()
        
    #     sign_JvN = util.betterWorse(jacobi_k_list, non_precond_k_list)

        
    #     txt_file.write(f'Jacobi: {util.statStr(jacobi_k_list)},, BvN: {sign_JvN[1]}, WvN: {sign_JvN[-1]}, flag: {np.sum(jacobi_flag_list)} \n\t{jacobi_k_list}\n\n')



    #     sign_PvN = util.betterWorse(final_k_list, non_precond_k_list)
    #     sign_PvJ = util.betterWorse(final_k_list, jacobi_k_list)

    #     txt_file.write(f'Last: {util.statStr(final_k_list)}, BvN: {sign_PvN[1]}, WvN: {sign_PvN[-1]}, BvJ: {sign_PvJ[1]}, WvJ: {sign_PvJ[-1]}, flag: {np.sum(final_flag_list)} \n\t{final_k_list}\n')
    #     # txt_file.write(f'Last: {util.statStr(final_k_list)}, B: {sign[1]}, W: {sign[-1]}, flag: {np.sum(final_flag_list)} \n\t{final_k_list}\n')

            
            
    #     txt_file.write(f'\n{coef_list}\n')


    #     pool.close()
    #     pool.join()

    # n1 ,_,_ = plt.hist([non_precond_k_list,final_k_list,jacobi_k_list],bins=40, alpha = 1, label=['Non', config.get('Precondition', 'type'), 'jacobi'], color=['c', 'm', 'y'])
    # plt.legend()
    # plt.show()