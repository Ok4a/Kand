import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['finalData/laplace matrix/laplace_data_15.txt','finalData/laplace matrix/laplace_data_25.txt','finalData/laplace matrix/laplace_data_35.txt','finalData/laplace matrix/laplace_data_45.txt']
    # file_names = ['SaveData/super_sub/diag_shift_jacobi_laplace_sign_1.txt']
    lines_list = []
    lines_list.append({'train': [], 'test': [37]})
    lines_list.append({'train': [], 'test': [37]})
    lines_list.append({'train': [], 'test': [37]})
    lines_list.append({'train': [], 'test': [37]})
    method = 'sign'
    k = 0
    index_list = [0,1,2,3]
    all_lines = loadAll(file_names, lines_list, index_list)
    names_list = ['15','25','35','45']
    Perp_list = ['N(1,1)', 'N(1,0.6^2)', 'N(1,0.1^2)','N(1,0.3^2)']

    c = colours()
    # density plots
    line_type = 'test'
    data_type = 'iter'
    temp = [(0,0),(0,1),(1,0),(1,1)]
    # dense_label = densityLabels()
    fig = plt.figure(figsize=(10,7))
    axs = fig.subplots(2,2)
    fig.suptitle('Histogram of solver iteration counts\n for different parameters of data generation')


    axs_flat = axs.flatten()
    for index in index_list:
        axs_flat[index].grid(zorder = 0, linestyle='--')
        axs_flat[index].set_axisbelow(True)

        # fig = plt.figure(100+1+file_index*10)
        # plt.title(f'With {num_diag[coef]} super and sub diags in precond')
        # fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {seed}')
        axs_flat[index].set_title(f'$N={names_list[index]}$, ${Perp_list[index]}$')
        lines = all_lines[index][line_type][data_type][0]['no precond']
        if names_list[index] == '15':
            print(np.max(lines))
            lines = lines[lines< 3000]
        elif names_list [index] == '25':
            lines = lines[lines< 8000]
            
        print(np.mean(lines),np.median(lines), np.std(lines))
        axs_flat[index].boxplot(lines[lines> 0], orientation='horizontal',showmeans=False, notch=True, positions = [5], widths=3, manage_ticks = False)

        axs_flat[index].hist(lines[lines> 0], bins = 40)


    fig.supxlabel('Solver iteration counts')
    fig.supylabel('')





    plt.show()
