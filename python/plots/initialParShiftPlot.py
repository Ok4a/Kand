import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['finalData/initial par shift test/initial_par_shift_eris1176.txt', 'finalData/initial par shift test/initial_par_shift_fidap004.txt','finalData/initial par shift test/initial_par_shift_orsirr_1.txt', 'finalData/initial par shift test/initial_par_shift_sherman5.txt']
    lines_list = []
    lines_list.append({'train': [], 'test': [39]})
    lines_list.append({'train': [], 'test': [39]})
    lines_list.append({'train': [], 'test': [39]})
    lines_list.append({'train': [], 'test': [39]})
    method = 'sign'
    k = 0
    index_list = [0,1,2,3]
    all_lines = loadAll(file_names, lines_list, index_list)
    names_list = ['eris1176','fidap004','orsirr_1','sherman5']

    c = colours()
    # density plots
    line_type = 'test'
    data_type = 'iter'
    temp = [(0,0),(0,1),(1,0),(1,1)]
    # dense_label = densityLabels()
    fig = plt.figure(figsize=(10,7))
    axs = fig.subplots(2,2)
    fig.suptitle('Histogram of solver iteration counts')


    axs_flat = axs.flatten()
    for index in index_list:
        axs_flat[index].grid(zorder = 0, linestyle='--')
        axs_flat[index].set_axisbelow(True)

        # fig = plt.figure(100+1+file_index*10)
        # plt.title(f'With {num_diag[coef]} super and sub diags in precond')
        # fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {seed}')
        axs_flat[index].set_title(names_list[index])
        lines = all_lines[index][line_type][data_type][0]['last']
        if names_list[index] == 'eris1176':
            lines = lines[lines>0]
        elif names_list [index] == 'sherman5':
            lines = lines[lines>2300]

        axs_flat[index].boxplot(lines[lines> 0], orientation='horizontal',showmeans=False, notch=True, positions = [10], widths=3, manage_ticks = False)

        axs_flat[index].hist(lines[lines> 0], bins = 40)
        no_pre = all_lines[index][line_type][data_type][0]['no precond'][0]
        axs_flat[index].axvline(no_pre, color = 'k', label='No Preconditioning', ls='--')
        print(names_list[index], np.size(lines[lines < no_pre]))
        axs_flat[index].legend()


    fig.supxlabel('Solver iteration')
    fig.supylabel('')





    plt.show()
