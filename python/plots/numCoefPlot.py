import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['SaveData/num_coef/par_shift_jacobi_laplace_sign.txt']
    lines_list = []
    lines_list.append({'train': [2196, 2387, 2584,2795,24,244,510,823,1151, 1491,1848], 'test': [2344, 2541, 2752,3027, 210,474,786,1113,1453, 1795, 2136]})
    order = ['sign']
    seeds = 1

    num_coef = [1,2,3,4,5,15,20,25,30,40,50]
    num_coef = [1,2,3,4,5]
    which2use = range(5,10)
    all_lines = loadAll(file_names, lines_list, order)


    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = ['m','b','g','c','k','y', 'r','m','b','g']
    fig = plt.figure(0)
    # fig, axs = plt.subplots(2,2)
    fig.suptitle('Improvement in training\n Sign as improvement function for different number of partitions in the precondtioner')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    for k in which2use:
        for coef in range(len(num_coef)):
            lines = all_lines[order[k]][line_type][data_type][coef]
            plt.plot(lines['iter1'],lines['mean'], label = f'mean: {num_coef[coef]}', linestyle='-', color=c[coef])
            plt.plot(lines['iter1'], lines['median'], label = f'median: {num_coef[coef]}', linestyle='--', color=c[coef])
           
            # plt.set_title(f'Improvement func: {order[k]}')
            # axs[temp[i_k]].legend()
    
            # axs[temp[i_k]].set_xlabel('Iteration count')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(loc='center right')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # c = ['m','b','g','c','k','y', 'r']
    fig = plt.figure(1)
    # fig, axs = plt.subplots(2,2)
    
    fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of partitions in the precondtioner')
    # temp = [(0,0),(0,1),(1,0),(1,1)]
    
    for k in which2use:
        for coef in range(len(num_coef)):
            lines = all_lines[order[k]][line_type][data_type][coef]
            # print((lines['mean']))
            # print((lines['median']))
            plt.plot(lines['mean'].values(), label = f'Mean: {num_coef[coef]}', linestyle='-', color=c[coef])
            plt.plot(lines['median'].values(), label = f'Median: {num_coef[coef]}', linestyle='--', color=c[coef])

    plt.xticks(range(4),['Non','Jacobi','Last','Best'])
    fig.legend(loc='center right')
    # fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')

    # density plots
    line_type = 'test'
    data_type = 'stat'
    for k in which2use:
        for coef in range(len(num_coef)):
            # fig, axs = plt.subplots(2,2)
            fig = plt.figure(100+coef)
            plt.title(f'With {num_coef[coef]} partitions/coef in precond')
            fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
            non = all_lines[order[k]][line_type]['iter'][coef]['no precond']
            try:
                with_ = all_lines[order[k]][line_type]['iter'][coef]['best']
            except:
                with_ = all_lines[order[k]][line_type]['iter'][coef]['last']
            # without = all_lines[order[k]][line_type]['iter'][0][2]

            plt.hist([non,with_],bins=40, alpha = 1, label=['Non', 'Precond'], color=['c','m'])

            plt.boxplot((non, with_), orientation='horizontal', tick_labels = ['non', 'Precond'],showmeans=True, positions = [5,8], widths=1.5)
            plt.legend()
        # axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot v Non
    line_type = 'test'
    b = []
    labels = []
    for k in which2use:
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for coef in range(len(num_coef)):
            try:
                b.append(all_lines[order[k]][line_type]['stat'][coef]['b']['best'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][coef]['bvn']['last'])

            labels.append(f'{order[k]}: {num_coef[coef]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    plt.figure(42)
    c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    c = ['m','b','g','c']
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    # print(all_lines[order[k]][line_type]['iter'][0].keys())
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    plt.legend()
    plt.title('How many that are better in testing v Non')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    for k in which2use:
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for coef in range(len(num_coef)):
            # print(all_lines[order[k]][line_type]['stat'][coef]['bvj'])
            try:
                b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['best'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['last'])

            labels.append(f'{order[k]}: {num_coef[coef]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    plt.figure(43)
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    plt.legend()
    plt.title('How many that are better in testing v Jacobi')




    plt.show()