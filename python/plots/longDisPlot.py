import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *






if __name__ == '__main__':
   

    file_names = ['SaveData/Long_dis_test/par_shift_jacobi_laplace_sign.txt', 'SaveData/Long_dis_test/par_shift_jacobi_laplace_median.txt', 'SaveData/Long_dis_test/par_shift_jacobi_laplace_mean.txt']
    lines_list = []
    lines_list.append({'train': [24], 'test': [664]})
    lines_list.append({'train': [24], 'test': [424]})
    lines_list.append({'train': [24], 'test': [508]})
    # lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})
    order = ['sign', 'median', 'mean']
    seeds = 1
    learn = ['Without', 'With']
    all_lines = loadAll(file_names, lines_list, order)

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    fig = plt.figure(0)
    # fig, axs = plt.subplots(2,2)
    fig.suptitle('Improvement in training')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    for k in range(3):
        for index in range(seeds):
            lines = all_lines[order[k]][line_type][data_type][index]
            plt.plot(lines['iter2'],lines['mean'], label = f'mean: {order[k]}', linestyle='-', color=c[k])
            plt.plot(lines['iter2'], lines['median'], label = f'median: {order[k]}', linestyle='--', color=c[k])
           
            # plt.set_title(f'Improvement func: {order[k]}')
            # axs[temp[i_k]].legend()
    
            # axs[temp[i_k]].set_xlabel('Iteration count')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    
    fig.legend(loc='center right')
    plt.ylabel('Solver iteration')
    plt.xlabel('Learn iteration')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    fig = plt.figure(1)
    # fig, axs = plt.subplots(2,2)
    
    fig.suptitle('Improvement in Testing\n Long test With disimprovement in training')
    # temp = [(0,0),(0,1),(1,0),(1,1)]
    
    for k in range(3):
        # for learn_method in range(seeds):
        lines = all_lines[order[k]][line_type][data_type][0]
        print((lines['mean']))
        print((lines['median']))
        plt.plot(lines['mean'].values(), label = f'Mean: {order[k]}', linestyle='-', color=c[k])
        plt.plot(lines['median'].values(), label = f'Median: {order[k]}', linestyle='--', color=c[k])

    plt.xticks(range(3),['Non','Jacobi','Precond'])

    fig.legend(loc='center right')
    # fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')

    # density plots
    line_type = 'test'
    data_type = 'stat'
    for k in range(len(order)):
        # fig, axs = plt.subplots(2,2)
        fig = plt.figure(100+k)
        plt.title("With and without disimprovement in training")
        fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
        non = all_lines[order[k]][line_type]['iter'][0]['no precond']
        with_ = all_lines[order[k]][line_type]['iter'][0]['last']
        # without = all_lines[order[k]][line_type]['iter'][0][2]

        plt.hist([non,with_],bins=40, alpha = 1, label=['Non','with'], color=c[:2])

        plt.boxplot((non,with_),orientation='horizontal',tick_labels=['non', 'with'],showmeans=True, positions=[5,8], widths=1.5)
        plt.legend()
        plt.xlabel('Solver iteration')
        
        # axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot v Non
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for index in range(seeds):
            try:
                b.append(all_lines[order[k]][line_type]['stat'][index]['b']['last'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

            labels.append(f'{order[k]}: {learn[index]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    plt.figure(42)
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    plt.legend()
    plt.title('How many that are better in testing v Non')

     # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for index in range(seeds):
            b.append(all_lines[order[k]][line_type]['stat'][index]['bvj']['last'])
            labels.append(f'{order[k]}: {learn[index]}')
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