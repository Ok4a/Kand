import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *

if __name__ == '__main__':
   

    # file_names = ['finalData/disimprovment/par_shift_jacobi_laplace_sign.txt', 'finalData/disimprovment/par_shift_jacobi_laplace_median.txt', 'finalData/disimprovment/par_shift_jacobi_laplace_mean.txt', 'SaveData/shift_laplace_jacobi.txt']
    file_names = ['finalData/disimprovment/par_shift_jacobi_laplace_sign.txt']
    lines_list = []
    lines_list.append({'train': [24,986, 139, 312], 'test': [104,1232,275,952]})
    # lines_list.append({'train': [24, 111], 'test': [76, 197]})
    # lines_list.append({'train': [24, 119], 'test': [84, 241]})
    # lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})
    # order = ['sign', 'median', 'mean', 'jacobi']
    order = ['sign']

    learn = ['Without (100)','Without (500)', 'With (100)', 'With (500)']
    which2use = range(len(learn))
    # seeds = len(learn)
    # seeds = 8
    all_lines = loadAll(file_names, lines_list, order)

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    fig = plt.figure(60, figsize=(10,4))
    axs = fig.subplots(1)
    fig.suptitle('Learning curves')
    axs.set_axisbelow(True)
    
    for index in which2use:
        lines = all_lines[order[0]][line_type][data_type][index]
        axs.plot(lines['iter2'],lines['mean'], label = f'{learn[index]}: mean', linestyle='-', color=c[index], zorder=100-index)
        axs.plot(lines['iter2'], lines['median'], label = f'{learn[index]}: median', linestyle='--', color=c[index], zorder=100-index)
        axs.legend(title='')
        
        # axs.set_title(f'Improvement func: {order[k]}')
        # axs[temp[i_k]].legend()

        axs.set_xlabel('Learning steps')
        axs.set_ylabel('Solver iterations')
    axs.grid(zorder = 0, linestyle='--')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab, loc='center right')


    # # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # # fig = plt.figure(60)
    # fig, axs = plt.subplots(2,2)
    
    # fig.suptitle('Improvement in Testing\n With and Without disimprovement in training')
    # temp = [(0,0),(0,1),(1,0),(1,1)]
    
    # for k in range(len(order)-1):
    #     for learn_method in range(seeds):
    #         lines = all_lines[order[k]][line_type][data_type][learn_method]
    #         if len(lines['mean']) == 3:
    #             # print(lines['mean'][::2],learn[learn_method])
    #             axs[temp[k]].plot(lines['mean'].values(), label = f'Mean: {learn[learn_method]}', linestyle='-', color=c[learn_method])
    #             axs[temp[k]].plot(lines['median'].values(), label = f'Median: {learn[learn_method]}', linestyle='--', color=c[learn_method])
    #         # else:
    #         #     axs[temp[k]].plot(lines['mean'], label = f'{order[k]}', linestyle='-', color=c[learn_method])
    #         #     axs[temp[learn_method]].plot(lines['median'], label = f'{order[k]}', linestyle='--', color=c[learn_method])
    #         axs[temp[k]].set_title(f'{order[k]}')
    #         axs[temp[k]].set_xticks(range(3),['Non','Jacobi', 'precond'])
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab, loc='center right')
    # fig.text(0.02, 0.34, 'Solver iteration', rotation='vertical')



    
    fig = plt.figure(45678, figsize=(6,4))
    axs = fig.subplots(1)
    
    tick_list = ['Non']+learn+['Jacobi']
    x = np.arange(len(tick_list))
    k = 0
    data ={}
    print(all_lines[order[0]][line_type][data_type][index]['mean'])
    data['Mean'] = [all_lines[order[0]][line_type][data_type][index]['mean']['no precond']]

    data['Median'] = [all_lines[order[0]][line_type][data_type][index]['median']['no precond']]
    for index in which2use:
        lines = all_lines[order[k]][line_type][data_type][index]
        data['Mean'].append(lines['mean']['last'])
        data['Median'].append(lines['median']['last'])
    data['Mean'].append(lines['mean']['jacobi'])
    data['Median'].append(lines['median']['jacobi'])
    width = 0.25
    mult  = 0
    for att, measure in data.items():
        offset = width * mult
        
        axs.grid(zorder = 0, linestyle='--')
        rects = axs.bar(x+offset+0.125,np.array(measure)-1500 , width, label = att, bottom=1500)
        # axs_flat[seed].bar_label(rects, padding=3)
        mult +=1

        # axs.set_title(f'Preconditioner evaluation')
        axs.set_axisbelow(True)
        axs.set_xticks(x+width, tick_list)
        axs.legend()
        axs.set_xlabel('')
        axs.set_ylabel('Solver iteration')
    fig.suptitle('Evaluation of learned preconditioner')
    plt.xticks(rotation=-15)
    print(data['Mean'])
    print(data['Median'])
            
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.supylabel('Solver iteration count')
    # fig.suptitle('Evaluation of learned preconditioner')
    # fig.supxlabel('Preconditioner')
    # fig.subplots_adjust(hspace=0.35)


    # # density plots
    # line_type = 'test'
    # data_type = 'stat'
    # for k in range(len(order)-1):
    #     # fig, axs = plt.subplots(2,2)
    #     fig = plt.figure(100+k)
    #     plt.title("With and without disimprovement in training")
    #     fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
    #     non = all_lines[order[k]][line_type]['iter'][0]['no precond']
    #     with_ = all_lines[order[k]][line_type]['iter'][1]['last']
    #     without = all_lines[order[k]][line_type]['iter'][0]['last']

    #     plt.hist([non,with_,without],bins=40, alpha = 1, label=['No precond','with', 'without'], color=c[:3])

    #     plt.boxplot((non,with_,without),orientation='horizontal',tick_labels=['No precond', 'with','without'],showmeans=True, positions=[5, 8, 11], widths=1.5)
    #     plt.legend()
    #     plt.xlabel('Solver iteration')
    #     # axs[temp[seed]].set_title(f'Seed: {seed}')



    # # How many are better bar plot
    # line_type = 'test'
    # b = []
    # labels = []
    # for k in range(len(order)-1):
    #     # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    #     for index in range(seeds):
    #         try:
    #             b.append(all_lines[order[k]][line_type]['stat'][index]['b']['last'])
    #         except:
    #             b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

    #         labels.append(f'{order[k]}: {learn[index]}')
    # # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b']['last'])
    # labels.append(f'{'jacobi'}')

    # plt.figure(42)
    # plt.bar(labels,b,color=c)
    # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v no Precond')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    # for k in range(len(order)-1):
    # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    for index in range(len(learn)):
        b.append(all_lines[order[k]][line_type]['stat'][index]['bvj']['last'])
        labels.append(f'{learn[index]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')
    print(b)

    fig = plt.figure(43, figsize=(6,4))
    axs = fig.subplots(1)
    plt.bar(labels,b,color=c)
    # plt.xticks(rotation=-45)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    plt.legend()
    plt.suptitle('The number of systems with lower\n solver iteration count compared with Jacobi')
    axs.grid(zorder = 0, linestyle='--')

    axs.set_axisbelow(True)



    plt.show()