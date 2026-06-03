import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['finalData/num par/par_shift_jacobi_laplace_sign.txt']
    lines_list = []
    lines_list.append({'train': [2196, 2387, 2584,2795,24,244,510,823,1151, 1491,1848], 'test': [2344, 2541, 2752,3027, 210,474,786,1113,1453, 1795, 2136]})
    order = ['sign']
    seeds = 1
    k = 0

    num_coef = [1,2,3,4,5,15,20,25,30,40,50]
    which2use = range(5,len(num_coef))
    which2use = range(5)
    all_lines = loadAll(file_names, lines_list, order)


    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    fig = plt.figure(0, figsize=(6,4))
    # fig = plt.figure(0, figsize=(10,5))
    # fig, axs = plt.subplots(2,2)
    axs = fig.subplots(1)
    fig.suptitle('Learning curves\n for a larger number of partitions')
    fig.suptitle('Learning curves\n for a smaller amount of partitions')
    
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    # for k in which2use:
    for coef in which2use:
        lines = all_lines[order[k]][line_type][data_type][coef]
        axs.plot(lines['iter2'],lines['mean'], label = f'{num_coef[coef]}: mean', linestyle='-', color=c[coef-which2use[0]])
        axs.plot(lines['iter2'], lines['median'], label = f'{num_coef[coef]}: median', linestyle='--', color=c[coef-which2use[0]])
        
        # plt.set_title(f'Improvement func: {order[k]}')
        # axs[temp[i_k]].legend()

        # axs[temp[i_k]].set_xlabel('Iteration count')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    axs.legend(title='Number of partitions: measure', ncol=2)
    # fig.legend(title='Seed: measure')
    fig.supylabel('Solver iteration count')
    fig.supxlabel('Learning steps')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # c = ['m','b','g','c','k','y', 'r']
    # fig = plt.figure(1)
    # # fig, axs = plt.subplots(2,2)
    
    # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of partitions in the precondtioner')
    # # temp = [(0,0),(0,1),(1,0),(1,1)]
    # Test_index = ['no precond', 'jacobi', 'best']
    
    # for k in which2use:
    #     for coef in range(len(num_coef)):
    #         lines = all_lines[order[k]][line_type][data_type][coef]
    #         mean_list = []
    #         median_list = []
    #         for key in Test_index:
    #             try:
    #                 mean_list.append(lines['mean'][key])
    #                 median_list.append(lines['median'][key])
    #             except:
    #                 mean_list.append(lines['mean']['last'])
    #                 median_list.append(lines['median']['last'])

    #         plt.plot(mean_list, label = f'{num_coef[coef]}: Mean', linestyle='-', color=c[coef])
    #         plt.plot(median_list, label = f'{num_coef[coef]}: Median', linestyle='--', color=c[coef])

    # plt.xticks(range(3),['No precond','Jacobi','Learned'])
    # fig.legend(loc='center right', title='Seed: measure')
    # plt.xlabel('Preconditioner')
    # plt.ylabel('Solver iteration')
    # # fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')


    fig = plt.figure(45678, figsize=(6,4))
    axs = fig.subplots(1)
    lab = ['Non'] + num_coef[which2use[0]:which2use[-1]+1] + ['Jacobi']
    x =np.arange(len(lab))
    # for seed in seeds:
    data ={}
    index = 0
    data['Mean'] = [all_lines[order[0]][line_type][data_type][index]['mean']['no precond']]
    data['Median'] = [all_lines[order[0]][line_type][data_type][index]['median']['no precond']]
    k = 0
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
        # rects = axs.bar(x+offset+0.125, np.array(measure)-1500 , width, bottom=1500)
        rects = axs.bar(x+offset+0.125, np.array(measure)-1500 , width,label = att, bottom=1500)
        # axs_flat[seed].bar_label(rects, padding=3)
        mult +=1

        # axs.set_title(f'Seed: {seed}')
        axs.set_axisbelow(True)
        axs.set_xticks(x+width, lab)
        axs.legend()
        axs.set_ylabel('Solver iteration count')
        axs.set_xlabel('Number of partitions')
    fig.suptitle('Evaluation of learned preconditioner')

            
    li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab)
    # fig.supylabel('Solver iteration count')
    # fig.suptitle('Evaluation of learned preconditioner')
    # fig.supxlabel('Preconditioner')
    # fig.subplots_adjust(hspace=0.35)




    # # density plots
    # line_type = 'test'
    # data_type = 'stat'
    # dense_label = densityLabels()
    # for k in which2use:
    #     for coef in range(len(num_coef)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+coef)
    #         plt.title(f'With {num_coef[coef]} partitions/coef in precond')
    #         fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
    #         non = all_lines[order[k]][line_type]['iter'][coef]['no precond']
    #         jacobi = all_lines[order[k]][line_type]['iter'][coef]['jacobi']
    #         try:
    #             with_ = all_lines[order[k]][line_type]['iter'][coef]['best']
    #         except:
    #             with_ = all_lines[order[k]][line_type]['iter'][coef]['last']
    #         # without = all_lines[order[k]][line_type]['iter'][0][2]

    #         plt.hist([non,jacobi,with_],bins=40, alpha = 1, label=dense_label, color=c[:3])

    #         plt.boxplot((non,jacobi, with_), orientation='horizontal', tick_labels = dense_label,showmeans=True, positions = [5,8, 11], widths=1.5)
    #         plt.legend(title = 'Preconditioner')
    #         plt.xlabel('Solver iteration')

    #     # axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot v Non
    # line_type = 'test'
    # b = []
    # labels = []
    # for coef in range(len(num_coef)):
    #     try:
    #         b.append(all_lines[order[k]][line_type]['stat'][coef]['bvn']['best'])
    #     except:
    #         b.append(all_lines[order[k]][line_type]['stat'][coef]['bvn']['last'])

    #     labels.append(f'{num_coef[coef]}')
    # # print('daw')
    # # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # # labels.append(f'{'jacobi'}')

    # plt.figure(42)
    # plt.bar(labels,b,color=c)
    # plt.xticks(rotation=-45)
    # # print(all_lines[order[k]][line_type]['iter'][0].keys())
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v Non')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    for coef in which2use:
        # print(all_lines[order[k]][line_type]['stat'][coef]['bvj'])
        try:
            b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['best'])
        except:
            b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['last'])

        labels.append(f'{num_coef[coef]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    fig = plt.figure(43, figsize=(6,4))
    axs = fig.subplots(1)
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    axs.bar(labels,b,color=c)
    # plt.xticks(rotation=-45)
    axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    axs.legend()
    fig.suptitle('The number of systems with lower\n solver iteration count compared with Jacobi')
    axs.set_xlabel('Number of partitions')




    plt.show()