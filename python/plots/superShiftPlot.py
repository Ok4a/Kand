import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['finalData/num super/super_shift_jacobi_laplace_sign.txt']
    lines_list = []
    lines_list.append({'train': [26, 217,366,546,735,918], 'test': [174, 323,504,691,875,1070]})
    order = ['sign']
    seeds = 1

    num_super = [1,2,3,4,5,7]
    num_super = [1,2,3,4,5]
    which2use = range(len(num_super))
    all_lines = loadAll(file_names, lines_list, order)
    k=0


    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    fig = plt.figure(0, figsize=(6,4))
    axs = fig.subplots(1)
    # fig, axs = plt.subplots(2,2)
    fig.suptitle('Learning curves\n for a different amount of superdiagonals')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    for coef in range(len(num_super)):
        lines = all_lines[order[k]][line_type][data_type][coef]
        axs.plot(lines['iter2'],lines['mean'], label = f'{num_super[coef]}: mean', linestyle='-', color=c[coef])
        axs.plot(lines['iter2'], lines['median'], label = f'{num_super[coef]}: median', linestyle='--', color=c[coef])
        
        # plt.set_title(f'Improvement func: {order[k]}')
        # axs[temp[i_k]].legend()

        # axs[temp[i_k]].set_xlabel('Iteration count')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    axs.legend(title='Number super diagonals', ncol=2)
    # fig.legend(title='Num Super')


    axs.set_ylabel('Solver iteration')
    axs.set_xlabel('Learning steps')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # c = ['m','b','g','c','k','y', 'r']
    # fig = plt.figure(1)
    # # fig, axs = plt.subplots(2,2)
    
    # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of super diags in the precondtioner')
    # # temp = [(0,0),(0,1),(1,0),(1,1)]
    # Test_index = ['no precond', 'jacobi', 'best']
    
    # for k in which2use:
    #     for coef in range(len(num_super)):
    #         mean_list = []
    #         median_list = []
    #         lines = all_lines[order[k]][line_type][data_type][coef]
    #         for key in Test_index:
    #             mean_list.append(lines['mean'][key])
    #             median_list.append(lines['median'][key])
    #         # print((lines['mean']))
    #         # print((lines['median']))
    #         plt.plot(mean_list, label = f'{num_super[coef]}: mean', linestyle='-', color=c[coef])
    #         plt.plot(median_list, label = f'{num_super[coef]}: median', linestyle='--', color=c[coef])

    # plt.xticks(range(3),['Non','Jacobi','Best'])
    # fig.legend(loc='center right')
    # # fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')


    
    fig = plt.figure(45678, figsize=(6,4))
    axs = fig.subplots(1)
    lab = ['Non'] + num_super[which2use[0]:which2use[-1]+1] + ['Jacobi']
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
        axs.set_xlabel('Number of superdiagonals')
    fig.suptitle('Evaluation of learned preconditioner')
            
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab)
    # fig.supylabel('Solver iteration count')
    # fig.suptitle('Evaluation of learned preconditioner')
    # fig.supxlabel('Preconditioner')
    # fig.subplots_adjust(hspace=0.35)



    # # density plots
    # line_type = 'test'
    # data_type = 'stat'
    # for k in which2use:
    #     for coef in range(len(num_super)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+coef)
    #         plt.title(f'With {num_super[coef]} super diag in precond')
    #         fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
    #         non = all_lines[order[k]][line_type]['iter'][coef]['no precond']
    #         try:
    #             with_ = all_lines[order[k]][line_type]['iter'][coef]['best']
    #         except:
    #             with_ = all_lines[order[k]][line_type]['iter'][coef]['last']
    #         # without = all_lines[order[k]][line_type]['iter'][0][2]

    #         plt.hist([non,with_],bins=40, alpha = 1, label=['Non', 'Precond'], color=['c','m'])

    #         plt.boxplot((non, with_), orientation='horizontal', tick_labels = ['non', 'Precond'],showmeans=True, positions = [5,8], widths=1.5)
    #         plt.legend()
    #         plt.xlabel('Solver iteration')
        
    #     # axs[temp[seed]].set_title(f'Seed: {seed}')



    # # How many are better bar plot v Non
    # line_type = 'test'
    # b = []
    # labels = []
    # for k in which2use:
    #     # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    #     for coef in range(len(num_super)):
    #         try:
    #             b.append(all_lines[order[k]][line_type]['stat'][coef]['b']['best'])
    #         except:
    #             b.append(all_lines[order[k]][line_type]['stat'][coef]['bvn']['best'])

    #         labels.append(f'{num_super[coef]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    # plt.figure(42)
    # c = ['m','b','g','c']
    # plt.bar(labels,b,color=c)
    # # print(all_lines[order[k]][line_type]['iter'][0].keys())
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v Non')
    # plt.xlabel('Num super diags')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    for coef in range(len(num_super)):
        # print(all_lines[order[k]][line_type]['stat'][coef]['bvj'])
        try:
            b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['best'])
        except:
            b.append(all_lines[order[k]][line_type]['stat'][coef]['bvj']['last'])

        labels.append(f'{num_super[coef]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    fig = plt.figure(43, figsize=(6,4))
    axs= fig.subplots(1)
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    plt.bar(labels,b,color=c)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    plt.legend()
    fig.suptitle('The number of systems with lower\n solver iteration count compared with Jacobi')
    plt.xlabel('Number of superdiagonals')




    plt.show()