import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from loadDataPlot import *


# def loadIterLines(file:str, line_num, line_type = 'Test'):
#     lines = []
#     with open(file, mode='r') as file:
#         for line_no, line in enumerate(file):
#             if line_type.lower() == 'test' and line_no+1 in [line_num+2, line_num+5]:
#                 lines.append(np.array([int(x) for x in line[2:-2].split(', ')]))
#                 if len(lines) == 2:
#                     break
#             elif line_type.lower() == 'train' and line_no >= line_num:
#                 if line[0] == '	':
#                     lines.append(np.array([int(x) for x in line[2:-2].split(', ')]))
#                 elif line[0:4] == 'Test':
#                     break


#     return lines




# def loadstatLines(file:str, line_num, line_type = 'test'):
    
    
#     lines = defaultdict(list)

#     with open(file, mode='r') as file:
#         for line_no, line in enumerate(file):


#             if line_type.lower() == 'test' and line_no+1 in [line_num+1, line_num+4, line_num+7]:
#                 split_line = re.split(': |, | \n', line)
#                 for i in range(len(split_line)):
#                     try:
#                         value = float(split_line[i])
#                         lines[split_line[i-1].lower()].append(value)

#                     except:
#                         pass

                    
                   


#             elif line_type.lower() == 'train' and line_no >= line_num:
#                 if line[0] == '(':
#                     split_line = re.split('[(]|[)]: |: |, | \n', line)
#                     lines['iter1'].append(int(split_line[1]))
#                     lines['iter2'].append(int(split_line[2]))
#                     for i in range(3, len(split_line) - 1, 2):
#                         lines[split_line[i].lower()].append(float(split_line[i+1]))
#                 elif line[0:4].lower() == 'test':
#                     break
#     return lines



# def loadLines(file,line_nums,line_type):
#     lines={'stat':[], 'iter':[]}
#     for ii in line_nums:
#         lines['stat'].append(loadstatLines(file,ii,line_type))
#         lines['iter'].append(loadIterLines(file,ii,line_type))
#     return lines


# def loadAll(files,lines_list, order):
#     # lines[improvement function][train or test][iter or stat][seed]
#     lines = defaultdict(dict)
#     for ii in range(len(order)):
#         lines[order[ii]]['test'] = loadLines(files[ii], lines_list[ii]['test'], line_type='test')
#         lines[order[ii]]['train'] = loadLines(files[ii], lines_list[ii]['train'], line_type='train')

#     return lines







if __name__ == '__main__':
   

    file_names = ['SaveData/with_or_without_dis/par_shift_jacobi_laplace_sign.txt', 'SaveData/with_or_without_dis/par_shift_jacobi_laplace_median.txt', 'SaveData/with_or_without_dis/par_shift_jacobi_laplace_mean.txt', 'SaveData/shift_laplace_jacobi.txt']
    lines_list = []
    lines_list.append({'train': [24, 145], 'test': [110,281]})
    lines_list.append({'train': [24, 111], 'test': [76, 197]})
    lines_list.append({'train': [24, 119], 'test': [84, 241]})
    lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})
    order = ['sign', 'median', 'mean', 'jacobi']
    seeds = 2
    learn = ['Without', 'With']
    all_lines = loadAll(file_names, lines_list, order)

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = ['m','b','g','c','k','y', 'r']
    # fig = plt.figure(60)
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Improvement in training')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    g = [0,1,2,4]
    for i_k in range(3):
        k = g[i_k]
        for seed in range(2):
            lines = all_lines[order[k]][line_type][data_type][seed]
            axs[temp[i_k]].plot(lines['iter1'],lines['mean'], label = f'mean: {learn[seed]}', linestyle='-', color=c[seed])
            axs[temp[i_k]].plot(lines['iter1'], lines['median'], label = f'median: {learn[seed]}', linestyle='--', color=c[seed])
           
            axs[temp[i_k]].set_title(f'Improvement func: {order[k]}')
            # axs[temp[i_k]].legend()
    
            axs[temp[i_k]].set_xlabel('Iteration count')
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='center right')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    c = ['m','b','g','c','k','y', 'r']
    # fig = plt.figure(60)
    fig, axs = plt.subplots(2,2)
    
    fig.suptitle('Improvement in Testing\n With and Without disimprovement in training')
    temp = [(0,0),(0,1),(1,0),(1,1)]
    
    for k in range(len(order)-1):
        for learn_method in range(seeds):
            lines = all_lines[order[k]][line_type][data_type][learn_method]
            if len(lines['mean']) == 3:
                # print(lines['mean'][::2],learn[learn_method])
                axs[temp[k]].plot(lines['mean'].values(), label = f'Mean: {learn[learn_method]}', linestyle='-', color=c[learn_method])
                axs[temp[k]].plot(lines['median'].values(), label = f'Median: {learn[learn_method]}', linestyle='--', color=c[learn_method])
            # else:
            #     axs[temp[k]].plot(lines['mean'], label = f'{order[k]}', linestyle='-', color=c[learn_method])
            #     axs[temp[learn_method]].plot(lines['median'], label = f'{order[k]}', linestyle='--', color=c[learn_method])
            axs[temp[k]].set_title(f'{order[k]}')
            axs[temp[k]].set_xticks(range(3),['Non','Jacobi', 'precond'])
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='center right')
    fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')

    # density plots
    line_type = 'test'
    data_type = 'stat'
    for k in range(len(order)-1):
        # fig, axs = plt.subplots(2,2)
        fig = plt.figure(100+k)
        plt.title("With and without disimprovement in training")
        fig.suptitle(f'Density of testing iterations with improve func: {order[k]} and precond: Jacobi par Shift')
        non = all_lines[order[k]][line_type]['iter'][0]['no precond']
        with_ = all_lines[order[k]][line_type]['iter'][1]['last']
        without = all_lines[order[k]][line_type]['iter'][0]['last']

        plt.hist([non,with_,without],bins=40, alpha = 1, label=['Non','with', 'without'], color=['c','m','r'])

        plt.boxplot((non,with_,without),orientation='horizontal',tick_labels=['non', 'with','without'],showmeans=True, positions=[5,8, 11], widths=1.5)
        plt.legend()
        # axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)-1):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for seed in range(seeds):
            try:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['bvn']['last'])

            labels.append(f'{order[k]}: {learn[seed]}')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    b.append(all_lines[order[k+1]][line_type]['stat'][0]['b']['last'])
    labels.append(f'{'jacobi'}')

    plt.figure(42)
    c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    c = ['m','b','g','c']
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    plt.legend()
    plt.title('How many that are better in testing v no Precond')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)-1):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for seed in range(seeds):
            b.append(all_lines[order[k]][line_type]['stat'][seed]['bvj']['last'])
            labels.append(f'{order[k]}: {learn[seed]}')
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
    plt.title('How many that are better in testing v Jacobi precond')




    plt.show()