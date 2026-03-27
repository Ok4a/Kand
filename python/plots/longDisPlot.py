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
    c = ['m','b','g','c','k','y', 'r']
    fig = plt.figure(0)
    # fig, axs = plt.subplots(2,2)
    fig.suptitle('Improvement in training')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    for k in range(3):
        for seed in range(seeds):
            lines = all_lines[order[k]][line_type][data_type][seed]
            plt.plot(lines['iter1'],lines['mean'], label = f'mean: {order[k]}', linestyle='-', color=c[k])
            plt.plot(lines['iter1'], lines['median'], label = f'median: {order[k]}', linestyle='--', color=c[k])
           
            # plt.set_title(f'Improvement func: {order[k]}')
            # axs[temp[i_k]].legend()
    
            # axs[temp[i_k]].set_xlabel('Iteration count')
    # li, lab = fig.axes[0].get_legend_handles_labels()
    
    fig.legend(loc='center right')

    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    c = ['m','b','g','c','k','y', 'r']
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

        plt.hist([non,with_],bins=40, alpha = 1, label=['Non','with'], color=['c','m'])

        plt.boxplot((non,with_),orientation='horizontal',tick_labels=['non', 'with'],showmeans=True, positions=[5,8], widths=1.5)
        plt.legend()
        # axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot v Non
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for seed in range(seeds):
            try:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['bvn']['last'])

            labels.append(f'{order[k]}: {learn[seed]}')
    # print('daw')
    # print(all_lines['jacobi'][line_type]['stat'][0]['b'][0])
    # b.append(all_lines[order[k+1]][line_type]['stat'][0]['b'][0])
    # labels.append(f'{'jacobi'}')

    plt.figure(42)
    c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    c = ['m','b','g','c']
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
    plt.title('How many that are better in testing v Jacobi')




    plt.show()