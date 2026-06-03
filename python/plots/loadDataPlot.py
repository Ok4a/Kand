import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
from collections import defaultdict

import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

# now we can import the module in the parent
# directory.
import util

rcParams['savefig.directory'] = 'C:/Users/owkaa/OneDrive/Dokumenter/SDU/Kand/tex/plots'


def loadIterLines(file:str, line_num, line_type = 'test'):
    if line_type.lower() == 'test':
        lines = {}
    else:
        lines = []
    break_next = False
    prev_line = ''
    with open(file, mode='r') as file:
        for line_no, line in enumerate(file):
            if line_type.lower() == 'test' and line_no+1 > line_num:
                if line[-2:-1] == ')':
                    
                    line_key = prev_line.split(':')[0].lower()
                    lines[line_key] = np.array([int(x) if x[-1] != ',' else int(x[:-1]) for x in line[2:-2].split(', ')])
                if break_next:
                    break
                if line[0] in ['[','{']:
                    break_next = True

                    
            elif line_type.lower() == 'train' and line_no >= line_num:
                if line[0] == '	':
                    lines.append(np.array([int(x) if x[-1] != ',' else int(x[:-1]) for x in line[2:-2].split(', ')]))
                elif line[0:4] == 'Test':
                    break
            prev_line = line

    return lines




def loadstatLines(file:str, line_num, line_type = 'test'):

    if line_type.lower() == 'test':
        lines = defaultdict(dict)
    elif line_type.lower() == 'train':
    
        lines = defaultdict(list)
    else:
        raise Exception('Incorrect line type')

    with open(file, mode = 'r') as file:
        for line_no, line in enumerate(file):
            if line_type.lower() == 'test' and line_no + 1 > line_num:
                if line[0] in ['[','{']:
                    break
                elif line[0:6] in ['No pre', 'Jacobi', 'Last: ','Best: ']:
                    split_line = re.split(': |, | \n', line)
                    for i in range(len(split_line)):
                        try:
                            value = float(split_line[i])
                            lines[split_line[i-1].lower()][split_line[0].lower()] = value

                        except:
                            pass


            elif line_type.lower() == 'train' and line_no >= line_num:
                if line[0] == '(':
                    split_line = re.split('[(]|[)]: |: |, | \n|,\n', line)
                    lines['iter1'].append(int(split_line[1]))
                    lines['iter2'].append(int(split_line[2]))
                    for i in range(3, len(split_line) - 1, 2):
                        lines[split_line[i].lower()].append(float(split_line[i+1]))
                elif line[0:4].lower() == 'test':
                    break
    return lines



def loadLines(file,line_nums,line_type):
    lines={'stat':[], 'iter':[]}
    for ii in line_nums:
        lines['stat'].append(loadstatLines(file, ii, line_type))
        lines['iter'].append(loadIterLines(file, ii, line_type))
    return lines


def loadAll(files,lines_list, order):
    # lines[improvement function][train or test][iter or stat][seed]
    lines = defaultdict(dict)
    for ii in range(len(order)):
        lines[order[ii]]['test'] = loadLines(files[ii], lines_list[ii]['test'], line_type='test')
        lines[order[ii]]['train'] = loadLines(files[ii], lines_list[ii]['train'], line_type='train')

    return lines


def colours():
    c = ['tab:blue','tab:red','tab:green','tab:orange','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','indigo','seagreen']
    return c

def densityLabels():
    return ['No Precond', 'Learned', 'Jacobi']

def FigSize():
    return (10,7)



if __name__ == '__main__':
    # file_names = ['SaveData/shift_laplace_sign.txt', 'SaveData/shift_laplace_median.txt', 'SaveData/shift_laplace_mean.txt', 'SaveData/shift_laplace_jacobi.txt', 'SaveData/shift_Jacobi_laplace_sign.txt']
    file_names = ['finalData/learn laplace/shift_laplace_sign.txt', 'finalData/learn laplace/shift_laplace_median.txt', 'finalData/learn laplace/shift_laplace_mean.txt']

    lines_list = []
    lines_list.append({'train': [22,170,325,464],'test': [136,292,431,594]})
    # lines_list.append({'test': [137,290,426,586,738,884,1040], 'train': [23,168,320,457,616,768,914] })
    lines_list.append({'train': [22,81,126,175],'test': [48,93,142,183]})
    lines_list.append({'train': [22,75,138,217], 'test': [42,105,184,269]})
    # lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})
    # lines_list.append({'test': [102,211,363,530,810], 'train': [22,135,245,398,564]})

    order = ['sign', 'median', 'mean', 'jacobi', 'sign_jacobi_shift']
    order = ['SIGN', 'MEDIAN', 'MEAN']

    all_lines = loadAll(file_names, lines_list, order)

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    seeds = [0,1,2,3]
    c = colours()

    fig = plt.figure(1, figsize=FigSize())
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
    fig.suptitle('Learning curves\n for different improvement functions')
    

    for k in range(len(order)):
        for index in seeds:

            lines = all_lines[order[k]][line_type][data_type][index]
            axs_flat[index].plot(lines['iter2'],lines['mean'], label = f'{order[k]}: Mean', linestyle='-', color=c[k])
            axs_flat[index].plot(lines['iter2'], lines['median'], label = f'{order[k]}: Median', linestyle='--', color=c[k])
            # print(order[k], seed, len(lines['mean']))
            axs_flat[index].set_title(f'Seed: {index}')
            axs_flat[index].grid(zorder = 0, linestyle='--')
            # axs[temp[i_k]].legend()
    
            # axs_flat[seed].set_xlabel('Iteration count')
            # axs_flat[seed].set_ylabel('Solver count')
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='lower center', title='Improvement function: measure', ncols=6, bbox_to_anchor=(0.5, -0.005))
    fig.supxlabel('Learning Steps',y=0.07)
    fig.supylabel('Solver iterations')
    fig.subplots_adjust(bottom=0.13, top=0.9)


    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # fig = plt.figure(2, figsize=FigSize())
    # axs = fig.subplots(2,2)
    # fig.suptitle('Improvement of test data')
    # temp = [(0,0),(0,1),(1,0),(1,1)]

    # keys = ['no precond','last', 'jacobi']
    
    # for k in range(len(order)):
    #     for index in seeds:
    #         lines = all_lines[order[k]][line_type][data_type][index]
    #         means = []
    #         medians = []
    #         for key in keys:
    #             means.append(lines['mean'][key])
    #             medians.append(lines['median'][key])
    #         axs[temp[index]].plot(means, label = f'{order[k]}: Mean', linestyle='-', color=c[k])
    #         axs[temp[index]].plot(medians, label = f'{order[k]}: Median', linestyle='--', color=c[k])
    #         axs[temp[index]].set_title(f'Seed {index}')
    #         axs[temp[index]].set_xticks([0,1,2],['Non','Learned','Jacobi'])
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab, loc='lower center', title='Improvement function: measure', ncols=6, bbox_to_anchor=(0.5, -0.005))
    # # fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')
    # fig.supylabel('Solver iteration')
    # fig.supxlabel('Preconditioner',y=0.08)
    # fig.subplots_adjust(bottom=0.15, top=0.93)



    fig = plt.figure(45678, figsize=(10,5))
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
    x =np.arange(len(order)+2)
    for index in seeds:
        data ={}
        data['Mean'] = [all_lines[order[0]][line_type][data_type][index]['mean']['no precond']]
        data['Median'] = [all_lines[order[0]][line_type][data_type][index]['median']['no precond']]
        for k in range(len(order)):
            lines = all_lines[order[k]][line_type][data_type][index]
            data['Mean'].append(lines['mean']['last'])
            data['Median'].append(lines['median']['last'])
        data['Mean'].append(lines['mean']['jacobi'])
        data['Median'].append(lines['median']['jacobi'])
        width = 0.25
        mult  = 0
        for att, measure in data.items():
            offset = width * mult
            
            axs_flat[index].grid(zorder = 0, linestyle='--')
            rects = axs_flat[index].bar(x+offset+0.125,np.array(measure)-1500 , width, label = att, bottom=1500)
            # axs_flat[seed].bar_label(rects, padding=3)
            mult +=1

            axs_flat[index].set_title(f'Seed: {index}')
            axs_flat[index].set_axisbelow(True)
            axs_flat[index].set_xticks(x+width, ['Non']+order+['Jacobi'])
            
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='center right')
    fig.supylabel('Solver iteration count')
    fig.suptitle('Evaluation of learned preconditioner')
    fig.supxlabel('Improvement function')
    fig.subplots_adjust(hspace=0.35)



    

    # # density plots
    # line_type = 'test'
    # data_type = 'stat'
    # for k in range(len(order)):
    #     fig = plt.figure(60+k)
    #     axs = fig.subplots(2,2)
    #     fig.suptitle(f'Density of testing iterations with improve func: {order[k]}')
    #     # for seed in range(len(lines_list[k][line_type])):
    #     temp = [(0,0),(0,1),(1,0),(1,1)]
    #     for index in seeds:
    #         # plt.figure(seed)
    #         non = all_lines[order[k]][line_type]['iter'][index]['no precond']
    #         pre = all_lines[order[k]][line_type]['iter'][index]['last']

    #         n1 ,_,_ = axs[temp[index]].hist([non,pre],bins=40, alpha = 1, label=['No precond','Learned'], color=c[:2])

    #         axs[temp[index]].boxplot((non,pre),orientation='horizontal',tick_labels=['No precond', 'Learned'],showmeans=True, positions=[5,8], widths=1.5)
    #         axs[temp[index]].legend()
    #         axs[temp[index]].set_title(f'Seed: {index}')
    #         axs[temp[index]].set_xlabel('Solver Iteration')



    # # boxplots

    # line_type = 'test'
    # data_type = 'stat'
    # fig = plt.figure(34123, figsize=FigSize())
    # axs = fig.subplots(2,2)
    # axs_flat = axs.flatten()
    # for index in seeds:
    #     data = []
    #     names = []
    #     non = all_lines[order[k]][line_type]['iter'][index]['no precond']
    #     data.append(non)
    #     names.append('No preconditioning')
    #     for k in range(len(order)):
    #         pre = all_lines[order[k]][line_type]['iter'][index]['last']
    #         data.append(pre)
    #         names.append(order[k])


    #     axs_flat[index].boxplot(data,tick_labels=names,notch=True,showmeans=True)
    #     # axs[seed].legend()
    #     axs_flat[index].set_title(f'Seed: {index}')
    #     # axs_flat[seed].set_xlabel('Solver Iteration')
    #     axs_flat[index].grid(True, linestyle='--', axis='y')

    # fig.supylabel('Solver iteration count')
    # fig.supxlabel('Improvement function')


    # How many are better bar plot
    # line_type = 'test'
    # b = []
    # labels = []
    # for k in range(len(order)):
    #     # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    #     for index in seeds:
    #         # try:
    #             # b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
    #         # except:
    #         b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

    #         labels.append(f'{order[k]}: {index}')
    
    # fig = plt.figure(42, figsize=(10,4))
    # axs = fig.subplots(1,1)
    # axs.grid(zorder = 0, linestyle='--')
    # axs.bar(labels,b,color=c, zorder = 1)
    # plt.xticks(rotation=-45)
    # axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # axs.legend()
    # plt.title('How many learned preconditioned systems are better than using no preconditioner')
    # axs.set_axisbelow(True)
    # fig.subplots_adjust(bottom=0.17)


    # How many are better bar plot
    line_type = 'test'
    fig = plt.figure(423, figsize=(10,5))
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
   
    for seed in seeds:
        b = []
        labels = []
        for k in range(len(order)):
            # except:
            b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

            labels.append(f'{order[k]}')
        axs_flat[seed].bar(labels,b,color=c, zorder = 1)
        axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
        axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')

        axs_flat[seed].set_yticks(np.arange(0, 250+1, 50))
    
    
        axs_flat[seed].grid(zorder = 0, linestyle='--')
        axs_flat[seed].set_axisbelow(True)
        axs_flat[seed].set_title(f'Seed: {seed}')
    # plt.xticks(rotation=-45)
    # axs.legend()
    fig.suptitle('The number of systems with lower solver iteration count compared with no precondition')
    fig.supxlabel('Improvement function')
    fig.subplots_adjust(hspace=0.275)

    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, ncols=1, loc='center right')
    
#  # How many are better bar plot
#     line_type = 'test'
#     b = []
#     labels = []
#     for k in range(len(order)):
#         # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
#         for index in seeds:
#             # try:
#                 # b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
#             # except:
#             b.append(all_lines[order[k]][line_type]['stat'][index]['bvj']['last'])

#             labels.append(f'{order[k]}: {index}')
    
#     fig = plt.figure(43, figsize=(10,4))
#     axs = fig.subplots(1,1)
#     axs.grid(zorder = 0, linestyle='--')
#     axs.bar(labels,b,color=c, zorder = 1)
#     plt.xticks(rotation=-45)
#     axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
#     axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
#     axs.legend()
#     plt.title('How many learned preconditioned systems are better than using Jacobi preconditioning')
#     axs.set_axisbelow(True)
#     fig.subplots_adjust(bottom=0.17)

    line_type = 'test'
    fig = plt.figure(433, figsize=(10,5))
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
   
    for seed in seeds:
        b = []
        labels = []
        for k in range(len(order)):
            # except:
            b.append(all_lines[order[k]][line_type]['stat'][index]['bvj']['last'])

            labels.append(f'{order[k]}')
        axs_flat[seed].bar(labels,b,color=c, zorder = 1)
        axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
        axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')

        axs_flat[seed].set_yticks(np.arange(0, 250+1, 50))
    
    
        axs_flat[seed].grid(zorder = 0, linestyle='--')
        axs_flat[seed].set_axisbelow(True)
        axs_flat[seed].set_title(f'Seed: {seed}')
    # plt.xticks(rotation=-45)
    # axs.legend()
    fig.suptitle('The number of systems with lower solver iteration count compared with Jacobi')
    fig.supxlabel('Improvement function')
    fig.subplots_adjust(hspace=0.275)

    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, ncols=1, loc='center right')

    plt.show()