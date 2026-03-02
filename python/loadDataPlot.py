import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def loadIterLines(file:str, line_num, line_type = 'Test'):
    lines = []
    with open(file, mode='r') as file:
        for line_no, line in enumerate(file):
            if line_type.lower() == 'test' and line_no+1 in [line_num+2, line_num+5]:
                lines.append(np.array([int(x) for x in line[2:-2].split(', ')]))
                if len(lines) == 2:
                    break
            elif line_type.lower() == 'train' and line_no >= line_num:
                if line[0] == '	':
                    lines.append(np.array([int(x) for x in line[2:-2].split(', ')]))
                elif line[0:4] == 'Test':
                    break


    return lines




def loadstatLines(file:str, line_num, line_type = 'test'):
    
    
    lines = defaultdict(list)

    with open(file, mode='r') as file:
        for line_no, line in enumerate(file):


            if line_type.lower() == 'test' and line_no+1 in [line_num+1, line_num+4]:
                split_line = re.split(': |, | \n', line)
                for i in range(1,len(split_line)-1,2):
                    lines[split_line[i].lower()].append(float(split_line[i+1]))
                   


            elif line_type.lower() == 'train' and line_no >= line_num:
                if line[0] == '(':
                    split_line = re.split('[(]|[)]: |: |, | \n', line)
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
        lines['stat'].append(loadstatLines(file,ii,line_type))
        lines['iter'].append(loadIterLines(file,ii,line_type))
    return lines


def loadAll(files,lines_list, order):
    # lines[improvement function][train or test][iter or stat][seed]
    lines = defaultdict(dict)
    for ii in range(len(order)):
        lines[order[ii]]['test'] = loadLines(files[ii], lines_list[ii]['test'], line_type='test')
        lines[order[ii]]['train'] = loadLines(files[ii], lines_list[ii]['train'], line_type='train')

    return lines







if __name__ == '__main__':
    file_names = ['SaveData/shift_laplace_sign.txt', 'SaveData/shift_laplace_median.txt', 'SaveData/shift_laplace_mean.txt', 'SaveData/shift_laplace_jacobi.txt']

    lines_list = []
    lines_list.append({'test': [137,290,426,586,738,884,1040], 'train': [23,168,320,457,616,768,914] })
    lines_list.append({'test': [48,90,136,174], 'train': [23,78,120,166]})
    lines_list.append({'test': [93,153,229,311], 'train': [74,123,183,259]})
    lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})

    order = ['sign', 'median', 'mean', 'jacobi']

    all_lines = loadAll(file_names, lines_list, order)

    # # # mean and median change in training or testing
    # line_type = 'train'
    # data_type = 'stat'
    # c = ['m','b','g','c','k','y', 'r']
    # # fig = plt.figure(60)
    # fig, axs = plt.subplots(3,1)
    # fig.suptitle('Improvement in training')
    
    # for k in range(3):
    #     for seed in range(4):
    #         lines = all_lines[order[k]][line_type][data_type][seed]
    #         if line_type == 'train':
    #             axs[k].plot(lines['iter1'],lines['mean'], label = f'mean: {seed}', linestyle='-', color=c[seed])
    #             axs[k].plot(lines['iter1'], lines['median'], label = f'median: {seed}', linestyle='--', color=c[seed])
    #         else:
    #             axs[k].plot(lines['mean'], label = f'mean: {seed}', linestyle='-', color=c[seed])
    #             axs[k].plot(lines['median'], label = f'median: {seed}', linestyle='--', color=c[seed])
    #         axs[k].set_title(f'Improvement func: {order[k]}')
    #         axs[k].legend()
    
    # axs[k].set_xlabel('Iteration count')


    # # mean and median change in training or testing
    line_type = 'test'
    data_type = 'stat'
    c = ['m','b','g','c','k','y', 'r']
    # fig = plt.figure(60)
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Improvement in Testing')
    temp = [(0,0),(0,1),(1,0),(1,1)]
    
    for k in range(4):
        for seed in range(4):
            lines = all_lines[order[k]][line_type][data_type][seed]
            axs[temp[seed]].plot(lines['mean'], label = f'{order[k]}', linestyle='-', color=c[k])
            axs[temp[seed]].plot(lines['median'], label = f'{order[k]}', linestyle='--', color=c[k])
            axs[temp[seed]].set_title(f'Seed {seed}')
            axs[temp[seed]].set_xticks([0,1],['non','pre'])
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='center right')
    fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')

    # # density plots
    # line_type = 'test'
    # data_type = 'stat'
    # for k in range(4):
    #     fig, axs = plt.subplots(2,2)
    #     fig.suptitle(f'Density of testing iterations with improve func: {order[k]}')
    #     # for seed in range(len(lines_list[k][line_type])):
    #     temp = [(0,0),(0,1),(1,0),(1,1)]
    #     for seed in range(4):
    #         # plt.figure(seed)
    #         non = all_lines[order[k]][line_type]['iter'][seed][0]
    #         pre = all_lines[order[k]][line_type]['iter'][seed][1]
    #         print(k,seed, max(pre))

    #         n1 ,_,_ = axs[temp[seed]].hist([non,pre],bins=40, alpha = 1, label=['Non','precond'], color=['c','m'])

    #         # axs[temp[seed]].boxplot((non,pre),orientation='horizontal',tick_labels=['non', 'Precond'],showmeans=True, positions=[5,8], widths=1.5)
    #         axs[temp[seed]].legend()
    #         axs[temp[seed]].set_title(f'Seed: {seed}')



    # # How many are better bar plot
    # line_type = 'test'
    # b = []
    # labels = []
    # for k in range(len(order)):
    #     for seed in range(len(all_lines[order[k]][line_type]['stat'])):
    #     # for seed in range(4):
    #         b.append(all_lines[order[k]][line_type]['stat'][seed]['b'][0])
    #         labels.append(f'{order[k]}: {seed}')
    

    # plt.figure(42)
    # c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    # plt.bar(labels,b,color=c)
    # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[order[k]][line_type]['iter'][0][0])/2,color='k',label='half')
    # plt.legend()
    # plt.title('How many that are better in testing')




    plt.show()