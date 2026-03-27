import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


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
                    lines[line_key] = np.array([int(x) for x in line[2:-2].split(', ')])
                if break_next:
                    break
                if line[0] == '[':
                    break_next = True

                    
            elif line_type.lower() == 'train' and line_no >= line_num:
                if line[0] == '	':
                    lines.append(np.array([int(x) for x in line[2:-2].split(', ')]))
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
                if line[0] == '[':
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







if __name__ == '__main__':
    file_names = ['SaveData/shift_laplace_sign.txt', 'SaveData/shift_laplace_median.txt', 'SaveData/shift_laplace_mean.txt', 'SaveData/shift_laplace_jacobi.txt', 'SaveData/shift_Jacobi_laplace_sign.txt']

    lines_list = []
    lines_list.append({'test': [137,290,426,586,738,884,1040], 'train': [23,168,320,457,616,768,914] })
    lines_list.append({'test': [48,90,136,174], 'train': [23,78,120,166]})
    lines_list.append({'test': [93,153,229,311], 'train': [74,123,183,259]})
    lines_list.append({'test': [5,15,25,35,45,55,65], 'train': []})
    lines_list.append({'test': [102,211,363,530,810], 'train': [22,135,245,398,564]})

    order = ['sign', 'median', 'mean', 'jacobi', 'sign_jacobi_shift']

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
    for i_k in range(4):
        k = g[i_k]
        for seed in range(len(all_lines[order[k]][line_type][data_type])):
            lines = all_lines[order[k]][line_type][data_type][seed]
            if line_type == 'train':
                axs[temp[i_k]].plot(lines['iter1'],lines['mean'], label = f'mean: {seed}', linestyle='-', color=c[seed])
                axs[temp[i_k]].plot(lines['iter1'], lines['median'], label = f'median: {seed}', linestyle='--', color=c[seed])
            else:
                axs[temp[i_k]].plot(lines['mean'], label = f'mean: {seed}', linestyle='-', color=c[seed])
                axs[temp[i_k]].plot(lines['median'], label = f'median: {seed}', linestyle='--', color=c[seed])
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
    fig.suptitle('Improvement in Testing')
    temp = [(0,0),(0,1),(1,0),(1,1)]
    
    for k in range(len(order)):
        for seed in range(4):
            lines = all_lines[order[k]][line_type][data_type][seed]
            if len(lines['mean']) == 3:
                axs[temp[seed]].plot(lines['mean'].values(), label = f'{order[k]}', linestyle='-', color=c[k])
                axs[temp[seed]].plot(lines['median'].values(), label = f'{order[k]}', linestyle='--', color=c[k])
            else:
                axs[temp[seed]].plot(lines['mean'].values(), label = f'{order[k]}', linestyle='-', color=c[k])
                axs[temp[seed]].plot(lines['median'].values(), label = f'{order[k]}', linestyle='--', color=c[k])
            axs[temp[seed]].set_title(f'Seed {seed}')
            axs[temp[seed]].set_xticks([0,1],['non','pre'])
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='center right')
    fig.text(0.02, 0.34, 'Solver iteration count', rotation='vertical')

    # density plots
    line_type = 'test'
    data_type = 'stat'
    for k in range(5):
        fig, axs = plt.subplots(2,2)
        fig.suptitle(f'Density of testing iterations with improve func: {order[k]}')
        # for seed in range(len(lines_list[k][line_type])):
        temp = [(0,0),(0,1),(1,0),(1,1)]
        for seed in range(4):
            # plt.figure(seed)
            non = all_lines[order[k]][line_type]['iter'][seed]['no precond']
            pre = all_lines[order[k]][line_type]['iter'][seed]['last']

            n1 ,_,_ = axs[temp[seed]].hist([non,pre],bins=40, alpha = 1, label=['Non','precond'], color=['c','m'])

            axs[temp[seed]].boxplot((non,pre),orientation='horizontal',tick_labels=['non', 'Precond'],showmeans=True, positions=[5,8], widths=1.5)
            axs[temp[seed]].legend()
            axs[temp[seed]].set_title(f'Seed: {seed}')



    # How many are better bar plot
    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)):
        # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        for seed in range(4):
            try:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
            except:
                b.append(all_lines[order[k]][line_type]['stat'][seed]['bvn']['last'])

            labels.append(f'{order[k]}: {seed}')
    

    plt.figure(42)
    c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    c = ['m','b','g','c']
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    plt.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    plt.legend()
    plt.title('How many that are better in testing')




    plt.show()