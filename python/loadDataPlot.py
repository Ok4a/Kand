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
    file_names = ['SaveData/shift_laplace_sign.txt', 'SaveData/shift_laplace_median.txt', 'SaveData/shift_laplace_mean.txt']

    lines_list = []
    lines_list.append({'test': [137,290,426,586,738,884,1040], 'train': [23,168,320,457,616,768,914] })
    lines_list.append({'test': [48,90,136,174], 'train': [23,78,120,166]})
    lines_list.append({'test': [93,153,229,311], 'train': [74,123,183,259]})

    order = ['sign', 'median', 'mean']

    all_lines = loadAll(file_names, lines_list, order)

    k = 1
    line_type = 'train'
    data_type = 'stat'


    # for seed in range(len(lines_list[k][line_type])):
    #     lines = all_lines[order[k]][line_type][data_type][seed]
    #     # lines = loadstatLines(file_names[k], lines_list[k][line_type][ii], line_type = line_type)
    #     # print(lines[0])
    #     plt.figure(123+seed)
    #     if line_type == 'train':
    #         plt.plot(lines['iter1'],lines['mean'], label = f'mean: {seed}')
    #         plt.plot(lines['iter1'], lines['median'], label = f'median: {seed}')
    #     else:
    #         plt.plot(lines['mean'], label = f'mean: {seed}')
    #         plt.plot(lines['median'], label = f'median: {seed}')
        
        
    #     plt.title(f'Improve func: {order[k]}')

    #     plt.legend()
    


    line_type = 'test'
    data_type = 'stat'
    # for seed in range(len(lines_list[k][line_type])):
    #     plt.figure(seed)
    #     non = all_lines[order[k]][line_type]['iter'][seed][0]
    #     pre = all_lines[order[k]][line_type]['iter'][seed][1]

    #     n1 ,_,_ = plt.hist([non,pre],bins=40, alpha = 0.5, label=['Non','pre'], color=['red','blue'])

    #     # plt.vlines(stat_lines['mean'][0],ymin=0,ymax=ymax, label='mean Non', colors='red')
    #     # plt.vlines(stat_lines['mean'][1],ymin=0,ymax=ymax, label='mean final', colors='blue')
    #     # plt.vlines(stat_lines['median'][0],ymin=0,ymax=ymax, label='median Non', colors='red', linestyles='dashed')
    #     # plt.vlines(stat_lines['median'][1],ymin=0,ymax=ymax, label='median final', colors='blue', linestyles='dashed')
    #     plt.legend()

    #     # plt.figure(ii+10)
    #     # plt.boxplot((iter_lines[0],iter_lines[1]),orientation='horizontal',tick_labels=['non', 'final'],meanline=True, positions=[5,8], widths=1.5)


    line_type = 'test'
    b = []
    labels = []
    for k in range(len(order)):
        for seed in range(len(all_lines[order[k]][line_type]['stat'])):
        # for seed in range(4):
            b.append(all_lines[order[k]][line_type]['stat'][seed]['b'][0])
            labels.append(f'{order[k]}: {seed}')
    

    plt.figure(42)
    c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    plt.bar(labels,b,color=c)
    plt.xticks(rotation=-45)
    plt.hlines(125,0,14, colors='k')
    plt.title('How many that are better in testing')


    plt.show()