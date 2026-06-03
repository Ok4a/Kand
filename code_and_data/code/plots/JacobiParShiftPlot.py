import numpy as np
import matplotlib.pyplot as plt

from loadDataPlot import *




if __name__ == '__main__':
    file_names = ['data/new precond (jacobi_shift)/shift_Jacobi_laplace_sign.txt','data/new precond (jacobi_shift)/shift_Jacobi_laplace_median.txt','data/new precond (jacobi_shift)/shift_Jacobi_laplace_mean.txt']

    lines_list = []
    lines_list.append({'train': [22,135, 245,398],'test': [102, 211, 363, 530]})

    lines_list.append({'train': [34,142,255,369],'test': [86,198,311,445]})

    lines_list.append({'train': [35,153,259,371],'test': [95,201,313,441]})

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
    fig.suptitle('Learning curves\n with Shifted Jacobi preconditioner')
    

    for k in range(len(order)):
        for index in seeds:

            lines = all_lines[order[k]][line_type][data_type][index]

            print(order[k], index, len(lines['mean']))
            axs_flat[index].grid(zorder = 0, linestyle='--')
            axs_flat[index].set_axisbelow(True)

            axs_flat[index].plot(lines['iter2'],lines['mean'], label = f'{order[k]}: Mean', linestyle='-', color=c[k])
            axs_flat[index].plot(lines['iter2'], lines['median'], label = f'{order[k]}: Median', linestyle='--', color=c[k])
            # print(order[k], seed, len(lines['mean']))
            axs_flat[index].set_title(f'Seed: {index}')
            # axs[temp[i_k]].legend()
    
            # axs_flat[seed].set_xlabel('Iteration count')
            # axs_flat[seed].set_ylabel('Solver count')
    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='lower center', title='Improvement function: measure', ncols=6, bbox_to_anchor=(0.5, -0.005))
    fig.supxlabel('Learning steps',y=0.07)
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



    # boxplots

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


#     # How many are better bar plot
#     line_type = 'test'
#     b = []
#     labels = []
#     for k in range(len(order)):
#         # for seed in range(len(all_lines[order[k]][line_type]['stat'])):
#         for index in seeds:
#             # try:
#                 # b.append(all_lines[order[k]][line_type]['stat'][seed]['b']['last'])
#             # except:
#             b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

#             labels.append(f'{order[k]}: {index}')
    
#     fig = plt.figure(42, figsize=(10,4))
#     axs = fig.subplots(1,1)
#     axs.grid(zorder = 0, linestyle='--')
#     axs.bar(labels,b,color=c, zorder = 1)
#     plt.xticks(rotation=-45)
#     axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
#     axs.axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
#     axs.legend()
#     plt.title('How many learned preconditioned systems are better than using no preconditioner')
#     axs.set_axisbelow(True)
#     fig.subplots_adjust(bottom=0.17)


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

    # line_type = 'test'
    # fig = plt.figure(423, figsize=(10,5))
    # axs = fig.subplots(2,2)
    # axs_flat = axs.flatten()
   
    # for seed in seeds:
    #     b = []
    #     labels = []
    #     for k in range(len(order)):
    #         # except:
    #         b.append(all_lines[order[k]][line_type]['stat'][index]['bvn']['last'])

    #         labels.append(f'{order[k]}')
    #     axs_flat[seed].bar(labels,b,color=c, zorder = 1)
    #     axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    #     axs_flat[seed].axhline(len(all_lines[order[k]][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')

    #     axs_flat[seed].set_yticks(np.arange(0, 250+1, 50))
    
    
    #     axs_flat[seed].grid(zorder = 0, linestyle='--')
    #     axs_flat[seed].set_axisbelow(True)
    #     axs_flat[seed].set_title(f'Seed: {seed}')
    # # plt.xticks(rotation=-45)
    # # axs.legend()
    # fig.suptitle('How many learned preconditioned systems are better')
    # fig.supxlabel('Improvement function')
    # fig.subplots_adjust(hspace=0.275)

    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab, ncols=3)



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
    # fig.legend(li, lab, ncols=3)

    plt.show()