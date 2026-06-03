import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['data/final_large/diag_shift_laplace_large.txt']
    lines_list = []
    lines_list.append({'train': [34,120,211,310], 'test': [66,158,257,346]})
    method = 'sign'
    seeds = [0,1,2,4]
    k = 0
    file_index_list = [0]
    all_lines = loadAll(file_names, lines_list, [0])

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    # fig.suptitle('Improvement in training\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    fig = plt.figure(0, figsize=(6,4))
    axs = fig.subplots(1)
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    # fig.suptitle('Training')
    # fig.suptitle('Improvement in training\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    for index in range(len(seeds)):
        lines = all_lines[0][line_type][data_type][index]
        

        axs.plot(lines['iter2'],lines['mean'], label = f'{seeds[index]}: mean', linestyle='-', color=c[index])
        axs.plot(lines['iter2'], lines['median'], label = f'{seeds[index]}: median', linestyle='--', color=c[index])

    axs.legend(title = 'Seed: measure')
    axs.set_ylabel('Solver iteration')
    axs.set_xlabel('Learning steps')



    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # fig = plt.figure(1)

    Test_index = ['no precond', 'jacobi', 'best']
    
    # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    
    # fig = plt.figure(5)
    # fig.suptitle('Testing')
    # # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of diags in the preconditioner')
    # for index in range(len(runs)):
    #     mean_list = []
    #     median_list = []
    #     lines = all_lines[0][line_type][data_type][index]
    #     for key in Test_index:
    #         mean_list.append(lines['mean'][key])
    #         median_list.append(lines['median'][key])
    #     plt.plot(mean_list, label = f'{runs[index]}: mean', linestyle='-', color=c[index])
    #     plt.plot(median_list, label = f'{runs[index]}: median', linestyle='--', color=c[index])

    #     plt.xticks(range(3),['No Precond','Jacobi','Learned'])
    #     fig.legend(loc='center right', title='Seed: measure')
    #     plt.ylabel('Solver iteration')
    #     plt.xlabel('Preconditioner type')
    


    fig = plt.figure(45678, figsize=(10,5.5))
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
    y_offset = 1000

    for index in range(len(seeds)):

        lab = ['Non'] + ['Learned'] + ['Jacobi']
        x =np.arange(len(lab))
        # for seed in seeds:
        data ={}
        # index = 0
        data['Mean'] = [all_lines[0][line_type][data_type][index]['mean']['no precond']]
        data['Median'] = [all_lines[0][line_type][data_type][index]['median']['no precond']]
        k = 0
        # for index in range(len(runs)):
        lines = all_lines[0][line_type][data_type][index]
        data['Mean'].append(lines['mean']['last'])
        data['Median'].append(lines['median']['last'])
        data['Mean'].append(lines['mean']['jacobi'])
        data['Median'].append(lines['median']['jacobi'])
        width = 0.25
        mult  = 0
        for att, measure in data.items():
            offset = width * mult
            
            axs_flat[index].grid(zorder = 0, linestyle='--')
            # rects = axs.bar(x+offset+0.125, np.array(measure)-1500 , width, bottom=1500)
            rects = axs_flat[index].bar(x+offset+0.125, np.array(measure)-y_offset , width,label = att, bottom=y_offset)
            # axs_flat[seed].bar_label(rects, padding=3)
            mult +=1

            axs_flat[index].set_title(f'Seed: {seeds[index]}')
            axs_flat[index].set_axisbelow(True)
            axs_flat[index].set_xticks(x+width, lab)
            # axs_flat[index].legend()
            # axs_flat[index].set_ylabel('Solver iteration count')
    # fig.subplots_adjust(hspace=0.3)

    li, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(li, lab, loc='lower center', ncol = 2)
    fig.supylabel('Solver iteration count')
    fig.suptitle('Evaluation of learned preconditioner')
    # fig.supxlabel('Preconditioner')
    fig.subplots_adjust(hspace=0.35)

    # # density plot
    # line_type = 'test'
    # data_type = 'iter'
    # dense_label = densityLabels()
    # for index in file_index_list:
    #     for index in range(len(runs)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+index+index*10)
    #         # plt.title(f'With {seeds[seed]} super and sub diags in precond')
    #         # fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {file_index}')
    #         non = all_lines[index][line_type][data_type][index]['no precond']
    #         jacobi = all_lines[index][line_type][data_type][index]['jacobi']
    #         try:
    #             with_ = all_lines[index][line_type][data_type][index]['best']
    #         except:
    #             with_ = all_lines[index][line_type][data_type][index]['last']

    #         plt.hist([non,with_,jacobi],bins=40, alpha = 1, label=dense_label, color=c[:3])

    #         plt.boxplot((non, with_, jacobi), orientation='horizontal', tick_labels = dense_label,showmeans=False, notch=True, positions = [5, 8, 11], widths=1.5)
    #         plt.legend(title='Preconditioner')
    #         plt.xlabel('Solver iteration')

        



#     def update_hist(num, data):
#         line_type = 'train'
#         data_type = 'iter'
#         seed = 0
#         coef = 3
#         plt.cla()
#         plt.hist(all_lines[seed][line_type][data_type][coef][num], bins=40, range=(0,10000))
#         plt.title(label=f'({data[seed][line_type]['stat'][coef]['iter1'][num]},{data[seed][line_type]['stat'][coef]['iter2'][num]})')



#  # density plots
#     line_type = 'train'
#     data_type = 'iter'
#     seed = 0
#     coef = 3

#     data = all_lines[seed][line_type][data_type][coef]
#     num_frame = len(data)

#     fig = plt.figure(70)
#     hist = plt.hist(all_lines[seed][line_type][data_type][coef][0])

#     ani = animation.FuncAnimation(fig, update_hist, num_frame, fargs=(all_lines,))
  





    # # How many are better bar plot v Non
    # line_type = 'test'
    # b = []
    # labels = []
    # for index in file_index_list:
    #     for index in range(len(runs)):
    #         try:
    #             b.append(all_lines[index][line_type]['stat'][index]['b']['best'])
    #         except:
    #             b.append(all_lines[index][line_type]['stat'][index]['bvn']['best'])

    #         labels.append(f'{runs[index]}')

    # plt.figure(42)
    # # c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    # # c = ['m','b','g','c']
    # plt.bar(labels,b,color=c)
    # # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v No Precond')
    # plt.xlabel('Seed')

    # How many are better bar plot v Jacobi
    offset = 150
    line_type = 'test'
    b = []
    labels = []
    for index in range(len(seeds)):
        # print(all_lines[method][line_type]['stat'][coef]['bvj'])
        try:
            b.append(all_lines[0][line_type]['stat'][index]['bvj']['best'])
        except:
            b.append(all_lines[0][line_type]['stat'][index]['bvj']['last'])

        labels.append(f'{seeds[index]}')

    fig = plt.figure(43, figsize=(6,4))
    axs= fig.subplots(1)
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    axs.bar(labels,np.array(b)-offset,color=c, bottom=offset)
    # plt.xticks(rotation=-45)
    axs.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    if offset <= len(all_lines[0][line_type]['iter'][0]['no precond'])/2:
        plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    fig.legend()
    fig.suptitle('The number of systems with lower\n solver iteration count compared with Jacobi')
    fig.supxlabel('Seed')



    plt.show()
