import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['data/Sub super sym/diag_shift_jacobi_laplace_sign_sym.txt']
    lines_list = []
    lines_list.append({'train': [30, 219, 459, 678], 'test': [172,411,629,858]})
    method = 'sign'
    seeds = [0]
    k = 0
    num_par = [1,2,3,4]
    which2use = range(len(num_par))
    all_lines = loadAll(file_names, lines_list, seeds)

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()

    fig = plt.figure(0, figsize=(10,4))
    axs = fig.subplots(1)
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    for coef in range(len(num_par)):
        lines = all_lines[0][line_type][data_type][coef]
        

        axs.plot(lines['iter2'],lines['mean'], label = f'{num_par[coef]}: mean', linestyle='-', color=c[coef])
        axs.plot(lines['iter2'], lines['median'], label = f'{num_par[coef]}: median', linestyle='--', color=c[coef])

    axs.legend( title='Num par')
    fig.supylabel('Solver iteration')
    fig.supxlabel('Learning steps')
    fig.suptitle('Learning curves\n for symmetric preconditioners')


    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # fig = plt.figure(1)

    # Test_index = ['no precond', 'jacobi', 'best']
    
    # # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    
    # for index in runs:
    #     fig = plt.figure(5+index)
    
    #     for coef in range(len(num_par)):
    #         mean_list = []
    #         median_list = []
    #         lines = all_lines[index][line_type][data_type][coef]
    #         for key in Test_index:
    #             mean_list.append(lines['mean'][key])
    #             median_list.append(lines['median'][key])
    #         plt.plot(mean_list, label = f'{num_par[coef]}: mean', linestyle='-', color=c[coef+index*4])
    #         plt.plot(median_list, label = f'{num_par[coef]}: median', linestyle='--', color=c[coef+index*4])

    #     plt.xticks(range(3),['No Precond','Jacobi','Learned'])
    #     fig.legend(loc='center right', title='Num diag')
    #     plt.ylabel('Solver iteration')
    #     plt.xlabel('Preconditioner type')

    
    
    fig = plt.figure(45678, figsize=(6,4))
    axs = fig.subplots(1)
    y_offset = 1200

    lab = ['Non'] + num_par[which2use[0]:which2use[-1]+1] + ['Jacobi']
    x =np.arange(len(lab))
    # for seed in seeds:
    data ={}
    index = 0
    data['Mean'] = [all_lines[0][line_type][data_type][index]['mean']['no precond']]
    data['Median'] = [all_lines[0][line_type][data_type][index]['median']['no precond']]
    k = 0
    for index in which2use:
        lines = all_lines[0][line_type][data_type][index]
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
        rects = axs.bar(x+offset+0.125, np.array(measure)-y_offset , width,label = att, bottom=y_offset)
        # axs_flat[seed].bar_label(rects, padding=3)
        mult +=1

        # axs.set_title(f'Seed: {seed}')
        axs.set_axisbelow(True)
        axs.set_xticks(x+width, lab)
        axs.legend()
        # axs.set_ylabel('Solver iteration count')
    fig.suptitle('Evaluation of learned preconditioner')
    fig.supylabel('Solver iteration count')
    fig.supxlabel('Number of partitions')   

    # # density plots
    # line_type = 'test'
    # data_type = 'iter'
    # dense_label = densityLabels()
    # for index in runs:
    #     for coef in range(len(num_diag)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+coef+index*10)
    #         plt.title(f'With {num_diag[coef]} super and sub diags in precond')
    #         fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {index}')
    #         non = all_lines[index][line_type][data_type][coef]['no precond']
    #         jacobi = all_lines[index][line_type][data_type][coef]['jacobi']
    #         try:
    #             with_ = all_lines[index][line_type][data_type][coef]['best']
    #         except:
    #             with_ = all_lines[index][line_type][data_type][coef]['last']

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
    # for index in runs:
    #     for coef in range(len(num_diag)):
    #         try:
    #             b.append(all_lines[index][line_type]['stat'][coef]['b']['best'])
    #         except:
    #             b.append(all_lines[index][line_type]['stat'][coef]['bvn']['best'])

    #         labels.append(f'Num diags: {num_diag[coef]}')

    # plt.figure(42)
    # # c = ['m','b','g','c','k','k','k','m','b','g','c','m','b','g','c']
    # # c = ['m','b','g','c']
    # plt.bar(labels,b,color=c)
    # # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v No Precond')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    offset = 220
    fig = plt.figure(43, figsize=(6,4))
    axs = fig.subplots(1)
    labels = []
    b = []
    index = 0
    axs.grid(zorder = 0, linestyle='--')
    axs.set_axisbelow(True)
    for coef in range(len(num_par)):
        # print(all_lines[method][line_type]['stat'][coef]['bvj'])
        try:
            b.append(all_lines[index][line_type]['stat'][coef]['bvj']['best'])
        except:
            b.append(all_lines[index][line_type]['stat'][coef]['bvj']['last'])

        labels.append(f'{num_par[coef]}')

    axs.bar(labels,np.array(b)-offset,color=c[len(num_par)*index:],bottom = offset)
    # plt.xticks(rotation=-45)  
    axs.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # axs.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    axs.legend()
    # axs.xlabel('Num super n sub diags, (seed)')
    fig.suptitle('The number of systems with lower\n solver iteration count compared with Jacobi')

    fig.supxlabel('Number of partitions')


    plt.show()
