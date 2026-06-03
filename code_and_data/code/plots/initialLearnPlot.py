import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['data/initial Learn/initial_par_shift_learn_eris1176.txt','data/initial Learn/initial_par_shift_learn_fidap004.txt','data/initial Learn/initial_par_shift_learn_orsirr_1.txt','data/initial Learn/initial_par_shift_learn_sherman5.txt']
    # file_names = ['SaveData/super_sub/diag_shift_jacobi_laplace_sign_1.txt']
    lines_list = []
    lines_list.append({'train': [37], 'test': [67]})
    lines_list.append({'train': [37], 'test': [59]})
    lines_list.append({'train': [37], 'test': [73]})
    lines_list.append({'train': [37], 'test': [49]})
    method = 'sign'
    k = 0
    index_list = [0,1,2,3]
    all_lines = loadAll(file_names, lines_list, index_list)
    names_list = ['eris1176','fidap004','orsirr_1','sherman5']
    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    fig = plt.figure(0, figsize=(10,7))
    axs = fig.subplots(2,2)
    axs_flat = axs.flatten()
    
    temp = [(0,0),(0,1),(1,0),(1,1)]

    for index in index_list:
        axs_flat[index].set_title(names_list[index])
        axs_flat[index].grid(zorder = 0, linestyle='--')
        axs_flat[index].set_axisbelow(True)

        lines = all_lines[index][line_type][data_type][0]
            

        axs_flat[index].plot(lines['iter2'],lines['mean'], linestyle='-', color=c[0])
        # plt.plot(lines['iter2'], lines['median'], label = f'{num_superNsub[coef]}: median', linestyle='--', color=c[coef])

        no_pre = all_lines[index]['test']['iter'][0]['no precond'][0]
        axs_flat[index].axhline(no_pre, color = 'k', label='No Preconditioning', ls='--')

        axs_flat[index].legend(loc='upper right')
        print(lines['mean'][-1],lines['mean'][0]-lines['mean'][-1],no_pre-lines['mean'][-1], len(lines['mean']))
        # plt.ylabel('Solver iteration')
        # plt.xlabel('Learn iteration')
    fig.suptitle('Learning curves of the Matrix Market matrices')
    fig.supxlabel('Learning steps')
    fig.supylabel('Solver iterations')


    # # mean and median change in testing
    # line_type = 'test'
    # data_type = 'stat'
    # # fig = plt.figure(1)
    # Test_index = ['no precond', 'best']
    
    # # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    
    # for file_index in index_list:
    #     fig = plt.figure(5+file_index)
    
    #     fig.suptitle(f'Improvement in Testing\n Sign as improvement function for different number of diags in the preconditioner\nSeed: {file_index}')
    #     # lines = all_lines[seed][line_type][data_type][coef]
    #     lines = all_lines[file_index][line_type][data_type][0]
    #     # mean_list = []
    #     median_list = []
    #     for key in Test_index:
    #         # mean_list.append(lines['mean'][key])
    #         median_list.append(lines['median'][key])
    #     # plt.plot(mean_list, label = f'{num_superNsub[coef]}: mean', linestyle='-', color=c[coef])
    #     plt.plot(median_list, linestyle='-', color=c[0])

    #     plt.xticks(range(2),['No preconditioner','Learned'])
    #     # fig.legend(loc='center right', title='Num diag')
    #     plt.ylabel('Solver Iteration')
        # plt.xlabel('Preconditioner')


    # # density plots
    # line_type = 'test'
    # data_type = 'iter'
    # for file_index in index_list:
    #     for coef in range(len(num_superNsub)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+coef+file_index*10)
    #         plt.title(f'With {num_superNsub[coef]} super and sub diags in precond')
    #         fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {file_index}')
    #         non = all_lines[file_index][line_type][data_type][coef]['no precond']
    #         jacobi = all_lines[file_index][line_type][data_type][coef]['jacobi']
    #         try:
    #             with_ = all_lines[file_index][line_type][data_type][coef]['best']
    #         except:
    #             with_ = all_lines[file_index][line_type][data_type][coef]['last']

    #         plt.hist([non,with_,jacobi],bins=40, alpha = 1, label=['No Precond', 'Precond', 'Jacobi'], color=c[:3])

    #         plt.boxplot((non, with_, jacobi), orientation='horizontal', tick_labels = ['No Precond', 'Precond', 'jacobi'],showmeans=False, notch=True, positions = [5,8, 11], widths=1.5)
    #         plt.legend()
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
    # for file_index in index_list:
    #     for coef in range(len(num_superNsub)):
    #         try:
    #             b.append(all_lines[file_index][line_type]['stat'][coef]['b']['best'])
    #         except:
    #             b.append(all_lines[file_index][line_type]['stat'][coef]['bvn']['best'])

    #         labels.append(f'{num_superNsub[coef]}, ({file_index})')

    # plt.figure(42)
    # plt.bar(labels,b,color=c)
    # # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    # plt.legend()
    # plt.title('How many that are better in testing v no preconditioner')
    # plt.xlabel('Num super n sub diags, (seed)')

    # # How many are better bar plot v Jacobi
    # line_type = 'test'
    # b = []
    # labels = []
    # for file_index in index_list:
    #     for coef in range(len(num_superNsub)):
    #         # print(all_lines[method][line_type]['stat'][coef]['bvj'])
    #         try:
    #             b.append(all_lines[file_index][line_type]['stat'][coef]['bvj']['best'])
    #         except:
    #             b.append(all_lines[file_index][line_type]['stat'][coef]['bvj']['last'])

    #         labels.append(f'{num_superNsub[coef]}, ({file_index})')

    # plt.figure(43)
    # plt.bar(labels,b,color=c)
    # # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
    # plt.legend()
    # plt.title('How many that are better in testing v Jacobi')
    # plt.xlabel('Num super n sub diags, (seed)')



    plt.show()
