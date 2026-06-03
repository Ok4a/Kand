import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from loadDataPlot import *



if __name__ == '__main__':
   

    file_names = ['finalData/sub super/diag_shift_jacobi_laplace_sign_0.txt','finalData/sub super/diag_shift_jacobi_laplace_sign_1.txt']
    # file_names = ['SaveData/super_sub/diag_shift_jacobi_laplace_sign_1.txt']
    lines_list = []
    lines_list.append({'train': [29,223,431,653], 'test': [177,385,607,811]})
    lines_list.append({'train': [29,207,431,627], 'test': [161,385,581,767]})
    method = 'sign'
    seeds = [0,1]
    k = 0
    num_superNsub = [1,2,3,4]
    all_lines = loadAll(file_names, lines_list, seeds)
    which2use=range(len(num_superNsub))

    # # mean and median change in training
    line_type = 'train'
    data_type = 'stat'
    c = colours()
    temp = [(0,0),(0,1),(1,0),(1,1)]
    fig = plt.figure(0, figsize=(10,5))
    axs = fig.subplots(1,2)
    fig.suptitle('Learning curves\n for super- and subdiagonals')
    
    for index in seeds:
        axs[index].grid(zorder = 0, linestyle='--')
        axs[index].set_axisbelow(True)
        # fig.suptitle('Improvement in training\n Sign as improvement function for different number of super and dub diags in the preconditioner')
        for coef in range(len(num_superNsub)):
            lines = all_lines[index][line_type][data_type][coef]
            

            axs[index].plot(lines['iter2'],lines['mean'], label = f'{num_superNsub[coef]}: mean', linestyle='-', color=c[coef+index*4])
            axs[index].plot(lines['iter2'], lines['median'], label = f'{num_superNsub[coef]}: median', linestyle='--', color=c[coef+index*4])
    
        axs[index].legend(title='Number of diagonals')
        axs[index].set_ylabel('Solver iteration')
        axs[index].set_xlabel('Learning steps')
        axs[index].set_title(f'Seed: {index}')



    # mean and median change in testing
    line_type = 'test'
    data_type = 'stat'
    # fig = plt.figure(1)
    Test_index = ['no precond', 'jacobi', 'best']
    
    # fig.suptitle('Improvement in Testing\n Sign as improvement function for different number of super and sub diags in the preconditioner')
    
    # for index in runs:
    #     fig = plt.figure(5+index)
    
    #     fig.suptitle(f'Improvement in Testing\n Sign as improvement function for different number of diags in the preconditioner\nSeed: {index}')
    #     for coef in range(len(num_superNsub)):
    #         # lines = all_lines[seed][line_type][data_type][coef]
    #         lines = all_lines[index][line_type][data_type][coef]
    #         mean_list = []
    #         median_list = []
    #         for key in Test_index:
    #             mean_list.append(lines['mean'][key])
    #             median_list.append(lines['median'][key])
    #         plt.plot(mean_list, label = f'{num_superNsub[coef]}: mean', linestyle='-', color=c[coef+index*4])
    #         plt.plot(median_list, label = f'{num_superNsub[coef]}: median', linestyle='--', color=c[coef+index*4])

    #     plt.xticks(range(3),['No preconditioner','Jacobi','Learned'])
    #     fig.legend(loc='center right', title='Num diag')
    #     plt.ylabel('Solver Iteration')
    #     plt.xlabel('Preconditioner')

    fig = plt.figure(45678, figsize=(10,4))
    axs = fig.subplots(1,2)
    y_offset = 1200

    for file_index in seeds:
        lab = ['Non'] + num_superNsub[which2use[0]:which2use[-1]+1] + ['Jacobi']
        x =np.arange(len(lab))
        # for seed in seeds:
        data ={}
        index = 0
        data['Mean'] = [all_lines[file_index][line_type][data_type][index]['mean']['no precond']]
        data['Median'] = [all_lines[file_index][line_type][data_type][index]['median']['no precond']]
        k = 0
        for index in which2use:
            lines = all_lines[file_index][line_type][data_type][index]
            data['Mean'].append(lines['mean']['last'])
            data['Median'].append(lines['median']['last'])
        data['Mean'].append(lines['mean']['jacobi'])
        data['Median'].append(lines['median']['jacobi'])
        width = 0.25
        mult  = 0
        for att, measure in data.items():
            offset = width * mult
            
            axs[file_index].grid(zorder = 0, linestyle='--')
            # rects = axs.bar(x+offset+0.125, np.array(measure)-1500 , width, bottom=1500)
            rects = axs[file_index].bar(x+offset+0.125, np.array(measure)-y_offset , width,label = att, bottom=y_offset)
            # axs_flat[seed].bar_label(rects, padding=3)
            mult +=1

            # axs.set_title(f'Seed: {seed}')
            axs[file_index].set_axisbelow(True)
            axs[file_index].set_xticks(x+width, lab)
            axs[file_index].legend()
            axs[file_index].set_ylabel('Solver iteration count')
            axs[file_index].set_title(f'Seed: {file_index}')
    fig.suptitle('Evaluation of learned preconditioner')
    fig.supxlabel('Number of super- and subdiagonals')
            
    # li, lab = fig.axes[0].get_legend_handles_labels()
    # fig.legend(li, lab)
    # fig.supylabel('Solver iteration count')
    # fig.suptitle('Evaluation of learned preconditioner')
    # fig.supxlabel('Preconditioner')
    # fig.subplots_adjust(hspace=0.35)


    # # density plots
    # line_type = 'test'
    # data_type = 'iter'
    # for index in runs:
    #     for coef in range(len(num_superNsub)):
    #         # fig, axs = plt.subplots(2,2)
    #         fig = plt.figure(100+coef+index*10)
    #         plt.title(f'With {num_superNsub[coef]} super and sub diags in precond')
    #         fig.suptitle(f'Density of testing iterations with improve func: {method}, precond: Jacobi diag Shift, seed: {index}')
    #         non = all_lines[index][line_type][data_type][coef]['no precond']
    #         jacobi = all_lines[index][line_type][data_type][coef]['jacobi']
    #         try:
    #             with_ = all_lines[index][line_type][data_type][coef]['best']
    #         except:
    #             with_ = all_lines[index][line_type][data_type][coef]['last']

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
    # for index in runs:
    #     for coef in range(len(num_superNsub)):
    #         try:
    #             b.append(all_lines[index][line_type]['stat'][coef]['b']['best'])
    #         except:
    #             b.append(all_lines[index][line_type]['stat'][coef]['bvn']['best'])

    #         labels.append(f'{num_superNsub[coef]}, ({index})')

    # plt.figure(42)
    # plt.bar(labels,b,color=c)
    # # plt.xticks(rotation=-45)
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='all')
    # plt.axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='half')
    # plt.legend()
    # plt.title('How many that are better in testing v no preconditioner')
    # plt.xlabel('Num super n sub diags, (seed)')

    # How many are better bar plot v Jacobi
    line_type = 'test'
    offset = 220
    fig = plt.figure(43, figsize=(10,4))
    axs = fig.subplots(1,2)
    
    for index in seeds:
        axs[index].grid(zorder = 0, linestyle='--')
        axs[index].set_axisbelow(True)
        labels = []
        b = []
        for coef in range(len(num_superNsub)):
            # print(all_lines[method][line_type]['stat'][coef]['bvj'])
            try:
                b.append(all_lines[index][line_type]['stat'][coef]['bvj']['best'])
            except:
                b.append(all_lines[index][line_type]['stat'][coef]['bvj']['last'])

            labels.append(f'{num_superNsub[coef]}')

        axs[index].bar(labels,np.array(b)-offset,color=c[len(num_superNsub)*index:],bottom = offset)
        # plt.xticks(rotation=-45)
        axs[index].axhline(len(all_lines[0][line_type]['iter'][0]['no precond']),color='k',linestyle='-',label='All')
        # axs[index].axhline(len(all_lines[0][line_type]['iter'][0]['no precond'])/2,color='k',linestyle='--',label='Half')
        axs[index].legend()
        axs[index].set_title(f'Seed: {index}')
        # axs[index].xlabel('Num super n sub diags, (seed)')
    fig.suptitle('The number of systems with lower solver iteration count\n compared with Jacobi')
    fig.supxlabel('Number of super- and subdiagonals')



    plt.show()
