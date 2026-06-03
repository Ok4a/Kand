from itertools import product

# [Data]
D_laplace_size = 65
amount = 250
dist_type_data = 'uniform'
params_data = '1 0.45'
seeds = range(4)
# seed = 5
# dim =

# [Solver]
tol = 1e-10
max_iteration = None

# [Learn]
iteration_count = 200
step_range = 0.1
# methods = ['sign','mean', 'median']
methods = ['sign']
# seed = 5
change_scales = '1 0.5 0.25'
allow_disimprovement = True
median_best_scale = 0.5
mean_best_scale = 0.5
flag_count_limit = 0

# [Precondition]
dist_type_pre = 'normal'
params_pre = '0 0.05'
par_list = '1 1'
diag_list = '1 -1'
# diags = [1,2,3,4]
# num_coef = 
type = 'super_sub_shift'
type = 'diag_shift'
diag_type = 'jacobi'
sym = True



# num_coef_list = [2,4]
# precond_type = 'super_sub_shift_jacobi'





for method, seed in product(methods, seeds):

    with open(f'config/config_{method}_{seed}_{D_laplace_size}.ini', mode = 'w') as config_file:
        config_file.write(f'[Data]')
        config_file.write(f'\n1D_laplace_size = {D_laplace_size}')
        config_file.write(f'\namount = {amount}')
        config_file.write(f'\ndist_type_data = {dist_type_data}')
        config_file.write(f'\nparams = {params_data}')
        config_file.write(f'\nseed = {seed}')
        config_file.write(f'\ndim =')

        config_file.write(f'\n\n[Solver]')
        config_file.write(f'\ntol = {tol}')
        config_file.write(f'\nmax_iteration = {max_iteration}')

        config_file.write(f'\n\n[Learn]')
        config_file.write(f'\niteration_count = {iteration_count}')
        config_file.write(f'\nstep_range = {step_range}')
        config_file.write(f'\nmethod = {method}')
        config_file.write(f'\nseed = {seed}')
        config_file.write(f'\nchange_scales = {change_scales}')
        config_file.write(f'\nallow_disimprovement = {allow_disimprovement}')
        config_file.write(f'\nmedian_best_scale = {median_best_scale}')
        config_file.write(f'\nmean_best_scale = {mean_best_scale}')
        config_file.write(f'\nflag_count_limit = {flag_count_limit}')

        config_file.write(f'\n\n[Precondition]')

        config_file.write(f'\ndist_type_pre = {dist_type_pre}')
        config_file.write(f'\nparams_pre = {params_pre}')
        config_file.write(f'\npar_list = {par_list}')
        config_file.write(f'\ndiag_list = {diag_list}')
        config_file.write(f'\nnum_coef = ')
        config_file.write(f'\ntype = {type}')
        config_file.write(f'\ndiag = {diag_type}')
        config_file.write(f'\nsym = {sym}')