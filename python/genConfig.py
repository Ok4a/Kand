from itertools import product

# [Data]
D_laplace_size = 45
amount = 250
params = '1 0.3'
seeds = range(1)
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

# [Precondition]
num_coef_list = range(1,10,2)
precond_type = 'super_shift_jacobi'





for method, seed, num_coef in product(methods, seeds, num_coef_list):
    with open(f'config/config_{method}_{seed}_numCoef_{num_coef}.ini', mode = 'w') as config_file:
        config_file.write(f'[Data]')
        config_file.write(f'\n1D_laplace_size = {D_laplace_size}')
        config_file.write(f'\namount = {amount}')
        config_file.write(f'\nparams = {params}')
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

        config_file.write(f'\n\n[Precondition]')
        config_file.write(f'\nnum_coef = {num_coef}')
        config_file.write(f'\ntype = {precond_type}')