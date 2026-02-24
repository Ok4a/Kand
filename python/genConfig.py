from itertools import product
seeds = range(4,7)

# [Data]
D_laplace_size = 45
amount = 250
params = '1 0.3'
# seed = 5
# dim =

# [Solver]
tol = 1e-10
max_iteration = None

# [Learn]
iteration_count = 100
step_range = 0.1
methods = ['sign','mean', 'median']
methods = ['sign']
# seed = 5
scale = '1 0.5 0.25'

# [Precondition]
num_coef = 10





for method, seed in product(methods, seeds):
    with open(f'config/config_{method}_{seed}.ini', mode = 'w') as config_file:
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
        config_file.write(f'\nscale = {scale}')

        config_file.write(f'\n\n[Precondition]')
        config_file.write(f'\nnum_coef = {num_coef}')