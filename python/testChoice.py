import numpy as np


rng = np.random.default_rng()



num_super = 5

par_list = range(1,6)

A = rng.choice(num_super)
B = rng.choice(par_list[A])
print(A,B)