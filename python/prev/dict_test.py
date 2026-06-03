import numpy as np
from copy import deepcopy
index_list = [1,-1,2,-2]

the_dict = {}
copy_dict = deepcopy(the_dict)


rng = np.random.default_rng(1)
for i in index_list:
    the_dict[i] = rng.normal(scale = 0.05, size=1)

copy_dict = deepcopy(the_dict)

for i in index_list:
    the_dict[i] += rng.normal(scale = 0.05, size=1)

print(the_dict)
print(copy_dict)