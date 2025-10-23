import linearSolver as ls
from RealNVPtest import *
import math
import numpy as np
import torch
import torch.optim as optim

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

dim = 100



masks = [[0]*math.ceil(dim/2)+[1]*math.floor(dim/2),
         [1]*math.ceil(dim/2)+[0]*math.floor(dim/2)]

model = realNVP(masks=masks, hidden_dim=128)

device = next(realNVP.parameters()).device


opti = optim.Adam(realNVP.parameters(), lr = 1e-5)
steps = 10
for i in range(steps):

    A,b = ls.randAb(dim, normal=True)
