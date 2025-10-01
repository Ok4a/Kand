import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import math


torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

class affine_coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super().__init__()

        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        self.mask = nn.Parameter(mask, requires_grad=False)

        # example
        self.scale_fn1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fn2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fn3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        nn.init.normal_(self.scale)

        self.trans_fn1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.trans_fn2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trans_fn3 = nn.Linear(self.hidden_dim, self.input_dim)

    # example
    def _comp_scale(self,x):
        s = self.scale_fn1(x*self.mask)
        s = self.scale_fn2(s)
        s = self.scale_fn3(s) * self.scale
        return s

    # example
    def _comp_trans(self,x):
        t = self.trans_fn1(x*self.mask)
        t = self.trans_fn2(t)
        t = self.trans_fn3(t)
        return t


    def forward(self,x):
        s = self._comp_scale(x)
        t = self._comp_trans(x)

        y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)

        log_determinant = torch.sum((1-self.mask)*(s),-1)

        return y, log_determinant


    def inverse(self, y):
        s = self._comp_scale(y)
        t = self._comp_trans(y)

        x = self.mask * y + (1-self.mask) * (y- t) *  torch.exp(-s)
        log_determinant = torch.sum((1-self.mask)*(-s),-1)


        return x, log_determinant

class realNVP(nn.Module):
    def __init__(self, masks, hidden_dim):
        super(realNVP, self).__init__()
        self.hidden_dim = hidden_dim

        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks])

        self.num_masks = len(self.masks)

        self.affine_couplings = nn.ModuleList([affine_coupling(self.masks[i], self.hidden_dim) for i in range(self.num_masks)])

        self.num_affine_couplings = len(self.affine_couplings)

    def forward(self, x):
        y = x
        log_determinant_total = 0
        for i in range(self.num_affine_couplings):
            y, log_determinant = self.affine_couplings[i](y)
            log_determinant_total += log_determinant


        return y, log_determinant_total
    
    def inverse(self, y):
        x = y
        log_determinant_total = 0
        for i in range(self.num_affine_couplings):
            x, log_determinant = self.affine_couplings[i].inverse(x)
            log_determinant_total += log_determinant


        return x, log_determinant_total


masks_ = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],
         [0.0, 1.0]]

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 128

## construct the RealNVP_2D object
# if torch.cuda.device_count():
#     realNVP = realNVP.cuda()
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
torch.set_default_device(device)


realNVP = realNVP(masks_, hidden_dim)

device = next(realNVP.parameters()).device

optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)

num_steps = 5000
# num_steps = 5000

## the following loop learns the RealNVP_2D model by data
## in each loop, data is dynamically sampled from the scipy moon dataset
for idx_step in range(num_steps):
    ## sample data from the scipy moon dataset
    X, label = datasets.make_moons(n_samples = 512, noise = 0.05)
    X = torch.Tensor(X).to(device = device)

    ## transform data X to latent space Z
    z, logdet = realNVP.inverse(X)

    ## calculate the negative loglikelihood of X
    loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    if (idx_step + 1) % 1000 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")
        