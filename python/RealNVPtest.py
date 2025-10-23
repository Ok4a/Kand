# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
import math
import linearSolver as ls

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
    def _comp_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fn1(x*self.mask))
        s = torch.relu(self.scale_fn2(s))
        s = torch.relu(self.scale_fn3(s)) * self.scale        
        return s

    def _comp_trans(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.trans_fn1(x*self.mask))
        t = torch.relu(self.trans_fn2(t))
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

        x = self.mask * y + (1 - self.mask) * (y - t) *  torch.exp(-s)
        log_determinant = torch.sum((1 - self.mask) * (-s), -1)


        return x, log_determinant

class realNVP(nn.Module):
    '''
    A vanilla RealNVP class for modeling 2 dimensional distributions
    '''
    def __init__(self, masks, hidden_dim):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(realNVP, self).__init__()        
        self.hidden_dim = hidden_dim        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [affine_coupling(self.masks[i], self.hidden_dim)
             for i in range(len(self.masks))])
        
    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)        
        y = 4*torch.tanh(y)
        logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables        
        x = y        
        logdet_tot = 0

        # inverse the normalization layer
        logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        x  = 0.5*torch.log((1+x/4)/(1-x/4))
        logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot

if __name__ == "__main__":
    dim = 100
    masks_ = [[0]*math.ceil(dim/2)+[1]*math.floor(dim/2),
         [1]*math.ceil(dim/2)+[0]*math.floor(dim/2)]

    ## dimenstion of hidden units used in scale and translation transformation
    hidden_dim = 128

    ## construct the RealNVP_2D object
    # if torch.cuda.device_count():
    #     realNVP = realNVP.cuda()
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(device)
    # torch.set_default_device(device)


    realNVP = realNVP(masks_, hidden_dim)

    device = next(realNVP.parameters()).device

    optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)

    num_steps = 5000
    # num_steps = 5000

    ## the following loop learns the RealNVP_2D model by data
    ## in each loop, data is dynamically sampled from the scipy moon dataset
    for idx_step in range(num_steps):

        A,b = ls.randAb(dim, normal=True)
            