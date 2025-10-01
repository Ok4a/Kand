import torch
import math


#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors
dtype = torch.float

device = torch.device("cpu")


x = torch.linspace(-math.pi,math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)


a = torch.randn((),device=device, dtype=dtype)
b = torch.randn((),device=device, dtype=dtype)
c = torch.randn((),device=device, dtype=dtype)
d = torch.randn((),device=device, dtype=dtype)

lr = 1e-6

for t in range(200):
    y_pred = a+b*x+c*x**2+d*x**3

    loss = (y_pred-y).pow(2).sum().item()
    if t % 100 == 99:
        print(t,loss)
    
    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()

    a -= lr*grad_a
    b -= lr*grad_b
    c -= lr*grad_c
    d -= lr*grad_d

print(f'y={a}+{b}x+{c}x^2+{d}x^3')