import torch
import math
#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors-and-autograd

dtype = torch.float

device = "cpu"#torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
torch.set_default_device(device)

x = torch.linspace(-math.pi,math.pi, 2000,  dtype=dtype)
y = torch.sin(x)


a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

lr = 1e-6

for t in range(2000):
    y_pred = a+b*x+c*x**2+d*x**3

    loss = (y_pred-y).pow(2).sum()
    if t % 100 == 99:
        print(t,loss.item())
    
    loss.backward()

    with torch.no_grad():
        a -= lr*a.grad
        b -= lr*b.grad
        c -= lr*c.grad
        d -= lr*d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'y={a}+{b}x+{c}x^2+{d}x^3')