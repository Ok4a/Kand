import torch
import math
#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-defining-new-autograd-functions
class LP3(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)

    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        return grad_outputs*1.5*(5*input**2-1)


dtype = torch.float

device = "cpu"#torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
torch.set_default_device(device)

x = torch.linspace(-math.pi,math.pi, 2000,  dtype=dtype)
y = torch.sin(x)


a = torch.full((),0.0, dtype=dtype, requires_grad=True)
b = torch.full((),-1.0, dtype=dtype, requires_grad=True)
c = torch.full((),0.0 ,dtype=dtype, requires_grad=True)
d = torch.full((),0.3, dtype=dtype, requires_grad=True)

lr = 1e-6

for t in range(2000):
    P3 = LP3.apply

    y_pred = a+b*P3(c+d*x)
    
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