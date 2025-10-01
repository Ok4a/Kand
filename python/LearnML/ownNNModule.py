import torch
import math

#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules


class Poly3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self,x):
        return self.a +self.b*x+self.c*x**2+self.d*x**3
    
    def string(self):
        return f'y={self.a.item()}+{self.b.item()}x+{self.c.item()}x^2+{self.d.item()}x^3'

dtype = torch.float

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
torch.set_default_device(device)

x = torch.linspace(-math.pi, math.pi, 2000,  dtype=dtype)
y = torch.sin(x)


model = Poly3()


criterion = torch.nn.MSELoss(reduction="sum")


lr = 1e-6

optimizer = torch.optim.SGD(model.parameters(),lr=lr)

for t in range(2000):

    y_pred= model(x)
    
    loss = criterion(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


print(model.string())