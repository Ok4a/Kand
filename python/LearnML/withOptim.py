import torch
import math

#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim

dtype = torch.float

device = "cpu"#torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
torch.set_default_device(device)

x = torch.linspace(-math.pi, math.pi, 2000,  dtype=dtype)
y = torch.sin(x)

p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

loss_fn = torch.nn.MSELoss(reduction="sum")


lr = 1e-3

optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)

for t in range(2000):

    y_pred= model(xx)
    
    loss = loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

linear_layer = model[0]

print(f'y={linear_layer.bias.item()}+{linear_layer.weight[:,0].item()}x+{linear_layer.weight[:,1].item()}x^2+{linear_layer.weight[:,2].item()}x^3')