import numpy as np
import math

#https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#warm-up-numpy
x = np.linspace(-math.pi,math.pi, 2000)
y = np.sin(x)


a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

lr = 1e-6

for t in range(2000):
    y_pred = a+b*x+c*x**2+d*x**3

    loss = np.square(y_pred-y).sum()
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