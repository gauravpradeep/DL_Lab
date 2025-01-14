import torch
x=torch.tensor(2.0, requires_grad=True)
y=8*x**4+3*x**3+7*x**2+6*x+3
ana = 32*x**3+9*x**2+14*x+6
print("Analytically: ",ana)
y.backward()
print("By pytorch: ",x.grad)