import torch

x = torch.tensor(2.,requires_grad=True)

f=torch.exp(-x**2 - 2*x - torch.sin(x))

def analytical_gradient():
	dx=f*(-2*x - 2 - torch.cos(x))
	print(f"Analytical : dx : {dx}")
analytical_gradient()
f.backward()
print(f"Autograd : dx : {x.grad}")
