import torch

b = torch.tensor(1.,requires_grad=True)
w = torch.tensor(1.,requires_grad=True)
x = torch.tensor(2.,requires_grad=True)

u=w*x
v=u+b
a=torch.sigmoid(v)

def analytical_gradient():
	dx=torch.sigmoid(v)*(1-torch.sigmoid(v))*1*w
	dw=torch.sigmoid(v)*(1-torch.sigmoid(v))*1*x
	db=torch.sigmoid(v)*(1-torch.sigmoid(v))*1

	print(f"Analytical : dx : {dx} dw : {dw} db : {db}")
analytical_gradient()
a.backward()
print(f"Autograd : dx : {x.grad} dw : {w.grad} db : {b.grad}")
