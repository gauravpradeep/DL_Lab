import torch

a = torch.tensor(1.,requires_grad=True)
b = torch.tensor(2.,requires_grad=True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

def analytical_gradient():
	da = 2*2 + 3*10*a
	db = 2*3 + 3*9*b**2
	print(f"Analytical : da : {da} db : {db}")
analytical_gradient()

	
z.backward()
print(f"Autograd : da : {a.grad} db : {b.grad}")
