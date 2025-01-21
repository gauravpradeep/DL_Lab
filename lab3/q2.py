import torch
import matplotlib.pyplot as plt

x = torch.tensor( [2.,4.])
y = torch.tensor( [20.,40.])

w = torch.tensor([1.],requires_grad=True)
b = torch.tensor([1.],requires_grad=True)

lr = torch.tensor(0.001)

loss_list = []

for epochs in range(2):
	loss=0.0
	for i in range(len(x)):
		yp = w*x[i] + b
		loss=loss+(y[i]-yp)**2
	loss=loss/len(x)
	loss_list.append(loss.item())
	loss.backward()

	with torch.no_grad():
		w-=lr*w.grad
		b-=lr*b.grad

	w.grad.zero_()
	b.grad.zero_()

plt.plot(list(range(2)),loss_list)
plt.show()
print(f"PYTORCH AUTOGRAD W : {w} B : {b}")

w = torch.tensor([1.])
b = torch.tensor([1.])


for epochs in range(2):
	for i in range(len(x)):
		yp = w*x[i] + b
		error=yp-y[i]
		w-=lr*(error)*x[i]
		b-=lr*(error)

print(f"ANALYTICAL SOLN W : {w} B : {b}")
