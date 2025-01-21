import torch
import matplotlib.pyplot as	plt

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
lr = torch.tensor(0.001)

class RegressionModel:
	def __init__(self):
		self.w = torch.tensor([1.],requires_grad=True)
		self.b = torch.tensor([1.],requires_grad=True)	

	def forward(self,x):
		return self.w*x + self.b

	def update(self):
		self.w -= lr*self.w.grad
		self.b -= lr*self.b.grad

	def reset_grad(self):
		self.w.grad.zero_()
		self.b.grad.zero_()


def criterion(y,yp):
	return (y-yp)**2

model=RegressionModel()
loss_list=[]

for epochs in range(100):
	loss=0.0
	for i in range(len(x)):
		yp=model.forward(x[i])
		loss+=criterion(y[i],yp)
	loss/=len(x)
	loss_list.append(loss.item())
	loss.backward()

	with torch.no_grad():
		model.update()

	model.reset_grad()

	print(f"W : {model.w.item()} B : {model.b.item()} Loss : {loss.item()}")

plt.plot(list(range(100)),loss_list)
plt.show()
print(f"FINAL W : {model.w.item()} B : {model.b.item()} Loss : {loss.item()}")

