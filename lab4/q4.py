import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

batch_size = 32
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

class FFN(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1=nn.Linear(28*28,100, bias=True)
		self.linear2=nn.Linear(100,100, bias=True)
		self.linear3=nn.Linear(100,10, bias=True)
		self.relu=nn.ReLU()

	def forward(self,x):
		x=x.view(-1,28*28)
		x=self.linear1(x)
		x=self.relu(x)
		x=self.linear2(x)
		x=self.relu(x)
		x=self.linear3(x)

		return x

device = torch.device("cuda")
model=FFN().to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)
loss_list=[]

epochs = 5
for epoch in range(epochs):
	epoch_loss=0
	for i,(inputs,labels) in enumerate(trainloader):
		inputs=inputs.to(device)
		labels=labels.to(device)
		optimizer.zero_grad()
		outputs=model(inputs)
		loss=criterion(outputs,labels)
		epoch_loss+=loss
		loss.backward()
		optimizer.step()

	epoch_loss/=len(trainloader)
	print(f"Epoch {epoch+1} loss : {epoch_loss.item()}")
	loss_list.append(epoch_loss.item())

plt.plot(list(range(epochs)),loss_list)
plt.show()
all_preds = []
all_labels = []		
with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs, 1)
			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of learnable parameters in the model: {num_params}")

'''
Epoch 1 loss : 0.5691441893577576
Epoch 2 loss : 0.2391878068447113
Epoch 3 loss : 0.18051466345787048
Epoch 4 loss : 0.14563588798046112
Epoch 5 loss : 0.12206336110830307
Confusion Matrix:
[[ 963    0    1    1    1    5    4    1    2    2]
 [   0 1115    3    1    0    1    4    1   10    0]
 [   6    1 1000    2    4    0    5    6    6    2]
 [   0    0    4  970    0   11    1   10    9    5]
 [   1    0    8    0  936    1    3    2    2   29]
 [   4    1    1   11    1  855    8    1    4    6]
 [   9    3    0    0    5    8  928    1    4    0]
 [   0    5   16    1    2    1    0  989    3   11]
 [   4    0    1   16    3    6    5    9  927    3]
 [   6    5    2    7   15    5    0    7    1  961]]
Total number of learnable parameters in the model: 89610

'''