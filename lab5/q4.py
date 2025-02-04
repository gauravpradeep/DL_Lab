import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

class MNIST_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.cnn=nn.Sequential(
			nn.Conv2d(1,16,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			nn.Conv2d(16,32,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			nn.Conv2d(32,16,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			)
		self.linear = nn.Sequential(
			nn.Linear(16,10,bias=True))

	def forward(self,x):
		x=self.cnn(x)
		x = x.flatten(start_dim=1)
		x=self.linear(x)
		return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

batch_size = 32
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
device = torch.device("cuda")
model=MNIST_CNN().to(device)

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
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of learnable parameters in the model: {num_params}")


'''
Epoch 1 loss : 0.7137051224708557
Epoch 2 loss : 0.18160632252693176
Epoch 3 loss : 0.13135571777820587
Epoch 4 loss : 0.11009354144334793
Epoch 5 loss : 0.09738120436668396
Accuracy: 97.46%
Total number of learnable parameters in the model: 9594

'''
