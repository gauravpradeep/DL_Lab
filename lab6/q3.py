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
			nn.Conv2d(1,64,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			nn.Conv2d(64,128,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			nn.Conv2d(128,64,3),
			nn.ReLU(),
			nn.MaxPool2d((2,2),stride=2),
			)
		self.linear = nn.Sequential(
			nn.Linear(64,20,bias=True),
			nn.ReLU(),
			nn.Linear(20,10,bias=True))

	def forward(self,x):
		x=self.cnn(x)
		x = x.flatten(start_dim=1)
		x=self.linear(x)
		return x



device=torch.device('cuda')
model=MNIST_CNN().to(device)

check_point = torch.load('../lab5/mnist_checkpoint.pt')
model.load_state_dict(check_point['model_state'])

loss_fn =torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)
batch_size=32
optimizer.load_state_dict(check_point['optimizer_state'])
loss = check_point['last_loss']
epochs=check_point['last_epoch']

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


new_epochs=10
for epoch in range(epochs,new_epochs):
	epoch_loss=0
	for i,(inputs,labels) in enumerate(trainloader):
		inputs=inputs.to(device)
		labels=labels.to(device)
		optimizer.zero_grad()
		outputs=model(inputs)
		loss=loss_fn(outputs,labels)
		epoch_loss+=loss
		loss.backward()
		optimizer.step()

	epoch_loss/=len(trainloader)
	print(f"Epoch {epoch+1} loss : {epoch_loss.item()}")


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

conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

'''
Epoch 6 loss : 0.061123915016651154
Epoch 7 loss : 0.05345950275659561
Epoch 8 loss : 0.04731995612382889
Epoch 9 loss : 0.043179597705602646
Epoch 10 loss : 0.038858961313962936
Confusion Matrix:
[[ 970    1    1    0    0    0    3    2    0    3]
 [   0 1127    0    1    0    2    3    2    0    0]
 [   8    0 1003    0    2    0    1   15    2    1]
 [   1    0    3  987    0   12    0    4    1    2]
 [   1    0    1    1  969    0    3    0    0    7]
 [   3    0    0    2    0  881    1    1    3    1]
 [   4    1    0    0    2    4  946    0    1    0]
 [   1    2    7    0    0    1    0 1012    2    3]
 [   4    0    1    0    3    2    1    2  955    6]
 [   3    0    0    0    5    4    0    1    2  994]]
Accuracy: 98.44%

'''