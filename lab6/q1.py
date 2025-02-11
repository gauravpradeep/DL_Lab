import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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

batch_size=32
mnist_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=False)

mnist_testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=MNIST_CNN()
model.load_state_dict(torch.load('../lab5/mnist_stateDict.pt',weights_only=True))
# model=torch.load('../lab5/mnist_model.pt'
model.to(device)

optimizer = optim.Adam(model.linear.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for param in model.cnn.parameters():
    param.requires_grad = False

# Ensure the classification head (linear layers) can still be trained
for param in model.linear.parameters():
    param.requires_grad = True

epochs=5
for epoch in range(epochs):
        # model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


model.eval()


print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
	print(param_tensor, "\t",model.state_dict()[param_tensor].size())
	print()

correct = 0
total = 0
for i, vdata in enumerate(test_loader):
	tinputs, tlabels = vdata
	tinputs = tinputs.to(device)
	tlabels = tlabels.to(device)
	toutputs = model(tinputs)

	_, predicted = torch.max(toutputs, 1)

	total += tlabels.size(0)
	correct += (predicted == tlabels).sum()
accuracy = 100.0 * correct / total
print("The overall accuracy is {}".format(accuracy))

'''
without finetuning:
The overall accuracy is 6.50012

with finetuning:
The overall accuracy is 75.66999816894531
'''