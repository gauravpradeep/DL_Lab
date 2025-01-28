import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

loss_list = []
torch.manual_seed(42)

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([0,1,1,0], dtype=torch.float32)

class XORModel(nn.Module):
	def __init__(self):
		super(XORModel, self).__init__()
		self.linear1=nn.Linear(2,2,bias=True)
		self.activation1=nn.Sigmoid()
		self.linear2=nn.Linear(2,1,bias=True)

	def forward(self,x):
		x=self.linear1(x)
		x=self.activation1(x)
		x=self.linear2(x)

		return x

class XORDataset(Dataset):
	def __init__(self,X,Y):
		self.X=X
		self.Y=Y

	def __len__(self):
		return len(self.X)

	def __getitem__(self,idx):
		return self.X[idx],self.Y[idx]

xor_dataset=XORDataset(X,Y)
train_dataloader = DataLoader(xor_dataset,batch_size=1,shuffle=True)
device = torch.device("cuda")
model=XORModel().to(device)

criterion=torch.nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

epochs = 50000
for epoch in range(epochs):
	epoch_loss=0
	for i,(inputs,labels) in enumerate(train_dataloader):
		inputs=inputs.to(device)
		labels=labels.to(device)
		optimizer.zero_grad()
		outputs=model(inputs)
		loss=criterion(outputs.flatten(),labels)
		epoch_loss+=loss
		loss.backward()
		optimizer.step()

	epoch_loss/=len(train_dataloader)
	loss_list.append(epoch_loss.item())

model.eval()
with torch.no_grad():
	for i, (inputs,labels) in enumerate(train_dataloader):
		inputs=inputs.to(device)
		outputs=model(inputs)
		print(inputs, outputs)

for param in model.named_parameters():
	print(param)

plt.plot(list(range(epochs)),loss_list)
plt.show()

'''
tensor([[0., 0.]], device='cuda:0') tensor([[1.8477e-06]], device='cuda:0')
tensor([[0., 1.]], device='cuda:0') tensor([[1.0000]], device='cuda:0')
tensor([[1., 0.]], device='cuda:0') tensor([[1.0000]], device='cuda:0')
tensor([[1., 1.]], device='cuda:0') tensor([[7.7486e-06]], device='cuda:0')
('linear1.weight', Parameter containing:
tensor([[1.6543, 1.6615],
        [3.1086, 3.1313]], device='cuda:0', requires_grad=True))
('linear1.bias', Parameter containing:
tensor([-2.4590, -0.6196], device='cuda:0', requires_grad=True))
('linear2.weight', Parameter containing:
tensor([[-3.1000,  2.9882]], device='cuda:0', requires_grad=True))
('linear2.bias', Parameter containing:
tensor([-0.8013], device='cuda:0', requires_grad=True))
'''