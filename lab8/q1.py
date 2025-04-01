import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
df = pd.read_csv("daily.csv")
# Preprocess the data - Drop NA values in the dataset
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y), 1)
# Normalize the input range between 0 and 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)
Sequence_Length = 10
X = []
Y = []
for i in range(0, 5900):
	list1 = []
	for j in range(i, i + Sequence_Length):
		list1.append(y[j])
	X.append(list1)
	Y.append(y[j + 1])

X = np.array(X)
Y = np.array(Y)
#Split the data as the train and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.10, random_state=42, shuffle=False, stratify=None)
class NGTimeSeries(Dataset):
	def __init__(self, x, y):	
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)
		self.len = x.shape[0]
	
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
	
	def __len__(self):
		return self.len
dataset = NGTimeSeries(x_train,y_train)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset,shuffle=True,batch_size=256)


class RNNModel(nn.Module):
	def __init__(self):
		super(RNNModel,self).__init__()
		self.rnn = nn.RNN(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
		self.fc1 = nn.Linear(in_features=5,out_features=1)
	def forward(self,x):
		output,_status = self.rnn(x)
		output = output[:,-1,:]
		output = self.fc1(torch.relu(output))
		return output

model = RNNModel()
# optimizer , loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1500

for i in range(epochs):
	for j, data in enumerate(train_loader):
		y_pred = model(data[:][0].view(-1, Sequence_Length,1)).reshape(-1)
		loss = criterion(y_pred, data[:][1])
		loss.backward()
		optimizer.step()
	if i % 50 == 0:
		print(i, "th iteration : ", loss)
#test set actual vs predicted
test_set = NGTimeSeries(x_test,y_test)
test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)
plt.plot(test_pred.detach().numpy(),label='predicted')
plt.plot(test_set[:][1].view(-1),label='original')
plt.legend()
plt.show()
#Undo normalization
y = y * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm
plt.plot(y)
plt.plot(range(len(y)-len(y_pred), len(y)), y_pred)
plt.show()