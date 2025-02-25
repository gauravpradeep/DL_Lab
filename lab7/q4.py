import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p  # Probability of dropping a unit (i.e., setting it to zero)
    
    def forward(self, x):
        # During training, apply dropout
        if self.training:
            # Create a random mask with the Bernoulli distribution
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p)).to(x.device)
            # Scale the mask to keep the expected value
            return x * mask / (1 - self.p)
        else:
            # During inference, no dropout, just return the input
            return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 224x224 -> 224x224
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 224x224 -> 224x224
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 224x224 -> 224x224
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces the spatial dimensions by half
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        # Dropout layers
        self.dropout1 = CustomDropout(0.5)  # Custom Dropout after first fully connected layer
        self.dropout2 = CustomDropout(0.5)  # Custom Dropout after second fully connected layer
        self.dropout3 = CustomDropout(0.5)  # Custom Dropout after third fully connected layer


    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten the output from (batch_size, 128, 28, 28) to (batch_size, 128*28*28)
        
        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after fc1
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after fc2
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after fc3
        x = self.fc4(x)  # Output layer
        
        return x


transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}
device = torch.device('cuda')
data_dir = 'cats_and_dogs_filtered'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform['train'])
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = CustomCNN(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

epochs = 10

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
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


train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Final Accuracy on the validation set: {accuracy * 100:.2f}%")

'''
Custom Dropout:
Epoch 1/10, Loss: 0.7061
Epoch 2/10, Loss: 0.6883
Epoch 3/10, Loss: 0.6703
Epoch 4/10, Loss: 0.6757
Epoch 5/10, Loss: 0.6477
Epoch 6/10, Loss: 0.6665
Epoch 7/10, Loss: 0.6563
Epoch 8/10, Loss: 0.6360
Epoch 9/10, Loss: 0.6318
Epoch 10/10, Loss: 0.6370
Final Accuracy on the validation set: 73.20%

'''