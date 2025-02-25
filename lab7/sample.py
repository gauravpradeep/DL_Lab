import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import PIL
import glob
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models

class Gaussian(object):
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img + torch.normal(self.mean, self.var, img.size())

preprocess = T.Compose([
    T.ToTensor(),
    T.RandomHorizontalFlip(),
    T.RandomRotation(45),
    Gaussian(0, 0.15),
])

class MyDataset(Dataset):
    def __init__(self, transform=None, str="train"):
        self.imgs_path = "./cats_and_dogs_filtered/"+ str + "/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "\\*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = PIL.Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        if self.transform:
            img = self.transform(img)
        return img, class_id

dataset = MyDataset(transform=preprocess, str="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model (simple CNN for binary classification)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # Output 2 classes: cats and dogs
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass (gradient calculation)
        optimizer.step()  # Update model weights
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions * 100
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Optionally, visualize a few augmented images each epoch
    if (epoch + 1) % 2 == 0:  # Display images every 2 epochs
        i = 0
        for data in iter(dataloader):
            img, _ = data
            tI = T.ToPILImage()
            img = tI(img.squeeze())
            plt.imshow(img)
            plt.show()
            i += 1
            if i == 3:  # Display only 3 images
                break

# Save the model
torch.save(model.state_dict(), "simple_cnn_model.pth")
