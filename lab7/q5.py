import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

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
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten the output from (batch_size, 128, 28, 28) to (batch_size, 128*28*28)
        
        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = CustomCNN(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Early stopping parameters
patience = 3  # Number of epochs to wait for improvement in validation loss before stopping
best_val_loss = float('inf')  # Initialize to a large value
epochs_no_improve = 0  # Counter for epochs without improvement
best_model_wts = model.state_dict()  # Save the model weights that achieved the best validation performance

epochs = 10

def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3):
    global epochs_no_improve, best_val_loss, best_model_wts
    
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        # Early stopping: Check if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict()  # Save best model weights
            epochs_no_improve = 0  # Reset counter if improvement is seen
        else:
            epochs_no_improve += 1
        
        # If no improvement for 'patience' epochs, stop training
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print("Training complete. Best model loaded.")

train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, patience=patience)

# Evaluate the model on validation data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Final Accuracy on the validation set: {accuracy * 100:.2f}%")


'''
Epoch 1/10, Loss: 0.6996
Validation Loss: 0.6779, Accuracy: 62.00%
Epoch 2/10, Loss: 0.6919
Validation Loss: 0.6893, Accuracy: 53.70%
Epoch 3/10, Loss: 0.6908
Validation Loss: 0.6570, Accuracy: 59.00%
Epoch 4/10, Loss: 0.6701
Validation Loss: 0.6526, Accuracy: 59.90%
Epoch 5/10, Loss: 0.6582
Validation Loss: 0.6515, Accuracy: 61.60%
Epoch 6/10, Loss: 0.6543
Validation Loss: 0.6417, Accuracy: 62.20%
Epoch 7/10, Loss: 0.6601
Validation Loss: 0.6452, Accuracy: 62.10%
Epoch 8/10, Loss: 0.6417
Validation Loss: 0.6375, Accuracy: 60.30%
Epoch 9/10, Loss: 0.6575
Validation Loss: 0.6455, Accuracy: 57.20%
Epoch 10/10, Loss: 0.6532
Validation Loss: 0.6209, Accuracy: 65.20%
Training complete. Best model loaded.
Final Accuracy on the validation set: 65.20%
'''