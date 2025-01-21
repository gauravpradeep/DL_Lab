import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Sample data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)  # Shape: (5, 1)
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)  # Shape: (5, 1)

# Hyperparameters
lr = 0.001
epochs = 100

# Define the regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))  # Weight
        self.b = nn.Parameter(torch.randn(1))
        # self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.w*x + self.b
        # return self.linear(x)

# Create a custom Dataset class
class RegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Create the dataset and data loader
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# List to track loss values
loss_list = []

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Backpropagate the loss
        optimizer.step()       # Update parameters

        epoch_loss += loss.item()

    # Average loss for this epoch
    avg_loss = epoch_loss / len(dataloader)
    loss_list.append(avg_loss)

    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Plot the loss over epochs
plt.plot(range(epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# Final learned parameters
print(f"Final W (Weight): {model.w.item()}")
print(f"Final B (Bias): {model.b.item()}")
