import torch
import torch.nn as nn
import numpy as np

# Generate numbers from 1 to 20
numbers = np.arange(1, 21)

# Prepare input and target sequences
sequence_length = 5
input_sequences = []
target_sequences = []

for i in range(len(numbers) - sequence_length):
    input_seq = numbers[i:i + sequence_length]
    target_seq = numbers[i + sequence_length]
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

# Convert to numpy arrays
X = np.array(input_sequences)
y = np.array(target_sequences)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Shape: (batch_size, sequence_length, input_size)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # Shape: (batch_size, 1)

# Define the RNN model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output size is 1 for the next number prediction

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch_size, sequence_length, hidden_dim)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out

# Hyperparameters
input_size = 1  # Since we are predicting a single number
hidden_dim = 10
num_epochs = 1000
learning_rate = 0.01
batch_size = 4

# Initialize the model
model = LSTMModel(input_size, hidden_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to predict the next number
def predict_next_number(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
        output = model(input_tensor)
        return output.item()

# Test the model with a sequence
# test_input = [16, 17, 18, 19, 20]  # Example input sequence
test_input = [10,11,12,13,14]
predicted_number = predict_next_number(model, test_input)
print(f'Input sequence: {test_input}, Predicted next number: {predicted_number:.2f}')