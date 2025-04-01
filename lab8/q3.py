import torch
import torch.nn as nn
import numpy as np

# Sample text data
text = "hello this is a simple next character prediction example. it is fun!"

# Create a set of unique characters in the text
chars = sorted(set(text))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}  # Map char to index
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}  # Map index to char

# Prepare the training sequences
sequence_length = 10  # We will use 10 previous characters to predict the next one
data_in = []
data_out = []

for i in range(len(text) - sequence_length):
    seq_in = text[i:i+sequence_length]
    seq_out = text[i+sequence_length]
    data_in.append([char_to_idx[ch] for ch in seq_in])
    data_out.append(char_to_idx[seq_out])

# Convert the input and output data to tensors
X = torch.tensor(data_in, dtype=torch.long)
y = torch.tensor(data_out, dtype=torch.long)

# One-hot encode the inputs
def one_hot_encode(x, vocab_size):
    return torch.eye(vocab_size)[x]

# Convert inputs to one-hot encoded tensors
X_one_hot = torch.stack([one_hot_encode(x, len(chars)) for x in X])

print(f'Input Tensor Shape: {X_one_hot.shape}, Output Tensor Shape: {y.shape}')

# Define the RNN Model for Next Character Prediction
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # Get the RNN outputs
        rnn_out = rnn_out[:, -1, :]  # Get the last output (for sequence prediction)
        output = self.fc(rnn_out)
        return output

# Model parameters
input_size = len(chars)  # Size of the vocabulary
hidden_size = 128        # Number of hidden units in the RNN
output_size = len(chars) # Size of the vocabulary (output size)

# Initialize the RNN model
model = RNNModel(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    output = model(X_one_hot)
    
    # Compute loss
    loss = criterion(output, y)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to predict the next character
def predict(model, input_sequence, char_to_idx, idx_to_char, sequence_length=10):
    model.eval()  # Set model to evaluation mode
    input_indices = [char_to_idx[ch] for ch in input_sequence]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Add batch dimension
    
    # One-hot encode the input
    input_tensor = one_hot_encode(input_tensor, len(chars))

    # Get the model's prediction
    output = model(input_tensor)
    
    # Convert the output to a predicted character
    _, predicted_idx = torch.max(output, 1)
    predicted_char = idx_to_char[predicted_idx.item()]
    return predicted_char

# Test the model on an example sequence
test_input = "hello this is a "
for i in range(10):
    predicted_char = predict(model, test_input, char_to_idx, idx_to_char)
    print(f"Input: {test_input}")
    print(f"Predicted Next Character: {predicted_char}")
    # test_input=test_input[1:]
    test_input=test_input+predicted_char
# print(type(predicted_char))