import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the Neural Network model
class XOR_Network(nn.Module):
    def __init__(self):
        super(XOR_Network, self).__init__()
        self.hidden = nn.Linear(2, 4)  # Hidden layer with 4 neurons
        self.output = nn.Linear(4, 1)  # Output layer with 1 neuron
        self.activation = nn.Sigmoid() # Activation function

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

# Instantiate the model
model = XOR_Network()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 5000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()   # Reset gradients
    outputs = model(X)      # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()         # Backward pass
    optimizer.step()        # Update weights
    
    losses.append(loss.item())  # Store loss for plotting
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

# Evaluate the model
with torch.no_grad():
    predictions = model(X)
    predictions = (predictions > 0.5).float()  # Convert to binary output
    accuracy = (predictions == y).float().mean()
    print(f"Model Accuracy: {accuracy.item() * 100:.2f}%")
