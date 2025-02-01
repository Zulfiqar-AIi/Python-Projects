import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Define the Neural Network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to hidden layer (2 inputs, 4 hidden neurons)
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer (4 hidden neurons, 1 output)
        self.sigmoid = nn.Sigmoid()  # Activation function

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Apply sigmoid to hidden layer
        x = self.sigmoid(self.fc2(x))  # Apply sigmoid to output layer
        return x

# Step 2: Prepare the XOR Dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Step 3: Initialize the Model, Loss Function, and Optimizer
model = XORNet()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Step 4: Train the Model
loss_history = []
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, outputs)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store loss for plotting
    loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Plot the Loss Curve
plt.plot(loss_history)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Step 6: Evaluate the Model's Accuracy
with torch.no_grad():
    predictions = model(inputs)
    predicted_labels = (predictions > 0.5).float()  # Convert probabilities to binary outputs
    accuracy = (predicted_labels == outputs).float().mean() * 100

print(f'Accuracy: {accuracy:.2f}%')

# Print predictions
print("Predictions:")
print(predictions.detach().numpy())