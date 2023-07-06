import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
import torch.optim as optim

# Synthetic data generation
x = np.linspace(0, 1, 100)  # Spatial grid
t = np.linspace(0, 1, 50)   # Temporal grid
X, T = np.meshgrid(x, t)
data = np.sin(np.pi * X) * np.exp(-T)
data += np.random.normal(0, 0.1, size=data.shape)  # Adding noise

# Convert data to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
T_tensor = torch.tensor(T, dtype=torch.float32)
data_tensor = torch.tensor(data, dtype=torch.float32)

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the neural network
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor, T_tensor)
    loss = criterion(output, data_tensor)
    loss.backward()
    optimizer.step()

# Compute gradients using central differences
dx = x[1] - x[0]
dt = t[1] - t[0]
grad_x = torch.autograd.grad(output, X_tensor, torch.ones_like(output), create_graph=True)[0]
grad_t = torch.autograd.grad(output, T_tensor, torch.ones_like(output), create_graph=True)[0]
grad_xx = torch.autograd.grad(grad_x, X_tensor, torch.ones_like(grad_x), create_graph=True)[0]
grad_tt = torch.autograd.grad(grad_t, T_tensor, torch.ones_like(grad_t), create_graph=True)[0]

# Sparse regression to discover the PDE terms
features = torch.stack([output, grad_x, grad_t, grad_xx, grad_tt], dim=1)
features = features.detach().numpy().reshape(-1, 5)

# Remove noisy data points
indices = np.random.choice(range(features.shape[0]), size=int(features.shape[0] * 0.9), replace=False)
features = features[indices]

# Create target variable
target = np.zeros((features.shape[0], 5))

# Generate the PDEs
target[:, 0] = -features[:, 2]  # dU/dt
target[:, 1] = -features[:, 1]  # dU/dx
target[:, 2] = -features[:, 3]  # d^2U/dx^2
target[:, 3] = -features[:, 4Sure, here's the continuation of the script:

```python
# Sparse regression using Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(features, target)

# Extract coefficients
coefficients = lasso.coef_

# Print the discovered PDE terms
print("Discovered PDE terms:")
print("dU/dt =", coefficients[0])
print("dU/dx =", coefficients[1])
print("d^2U/dx^2 =", coefficients[2])
print("d^2U/dt^2 =", coefficients[3])
# Add more discovered terms as needed based on the complexity of the problem

# Plotting the discovered PDE terms
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), coefficients, tick_label=['dU/dt', 'dU/dx', 'd^2U/dx^2', 'd^2U/dt^2'])
plt.xlabel('PDE Term')
plt.ylabel('Coefficient')
plt.title('Discovered PDE Terms')
plt.show()
