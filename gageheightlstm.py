import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the LSTM-based PINN architecture
class RiverPINN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RiverPINN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Pass the input through the LSTM layers
        lstm_out, _ = self.lstm(x)
        # Pass the LSTM output through the fully connected layer
        output = self.fc(lstm_out)
        # Split the output into discharge and gage height
        discharge, gage_height = output[:, :, 0], output[:, :, 1]
        # Set requires_grad to True for discharge and gage height
        discharge.requires_grad_(True)
        gage_height.requires_grad_(True)
        return discharge, gage_height

# Define the loss function with Saint-Venant equations
def loss_function(discharge, gage_height, discharge_obs, gage_height_obs, x, y, bed_slope, precipitation, temperature, time, width, mannings_n):
    # Data loss: Mean squared error between predicted and observed discharge and gage height
    data_loss = torch.mean((discharge - discharge_obs)**2 + (gage_height - gage_height_obs)**2)

    # Physics loss - Continuity equation
    # Compute the gradients of gage height with respect to time
    gage_height_t = torch.autograd.grad(gage_height, time, grad_outputs=torch.ones_like(gage_height),
                                        create_graph=True, allow_unused=True)[0]
    # Compute the gradients of discharge with respect to x
    discharge_x = torch.autograd.grad(discharge, x, grad_outputs=torch.ones_like(discharge),
                                      create_graph=True, allow_unused=True)[0]

    # Handle None gradients by replacing them with zero tensors
    if gage_height_t is None:
        gage_height_t = torch.zeros_like(gage_height)
    if discharge_x is None:
        discharge_x = torch.zeros_like(discharge)

    # Compute the residual of the continuity equation
    continuity_residual = gage_height_t + discharge_x / width
    # Compute the mean squared error of the continuity residual
    continuity_loss = torch.mean(continuity_residual**2)

    # Physics loss - Momentum equation
    # Compute the velocity using discharge and gage height
    velocity = discharge / (gage_height * width)
    # Compute the gradients of velocity with respect to time
    velocity_t = torch.autograd.grad(velocity, time, grad_outputs=torch.ones_like(velocity),
                                     create_graph=True, allow_unused=True)[0]
    # Compute the gradients of gage height with respect to x
    gage_height_x = torch.autograd.grad(gage_height, x, grad_outputs=torch.ones_like(gage_height),
                                        create_graph=True, allow_unused=True)[0]
    # Compute the gradients of velocity with respect to x
    velocity_x = torch.autograd.grad(velocity, x, grad_outputs=torch.ones_like(velocity),
                                     create_graph=True, allow_unused=True)[0]

    # Handle None gradients by replacing them with zero tensors
    if velocity_t is None:
        velocity_t = torch.zeros_like(velocity)
    if gage_height_x is None:
        gage_height_x = torch.zeros_like(gage_height)
    if velocity_x is None:
        velocity_x = torch.zeros_like(velocity)

    # Constants
    g = 9.81  # Gravitational acceleration
    rho = 1000  # Water density

    # Bed slope and friction terms
    bed_slope_term = g * bed_slope
    friction_term = g * mannings_n**2 * velocity * torch.abs(velocity) / (gage_height**(4/3))

    # Compute the residual of the momentum equation
    momentum_residual = velocity_t + velocity * velocity_x + g * gage_height_x + bed_slope_term - friction_term
    # Compute the mean squared error of the momentum residual
    momentum_loss = torch.mean(momentum_residual**2)

    # Combine the continuity and momentum losses to get the physics loss
    physics_loss = continuity_loss + momentum_loss

    # Return the total loss (data loss + physics loss)
    return data_loss + physics_loss

# Training loop
def train(model, dataloader, num_epochs, learning_rate):
    # Define the optimizer (e.g., Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        for batch in dataloader:
            # Extract input data, observed discharge, and observed gage height from the batch
            input_data, discharge_obs, gage_height_obs = batch

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Predict discharge and gage height using the model
            discharge_pred, gage_height_pred = model(input_data)

            # Compute the loss using the predicted and observed values
            loss = loss_function(discharge_pred, gage_height_pred, discharge_obs, gage_height_obs,
                                 input_data[:, :, 0], input_data[:, :, 1], input_data[:, :, 2],
                                 input_data[:, :, 3], input_data[:, :, 4], input_data[:, :, 5],
                                 input_data[:, :, 6], input_data[:, :, 7])

            # Backward pass: Compute the gradients
            loss.backward(retain_graph=True)

            # Update the model parameters
            optimizer.step()

        # Print the loss every 100 epochs
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Generate hydrographs and stage graphs
def generate_graphs(model, x_locs, y_locs, bed_slope, precipitation, temperature, t_range, width, mannings_n):
    hydrographs = []
    stage_graphs = []

    for x, y in zip(x_locs, y_locs):
        # Create input tensors for the given location and time range
        x_tensor = torch.tensor([x] * len(t_range), dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor([y] * len(t_range), dtype=torch.float32).view(-1, 1)
        bed_slope_tensor = torch.tensor([bed_slope] * len(t_range), dtype=torch.float32).view(-1, 1)
        precipitation_tensor = torch.tensor([precipitation] * len(t_range), dtype=torch.float32).view(-1, 1)
        time_tensor = torch.tensor(t_range, dtype=torch.float32).view(-1, 1)
        width_tensor = torch.tensor([width] * len(t_range), dtype=torch.float32).view(-1, 1)
        temperature_tensor = torch.tensor([temperature] * len(t_range), dtype=torch.float32).view(-1, 1)
        mannings_n_tensor = torch.tensor([mannings_n] * len(t_range), dtype=torch.float32).view(-1, 1)

        # Concatenate the input tensors and add an extra dimension for the LSTM input
        input_data = torch.cat([x_tensor, y_tensor, bed_slope_tensor, precipitation_tensor, temperature_tensor,
                                time_tensor, width_tensor, mannings_n_tensor], dim=1).unsqueeze(0)

        # Predict discharge and gage height using the trained model
        with torch.no_grad():
            discharge_pred, gage_height_pred = model(input_data)

        # Append the predicted discharge and gage height to the lists
        hydrographs.append(discharge_pred.squeeze().detach().numpy())
        stage_graphs.append(gage_height_pred.squeeze().detach().numpy())

    return hydrographs, stage_graphs

# Load data from CSV file
data = pd.read_csv('FormattedData.csv')

# Convert time column to datetime
data['datetime'] = pd.to_datetime(data['datetime'], format='mixed', dayfirst=False)

# Extract input variables from the dataframe
x_data = torch.tensor(data['Lat'].values, dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(data['Long'].values, dtype=torch.float32).view(-1, 1)
bed_slope_data = torch.tensor(data['slope'].values, dtype=torch.float32).view(-1, 1)
precipitation_data = torch.tensor(data['precip'].values, dtype=torch.float32).view(-1, 1)
time_data = torch.tensor(data['datetime'].astype(int).values, dtype=torch.float32, requires_grad=True).view(-1, 1) / 1e9
width_data = torch.tensor(data['width'].values, dtype=torch.float32).view(-1, 1)
mannings_n_data = torch.tensor(data['n'].values, dtype=torch.float32).view(-1, 1)
gage_height_data = torch.tensor(data['gage_height'].values, dtype=torch.float32).view(-1, 1)
discharge_data = torch.tensor(data['discharge'].values, dtype=torch.float32).view(-1, 1)
temperature_data = torch.tensor(data['temp'].values, dtype=torch.float32).view(-1, 1)

# Prepare input data and target variables
input_data = torch.cat([x_data, y_data, bed_slope_data, precipitation_data,
                        time_data, width_data, mannings_n_data, temperature_data], dim=1).unsqueeze(1)
target_data = torch.cat([discharge_data, gage_height_data], dim=1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(input_data, discharge_data, gage_height_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the LSTM model
input_size = 8
hidden_size = 64
num_layers = 2
model = RiverPINN(input_size, hidden_size, num_layers)

# Train the model
num_epochs = 20
learning_rate = 0.001
train(model, dataloader, num_epochs, learning_rate)

# Generate hydrographs and stage graphs at specific locations
x_locs = [0.1, 0.5, 0.9]
y_locs = [0.5, 0.5, 0.5]
bed_slope = 0.001
precipitation = 0.01
t_range = np.linspace(0, 1, 100)
width = 10.0
mannings_n = 0.03
temperature = 20.0
hydrographs, stage_graphs = generate_graphs(model, x_locs, y_locs, bed_slope, precipitation, temperature, t_range, width, mannings_n)

# Plot the hydrographs and stage graphs
for i, (hydrograph, stage_graph) in enumerate(zip(hydrographs, stage_graphs)):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_range, hydrograph)
    plt.xlabel("Time")
    plt.ylabel("Discharge")
    plt.title(f"Hydrograph at Location ({x_locs[i]}, {y_locs[i]})")

    plt.subplot(1, 2, 2)
    plt.plot(t_range, stage_graph)
    plt.xlabel("Time")
    plt.ylabel("Gage Height")
    plt.title(f"Stage Graph at Location ({x_locs[i]}, {y_locs[i]})")

    plt.tight_layout()
    plt.show()
