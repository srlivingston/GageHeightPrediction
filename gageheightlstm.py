import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.dates as mdates

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_data, discharge_obs, gage_height_obs = batch

            optimizer.zero_grad()

            discharge_pred, gage_height_pred = model(input_data)

            loss = loss_function(discharge_pred, gage_height_pred, discharge_obs, gage_height_obs,
                                 input_data[:, :, 0], input_data[:, :, 1], input_data[:, :, 2],
                                 input_data[:, :, 3], input_data[:, :, 4], input_data[:, :, 5],
                                 input_data[:, :, 6], input_data[:, :, 7])

            loss.backward(retain_graph=True)

            # Clip gradients to mitigate "nan" loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Generate hydrographs and stage graphs
def generate_graphs(model, input_data):
    discharge_pred_list = []
    stage_pred_list = []
    location_list = []

    for _, row in input_data.iterrows():
        x = row['Lat']
        y = row['Long']
        bed_slope = row['slope']
        precipitation = row['precip']
        time = row['datetime'].timestamp()
        width = row['width']
        mannings_n = row['n']
        temperature = row['temp']

        # Create input tensors for the given location and time
        x_tensor = torch.tensor([x], dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor([y], dtype=torch.float32).view(-1, 1)
        bed_slope_tensor = torch.tensor([bed_slope], dtype=torch.float32).view(-1, 1)
        precipitation_tensor = torch.tensor([precipitation], dtype=torch.float32).view(-1, 1)
        time_tensor = torch.tensor([time], dtype=torch.float32).view(-1, 1)
        width_tensor = torch.tensor([width], dtype=torch.float32).view(-1, 1)
        temperature_tensor = torch.tensor([temperature], dtype=torch.float32).view(-1, 1)
        mannings_n_tensor = torch.tensor([mannings_n], dtype=torch.float32).view(-1, 1)

        # Concatenate the input tensors and add an extra dimension for the LSTM input
        input_tensor = torch.cat([x_tensor, y_tensor, bed_slope_tensor, precipitation_tensor, temperature_tensor,
                                  time_tensor, width_tensor, mannings_n_tensor], dim=1).unsqueeze(0)

        # Predict discharge and stage using the trained model
        with torch.no_grad():
            discharge_pred, stage_pred = model(input_tensor)

        # Append the predicted discharge, stage, and location to the lists
        discharge_pred_list.append(discharge_pred.item())
        stage_pred_list.append(stage_pred.item())
        location_list.append((x, y))

    # Convert the predicted discharge and stage values to NumPy arrays
    discharge_pred_array = np.array(discharge_pred_list).reshape(-1, 1)
    stage_pred_array = np.array(stage_pred_list).reshape(-1, 1)

    return discharge_pred_array, stage_pred_array, location_list


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
num_epochs = 1000
learning_rate = 0.0001
train(model, dataloader, num_epochs, learning_rate)


#Validation Data
input_data = pd.read_csv('ValidationData.csv')

input_data['datetime'] = pd.to_datetime(data['datetime'], format='mixed', dayfirst=False)

# Extract input variables from the dataframe
x_data = torch.tensor(input_data['Lat'].values, dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(input_data['Long'].values, dtype=torch.float32).view(-1, 1)
bed_slope_data = torch.tensor(input_data['slope'].values, dtype=torch.float32).view(-1, 1)
precipitation_data = torch.tensor(input_data['precip'].values, dtype=torch.float32).view(-1, 1)
time_data = torch.tensor(data['datetime'].astype(int).values, dtype=torch.float32, requires_grad=True).view(-1, 1) / 1e9
width_data = torch.tensor(input_data['width'].values, dtype=torch.float32).view(-1, 1)
mannings_n_data = torch.tensor(input_data['n'].values, dtype=torch.float32).view(-1, 1)
temperature_data = torch.tensor(input_data['temp'].values, dtype=torch.float32).view(-1, 1)

discharge_pred, stage_pred, locations = generate_graphs(model, input_data)

# Create a DataFrame with the predicted discharge and stage values
output_data = pd.DataFrame({
    'Lat': [loc[0] for loc in locations],
    'Long': [loc[1] for loc in locations],
    'Discharge': discharge_pred.flatten(),
    'Stage': stage_pred.flatten(),
})

# Save the output data to a CSV file
output_data.to_csv('output_data.csv', index=False)

discharge_pred_all = np.concatenate(discharge_pred)
stage_pred_all = np.concatenate(stage_pred)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(mdates.date2num(input_data['datetime']), discharge_pred_all)
plt.xlabel("Time")
plt.ylabel("Discharge")
plt.title("Hydrograph")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
plt.gcf().autofmt_xdate()  # Rotate and align the x-axis labels

plt.subplot(1, 2, 2)
plt.plot(mdates.date2num(input_data['datetime']), stage_pred_all)
plt.xlabel("Time")
plt.ylabel("Stage")
plt.title("Stage Graph")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
plt.gcf().autofmt_xdate()  # Rotate and align the x-axis labels

plt.tight_layout()
plt.show()
