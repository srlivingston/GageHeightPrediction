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
def loss_function(discharge, gage_height, discharge_obs, gage_height_obs, bed_slope, precipitation, temperature, time, width, mannings_n):
    # Data loss: Mean squared error between predicted and observed discharge and gage height
    data_loss = torch.mean((discharge - discharge_obs)**2 + (gage_height - gage_height_obs)**2)

    # Physics loss - Continuity equation
    # Compute the gradients of gage height with respect to time
    gage_height_t = torch.autograd.grad(gage_height, time, grad_outputs=torch.ones_like(gage_height),
                                        create_graph=True, allow_unused=True)[0]
    # Compute the gradients of discharge with respect to time
    discharge_t = torch.autograd.grad(discharge, time, grad_outputs=torch.ones_like(discharge),
                                      create_graph=True, allow_unused=True)[0]

    # Handle None gradients by replacing them with zero tensors
    gage_height_t = torch.zeros_like(gage_height) if gage_height_t is None else gage_height_t
    discharge_t = torch.zeros_like(discharge) if discharge_t is None else discharge_t

    # Compute the residual of the continuity equation
    continuity_residual = gage_height_t + discharge_t / width
    # Compute the mean squared error of the continuity residual
    continuity_loss = torch.mean(continuity_residual**2)

    # Physics loss - Momentum equation
    # Compute the velocity using discharge and gage height
    velocity = discharge / (gage_height * width)
    # Compute the gradients of velocity with respect to time
    velocity_t = torch.autograd.grad(velocity, time, grad_outputs=torch.ones_like(velocity),
                                     create_graph=True, allow_unused=True)[0]

    # Handle None gradients by replacing them with zero tensors
    velocity_t = torch.zeros_like(velocity) if velocity_t is None else velocity_t

    # Constants
    g = 9.81  # Gravitational acceleration
    rho = 1000  # Water density

    # Bed slope and friction terms
    bed_slope_term = g * bed_slope
    friction_term = g * mannings_n**2 * velocity * torch.abs(velocity) / (gage_height**(4/3) + 1e-8)  # Adding a small epsilon to prevent division by zero

    # Compute the residual of the momentum equation
    momentum_residual = velocity_t + g * gage_height_t + bed_slope_term - friction_term
    # Compute the mean squared error of the momentum residual
    momentum_loss = torch.mean(momentum_residual**2)

    # Combine the continuity and momentum losses to get the physics loss
    physics_loss = continuity_loss + momentum_loss

    # Return the combined loss (data_loss + physics_loss)
    return data_loss + physics_loss

# Training loop
def train(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_data, discharge_obs, gage_height_obs = batch

            optimizer.zero_grad()

            discharge_pred, gage_height_pred = model(input_data)

            loss = loss_function(discharge_pred, gage_height_pred, discharge_obs, gage_height_obs,
                                 input_data[:, :, 0], input_data[:, :, 1], input_data[:, :, 2],
                                 input_data[:, :, 3], input_data[:, :, 4], input_data[:, :, 5])

            loss.backward(retain_graph=True)

            # Clip gradients to mitigate "nan" loss
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()

            # Compute data loss
            #data_loss = torch.mean((discharge_pred - discharge_obs)**2 + (gage_height_pred - gage_height_obs)**2)

            # Compute continuity and momentum losses directly from the loss function
            #continuity_loss, momentum_loss = loss - data_loss, data_loss

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        scheduler.step(loss)

# Generate hydrographs and stage graphs
def generate_graphs(model, input_data):
    discharge_pred_list = []
    stage_pred_list = []
    location_list = []

    # Min-max scaling for input data
    bed_slope_min, bed_slope_max = input_data['slope'].min(), input_data['slope'].max()
    precipitation_min, precipitation_max = input_data['precip'].min(), input_data['precip'].max()
    time_min, time_max = input_data['datetime'].min().timestamp(), input_data['datetime'].max().timestamp()
    width_min, width_max = input_data['width'].min(), input_data['width'].max()
    mannings_n_min, mannings_n_max = input_data['n'].min(), input_data['n'].max()
    temperature_min, temperature_max = input_data['temp'].min(), input_data['temp'].max()

    for _, row in input_data.iterrows():
        bed_slope = row['slope']
        precipitation = row['precip']
        time = row['datetime'].timestamp()
        width = row['width']
        mannings_n = row['n']
        temperature = row['temp']

        # Create input tensors for the given location and time
        bed_slope_tensor = torch.tensor([bed_slope], dtype=torch.float32).view(-1, 1)
        precipitation_tensor = torch.tensor([precipitation], dtype=torch.float32).view(-1, 1)
        time_tensor = torch.tensor([time], dtype=torch.float32).view(-1, 1)
        width_tensor = torch.tensor([width], dtype=torch.float32).view(-1, 1)
        temperature_tensor = torch.tensor([temperature], dtype=torch.float32).view(-1, 1)
        mannings_n_tensor = torch.tensor([mannings_n], dtype=torch.float32).view(-1, 1)

        # Normalize the input data using min-max scaling
        bed_slope_norm = (bed_slope_tensor - bed_slope_min) / (bed_slope_max - bed_slope_min)
        precipitation_norm = (precipitation_tensor - precipitation_min) / (precipitation_max - precipitation_min)
        time_norm = (time_tensor - time_min) / (time_max - time_min)
        width_norm = (width_tensor - width_min) / (width_max - width_min)
        mannings_n_norm = (mannings_n_tensor - mannings_n_min) / (mannings_n_max - mannings_n_min)
        temperature_norm = (temperature_tensor - temperature_min) / (temperature_max - temperature_min)

        # Concatenate the normalized input tensors and add an extra dimension for the LSTM input
        input_tensor = torch.cat([bed_slope_norm, precipitation_norm, temperature_norm,
                                  time_norm, width_norm, mannings_n_norm], dim=1).unsqueeze(0)

        # Predict discharge and stage using the trained model
        with torch.no_grad():
            discharge_pred, stage_pred = model(input_tensor)

        # Append the predicted discharge, stage, and location to the lists
        discharge_pred_list.append(discharge_pred.item())
        stage_pred_list.append(stage_pred.item())

    # Convert the predicted discharge and stage values to NumPy arrays
    discharge_pred_array = np.array(discharge_pred_list).reshape(-1, 1)
    stage_pred_array = np.array(stage_pred_list).reshape(-1, 1)

    return discharge_pred_array, stage_pred_array, location_list

# Load data from CSV file
data = pd.read_csv('FormattedData.csv')

# Convert time column to datetime
data['datetime'] = pd.to_datetime(data['datetime'], format='mixed', dayfirst=False)

# Extract input variables from the dataframe
bed_slope_data = torch.tensor(data['bed_slope'].values, dtype=torch.float32).view(-1, 1)
precipitation_data = torch.tensor(data['precipitation'].values, dtype=torch.float32).view(-1, 1)
time_data = torch.tensor(data['datetime'].astype(int).values, dtype=torch.float32, requires_grad=True).view(-1, 1) / 1e9
width_data = torch.tensor(data['channel_width'].values, dtype=torch.float32).view(-1, 1)
mannings_n_data = torch.tensor(data['mannings_n'].values, dtype=torch.float32).view(-1, 1)
gage_height_data = torch.tensor(data['gage_height'].values, dtype=torch.float32).view(-1, 1)
discharge_data = torch.tensor(data['discharge'].values, dtype=torch.float32).view(-1, 1)
temperature_data = torch.tensor(data['temperature'].values, dtype=torch.float32).view(-1, 1)

# Min-max scaling for input data
bed_slope_min, bed_slope_max = bed_slope_data.min(), bed_slope_data.max()
precipitation_min, precipitation_max = precipitation_data.min(), precipitation_data.max()
time_min, time_max = time_data.min(), time_data.max()
width_min, width_max = width_data.min(), width_data.max()
mannings_n_min, mannings_n_max = mannings_n_data.min(), mannings_n_data.max()
temperature_min, temperature_max = temperature_data.min(), temperature_data.max()

# Normalize the input data using min-max scaling
bed_slope_data_norm = (bed_slope_data - bed_slope_min) / (bed_slope_max - bed_slope_min)
precipitation_data_norm = (precipitation_data - precipitation_min) / (precipitation_max - precipitation_min)
time_data_norm = (time_data - time_min) / (time_max - time_min)
width_data_norm = (width_data - width_min) / (width_max - width_min)
mannings_n_data_norm = (mannings_n_data - mannings_n_min) / (mannings_n_max - mannings_n_min)
temperature_data_norm = (temperature_data - temperature_min) / (temperature_max - temperature_min)

# Prepare input data and target variables
input_data = torch.cat([bed_slope_data_norm, precipitation_data_norm,
                        time_data_norm, width_data_norm, mannings_n_data_norm, temperature_data_norm], dim=1).unsqueeze(1)
target_data = torch.cat([discharge_data, gage_height_data], dim=1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(input_data, discharge_data, gage_height_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the LSTM model
input_size = 6
hidden_size = 64
num_layers = 2
model = RiverPINN(input_size, hidden_size, num_layers)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model.apply(init_weights)

# Train the model
num_epochs = 20
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
time_data = torch.tensor(input_data['datetime'].astype(int).values, dtype=torch.float32, requires_grad=True).view(-1, 1) / 1e9
width_data = torch.tensor(input_data['width'].values, dtype=torch.float32).view(-1, 1)
mannings_n_data = torch.tensor(input_data['n'].values, dtype=torch.float32).view(-1, 1)
temperature_data = torch.tensor(input_data['temp'].values, dtype=torch.float32).view(-1, 1)

discharge_pred, stage_pred, locations = generate_graphs(model, input_data)

# Create a DataFrame with the predicted discharge and stage values
output_data = pd.DataFrame({
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
