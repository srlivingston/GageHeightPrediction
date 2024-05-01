import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

date_parser = lambda x: datetime.strptime(x, '%m/%d/%y %H:%M')

# Load data and parse dates
data = pd.read_csv('Training_Data.csv', parse_dates=['datetime'], date_parser=date_parser)

# Convert location to categorical and apply one-hot encoding
data['locationid'] = data['locationid'].astype('category')
location_dummies = pd.get_dummies(data['locationid'], prefix='locationid')
data = pd.concat([data, location_dummies], axis=1)

# Save the categories of locationid for use in prediction
location_categories = pd.DataFrame(data['locationid'].cat.categories, columns=['locationid'])
location_categories.to_csv('location_categories.csv', index=False)

data.drop('locationid', axis=1, inplace=True)  # Drop original location column after encoding

#convert to metric units
data['precipitation'] *= 25.4  # inches to mm
data['discharge'] *= 0.0283168  # cubic feet/sec to cubic meters/sec

# Feature engineering for datetime
data['month_sin'] = np.sin(2 * np.pi * data['datetime'].dt.month / 12)
data['month_cos'] = np.cos(2 * np.pi * data['datetime'].dt.month / 12)
data['day_sin'] = np.sin(2 * np.pi * data['datetime'].dt.day / 31)
data['day_cos'] = np.cos(2 * np.pi * data['datetime'].dt.day / 31)
data['hour_sin'] = np.sin(2 * np.pi * data['datetime'].dt.hour / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['datetime'].dt.hour / 24)
data['minute_sin'] = np.sin(2 * np.pi * data['datetime'].dt.minute / 60)
data['minute_cos'] = np.cos(2 * np.pi * data['datetime'].dt.minute / 60)

data['precipitation'] = np.log1p(data['precipitation'])


# Normalize features
numeric_columns = ['temperature', 'bed_slope', 'channel_width', 'mannings_n', 'channel_depth', 'discharge', 'channel_bed_elevation', 'gage_elevation', 'precipitation']

mean_values = data[numeric_columns].mean()
std_values = data[numeric_columns].std()

# Save to CSV
mean_values.to_csv('train_mean.csv', header=True)
std_values.to_csv('train_std.csv', header=True)


# Method for calculating haversine distance
def haversine(lat1, lon1, lat2, lon2):
    radius = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * radius

# Prepare dataset for PyTorch
class RiverDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_columns):

        self.features = torch.tensor(dataframe[feature_columns].astype(float).values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_columns].astype(float).values, dtype=torch.float32)
        self.lat = dataframe['lat'].values
        self.long = dataframe['long'].values
        self.datetime = pd.to_datetime(dataframe['datetime'])
        self.bed_slope = torch.tensor(dataframe['bed_slope'].values, dtype=torch.float32)
        self.channel_width = torch.tensor(dataframe['channel_width'].values, dtype=torch.float32)
        self.channel_depth = torch.tensor(dataframe['channel_depth'].values, dtype=torch.float32)
        self.discharge = torch.tensor(dataframe['discharge'].values, dtype=torch.float32)

        # Compute spatial and temporal derivatives
        self.spatial_derivatives = self.compute_spatial_derivatives()
        self.temporal_derivatives = self.compute_temporal_derivatives()

    def compute_spatial_derivatives(self):
        spatial_derivatives = []
        for i in range(len(self.lat) - 1):
            # Check if consecutive points belong to the same location
            if self.lat[i] == self.lat[i + 1] and self.long[i] == self.long[i + 1]:
                dist = haversine(self.lat[i], self.long[i], self.lat[i + 1], self.long[i + 1])
                if dist > 0:
                    depth_derivative = (self.channel_depth[i+1] - self.channel_depth[i]) / dist
                    discharge_derivative = (self.discharge[i+1] - self.discharge[i]) / dist
                    spatial_derivatives.append(torch.tensor([depth_derivative, discharge_derivative], dtype=torch.float32))
                else:
                    spatial_derivatives.append(torch.zeros(2, dtype=torch.float32))
            else:
                spatial_derivatives.append(torch.zeros(2, dtype=torch.float32))  # Reset at location change
        return torch.stack(spatial_derivatives) if spatial_derivatives else torch.empty(0, 2)

    def compute_temporal_derivatives(self):
        temporal_derivatives = []
        for i in range(len(self.datetime) - 1):
            if self.lat[i] == self.lat[i + 1] and self.long[i] == self.long[i + 1]:  # Check for same location
                delta_t = (self.datetime[i+1] - self.datetime[i]).total_seconds()
                if delta_t > 0:
                    depth_derivative = (self.channel_depth[i+1] - self.channel_depth[i]) / delta_t
                    discharge_derivative = (self.discharge[i+1] - self.discharge[i]) / delta_t
                    temporal_derivatives.append(torch.tensor([depth_derivative, discharge_derivative], dtype=torch.float32))
                else:
                    temporal_derivatives.append(torch.zeros(2, dtype=torch.float32))
            else:
                temporal_derivatives.append(torch.zeros(2, dtype=torch.float32))  # Reset at location change
        return torch.stack(temporal_derivatives) if temporal_derivatives else torch.empty(0, 2)

    def __len__(self):
        return len(self.features) - 1

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx], self.spatial_derivatives[idx], self.temporal_derivatives[idx], self.bed_slope[idx], self.channel_width[idx], self.channel_depth[idx])

feature_columns = [col for col in data.columns if col not in ['datetime', 'channel_depth', 'discharge']]
print("Number of feature columns:", len(feature_columns))
print("Feature columns:", feature_columns)
target_columns = ['channel_depth', 'discharge']
dataset = RiverDataset(data, feature_columns, target_columns)

# Splitting dataset into training and testing
num_items = len(dataset)
num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def compute_physics_loss(outputs, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth, g=9.81):
    dh_dt = temporal_derivatives[:, 0]
    dQ_dt = temporal_derivatives[:, 1]
    dh_dx = spatial_derivatives[:, 0]
    dQ_dx = spatial_derivatives[:, 1]

    A = channel_depth * channel_width  # Cross-sectional area
    u = outputs[:, 1] / A  # Flow velocity using discharge and area

    continuity_loss = torch.mean((dh_dt + dQ_dx / A) ** 2)
    momentum_loss = torch.mean((dQ_dt + u * dQ_dx + g * A * dh_dx) ** 2)

    return continuity_loss + momentum_loss


# Model definition
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PINN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.5, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        fc1_out = self.relu(self.fc1(last_time_step))
        output = self.fc2(fc1_out)
        return self.output_relu(output)

# Model, loss, and optimizer
input_dim_size = len(feature_columns)
model = PINN(input_dim=input_dim_size, hidden_dim=64, output_dim=2, num_layers=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 2000

#weights for physics and data


#weights for physics and data
data_weight_start = 0.9
physics_weight_start = 0.1

train_losses = []
data_losses = []
physics_losses = []

# Training loop with physics-informed loss
for epoch in range(num_epochs):
    train_loss = 0.0
    data_loss = 0.0
    epoch_physics_loss = 0.0

    data_weight = data_weight_start * (1 - epoch / num_epochs)
    physics_weight = physics_weight_start * (epoch / num_epochs)

    for features, targets, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth in train_loader:
        optimizer.zero_grad()
        outputs = model(features.unsqueeze(1))
        mse_loss = criterion(outputs, targets)
        physics_loss = compute_physics_loss(outputs, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth)
        total_loss = (data_weight * mse_loss) + (physics_weight * physics_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += total_loss.item()
        data_loss += mse_loss.item()
        epoch_physics_loss += physics_loss.item()
    print(f'Epoch {epoch+1} / {num_epochs}, Loss: {total_loss.item()}')
    train_losses.append(train_loss / len(train_loader))
    data_losses.append(data_loss / len(train_loader))
    physics_losses.append(epoch_physics_loss / len(train_loader))
#save the model
torch.save(model.state_dict(), 'pinn_model.pth')
