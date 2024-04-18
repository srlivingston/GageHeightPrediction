import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split

date_parser = lambda x: datetime.strptime(x, '%m/%d/%y %H:%M')

# Load data and parse dates
data = pd.read_csv('TrainingData.csv', parse_dates=['datetime'], date_parser=date_parser)

# Convert location to categorical and apply one-hot encoding
data['locationid'] = data['locationid'].astype('category')
location_dummies = pd.get_dummies(data['locationid'], prefix='locationid')
data = pd.concat([data, location_dummies], axis=1)
data.drop('locationid', axis=1, inplace=True)  # Drop original location column after encoding

#convert to metric units
data['precipitation'] *= 25.4  # inches to mm
data['gage_height'] *= 0.3048  # feet to meters
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

# Conversion of gage height into channel depth
data['channel_depth'] = data['gage_height'] + (data['gage_elevation'] - data['channel_bed_elevation'])

# Normalize features
numeric_columns = ['precipitation', 'temperature', 'bed_slope', 'channel_width', 'mannings_n']
for column in numeric_columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()

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
        # Calculate spatial derivatives for channel depth and discharge
        spatial_derivatives = []
        for i in range(len(self.lat) - 1):
            dist = haversine(self.lat[i], self.long[i], self.lat[i + 1], self.long[i + 1])
            if dist > 0:
                depth_derivative = (self.channel_depth[i+1] - self.channel_depth[i]) / dist
                discharge_derivative = (self.discharge[i+1] - self.discharge[i]) / dist
                spatial_derivatives.append(torch.tensor([depth_derivative, discharge_derivative], dtype=torch.float32))
            else:
                spatial_derivatives.append(torch.zeros(2, dtype=torch.float32))
        return torch.stack(spatial_derivatives) if spatial_derivatives else torch.empty(0, 2)

    def compute_temporal_derivatives(self):
        # Calculate temporal derivatives for channel depth and discharge
        temporal_derivatives = []
        for i in range(len(self.datetime) - 1):
            delta_t = (self.datetime[i+1] - self.datetime[i]).total_seconds()
            if delta_t > 0:
                depth_derivative = (self.channel_depth[i+1] - self.channel_depth[i]) / delta_t
                discharge_derivative = (self.discharge[i+1] - self.discharge[i]) / delta_t
                temporal_derivatives.append(torch.tensor([depth_derivative, discharge_derivative], dtype=torch.float32))
            else:
                temporal_derivatives.append(torch.zeros(2, dtype=torch.float32))
        return torch.stack(temporal_derivatives) if temporal_derivatives else torch.empty(0, 2)

    def __len__(self):
        return len(self.features) - 1

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx], self.spatial_derivatives[idx], self.temporal_derivatives[idx], self.bed_slope[idx], self.channel_width[idx], self.channel_depth[idx])

feature_columns = [col for col in data.columns if col not in ['datetime', 'gage_height', 'discharge', 'bed_slope', 'channel_width']]
print("Number of feature columns:", len(feature_columns))
print("Feature columns:", feature_columns)
target_columns = ['gage_height', 'discharge']
dataset = RiverDataset(data, feature_columns, target_columns)

# Splitting dataset into training and testing
num_items = len(dataset)
num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def compute_physics_loss(outputs, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth, g=9.81):
    dh_dt = temporal_derivatives[:, 0]
    dQ_dt = temporal_derivatives[:, 1]
    dh_dx = spatial_derivatives[:, 0]
    dQ_dx = spatial_derivatives[:, 1]

    A = channel_depth * channel_width  # Cross-sectional area
    u = outputs[:, 1] / A  # Flow velocity using discharge and area

    continuity_loss = torch.mean((dh_dt + dQ_dx / A) ** 2)
    momentum_loss = torch.mean((dQ_dt + u * dQ_dx + g * dh_dx * A) ** 2)

    return continuity_loss + momentum_loss


# Model definition
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PINN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)

# Assume dataset and dataloader setup is already defined as previous messages
# Model, loss, and optimizer
input_dim_size = len(feature_columns)
model = PINN(input_dim=input_dim_size, hidden_dim=64, output_dim=2, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

#weights for physics and data
data_weight = 0.4
physics_weight = 0.6

# Training loop with physics-informed loss
for epoch in range(num_epochs):
    for features, targets, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth in train_loader:
        optimizer.zero_grad()
        outputs = model(features.unsqueeze(1))
        mse_loss = criterion(outputs, targets)
        physics_loss = compute_physics_loss(outputs, spatial_derivatives, temporal_derivatives, bed_slope, channel_width, channel_depth)
        total_loss = (data_weight * mse_loss) + (physics_weight * physics_loss)
        total_loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {total_loss.item()}')

#save the model
torch.save(model.state_dict(), 'pinn_model.pth')


# Evaluate the model
new_data = pd.read_csv('validation_data.csv', parse_dates=['datetime'])

# Apply the same preprocessing steps as for the training data
new_data['locationid'] = new_data['locationid'].astype('category')
location_dummies = pd.get_dummies(new_data['locationid'], prefix='locationid')
new_data = pd.concat([new_data, location_dummies], axis=1)
new_data.drop('locationid', axis=1, inplace=True)

#convert to metric units
new_data['precipitation'] *= 25.4  # inches to mm
new_data['channel_width'] *= 0.3048  # feet to meters

# Feature engineering for datetime and normalization
new_data['month_sin'] = np.sin(2 * np.pi * new_data['datetime'].dt.month / 12)
new_data['month_cos'] = np.cos(2 * np.pi * new_data['datetime'].dt.month / 12)
new_data['day_sin'] = np.sin(2 * np.pi * new_data['datetime'].dt.day / 31)
new_data['day_cos'] = np.cos(2 * np.pi * new_data['datetime'].dt.day / 31)
new_data['hour_sin'] = np.sin(2 * np.pi * new_data['datetime'].dt.hour / 24)
new_data['hour_cos'] = np.cos(2 * np.pi * new_data['datetime'].dt.hour / 24)
new_data['minute_sin'] = np.sin(2 * np.pi * new_data['datetime'].dt.minute / 60)
new_data['minute_cos'] = np.cos(2 * np.pi * new_data['datetime'].dt.minute / 60)

for column in numeric_columns:
    new_data[column] = (new_data[column] - new_data[column].mean()) / new_data[column].std()

# Preparing new_data for prediction
feature_columns_pred = [col for col in new_data.columns if col not in ['datetime']]
new_features = torch.tensor(new_data[feature_columns_pred].astype(float).values, dtype=torch.float32)

def predict(model, new_features):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(new_features.unsqueeze(1))
    return predictions

# Load the model (make sure the model path is correct)
model = PINN(input_dim=input_dim_size, hidden_dim=64, output_dim=2, num_layers=2)
model.load_state_dict(torch.load('pinn_model.pth'))

# Make predictions
predictions = predict(model, new_features)


# Convert predictions to DataFrame and prepare for output
predicted_data = pd.DataFrame(predictions.numpy(), columns=['Predicted Gage Height', 'Predicted Discharge'])
result_data = pd.concat([new_data.reset_index(drop=True), predicted_data], axis=1)

# Save the result to a new CSV file
result_data.to_csv('PredictedOutput.csv', index=False)
