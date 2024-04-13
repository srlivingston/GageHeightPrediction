import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Load data and parse dates
data = pd.read_csv('FormattedData.csv', parse_dates=['datetime'])

# Convert location to categorical and apply one-hot encoding
data['location'] = data['location'].astype('category')
location_dummies = pd.get_dummies(data['location'], prefix='location')
data = pd.concat([data, location_dummies], axis=1)
data.drop('location', axis=1, inplace=True)  # Drop original location column after encoding

# Feature engineering for datetime
data['hour_sin'] = np.sin(2 * np.pi * data['datetime'].dt.hour/24)
data['hour_cos'] = np.cos(2 * np.pi * data['datetime'].dt.hour/24)
data['minute_sin'] = np.sin(2 * np.pi * data['datetime'].dt.minute/60)
data['minute_cos'] = np.cos(2 * np.pi * data['datetime'].dt.minute/60)

# Normalize features per station using PyTorch
numeric_columns = ['precipitation', 'temperature', 'bed_slope', 'channel_width']
for column in numeric_columns:
    tensor_subset = torch.tensor(data[column].values, dtype=torch.float32)
    mean = torch.mean(tensor_subset)
    std = torch.std(tensor_subset)
    normalized_tensor = (tensor_subset - mean) / std
    data[column] = normalized_tensor.numpy()

# Converting data to PyTorch tensors
class RiverDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_columns):
        self.features = torch.tensor(dataframe[feature_columns].astype(float).values, dtype=torch.float32)
        self.targets = torch.tensor(dataframe[target_columns].astype(float).values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

feature_columns = [col for col in data.columns if col not in ['datetime', 'gage_height', 'discharge']]
target_columns = ['gage_height', 'discharge']
dataset = RiverDataset(data, feature_columns, target_columns)

# Splitting dataset into training and testing using PyTorch
num_items = len(dataset)
num_train = round(num_items * 0.8)
num_test = num_items - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PINN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Get output from LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape is [batch_size, sequence_length, hidden_dim]
        # We want the last time step output
        last_time_step = lstm_out[:, -1, :]
        output = self.fc(last_time_step)
        return output

input_dim = len(feature_columns)
hidden_dim = 64
output_dim = 2
num_layers = 2
model = PINN(input_dim, hidden_dim, output_dim, num_layers)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    for features, targets in train_loader:
        # Ensure features have an extra dimension for sequence length
        features = features.unsqueeze(1)  # Add a sequence length of 1
        optimizer.zero_grad()
        outputs = model(features)  # Ensure features are [batch_size, sequence_length, num_features]
        loss = criterion(outputs, targets)  # outputs and targets should now match in size
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")
