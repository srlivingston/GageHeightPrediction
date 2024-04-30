from gageheight2 import PINN
from gageheight2 import feature_columns, location_categories, mean_values, std_values, numeric_columns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

window_size = 10
date_parser = lambda x: datetime.strptime(x, '%m/%d/%y %H:%M')

def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padding_length = max_len - seq.shape[0]
        if padding_length > 0:
            padding = np.zeros((padding_length, seq.shape[1]))
            seq_padded = np.vstack([seq, padding])
        else:
            seq_padded = seq
        padded_sequences.append(seq_padded)
    return np.array(padded_sequences, dtype=np.float32)

def create_future_sequences(data, window_size, feature_columns, mean_values, std_values):
    sequences = []
    start_indices = []
    is_missing = data[['channel_depth', 'discharge']].isna().any(axis=1)
    for i in range(len(data)):
        if is_missing[i] and (i == 0 or not is_missing[i-1]):
            start_idx = i
            seq_start = max(0, start_idx - window_size)
            seq_end = start_idx
            seq = data.loc[seq_start:seq_end - 1, feature_columns].copy()
            for col in feature_columns:
                if col in mean_values and col in std_values:
                    seq[col] = seq[col].fillna(mean_values[col])
                    seq[col] = (seq[col] - mean_values[col]) / std_values[col]
            sequences.append(seq.values)
            start_indices.append(start_idx)
    sequences = pad_sequences(sequences)
    return np.array(sequences), start_indices


def evaluate_and_complete_missing(data_path, model, feature_columns, location_categories, mean_values, std_values, window_size):
    # Load the evaluation data
    eval_data = pd.read_csv(data_path, parse_dates=['datetime'], date_parser=date_parser)

    # Convert location to categorical using previously saved categories
    eval_data['locationid'] = pd.Categorical(eval_data['locationid'], categories=location_categories['locationid'])
    location_dummies = pd.get_dummies(eval_data['locationid'], prefix='locationid')
    eval_data = pd.concat([eval_data, location_dummies], axis=1)
    eval_data.drop('locationid', axis=1, inplace=True)

    eval_data['month_sin'] = np.sin(2 * np.pi * eval_data['datetime'].dt.month / 12)
    eval_data['month_cos'] = np.cos(2 * np.pi * eval_data['datetime'].dt.month / 12)
    eval_data['day_sin'] = np.sin(2 * np.pi * eval_data['datetime'].dt.day / 31)
    eval_data['day_cos'] = np.cos(2 * np.pi * eval_data['datetime'].dt.day / 31)
    eval_data['hour_sin'] = np.sin(2 * np.pi * eval_data['datetime'].dt.hour / 24)
    eval_data['hour_cos'] = np.cos(2 * np.pi * eval_data['datetime'].dt.hour / 24)
    eval_data['minute_sin'] = np.sin(2 * np.pi * eval_data['datetime'].dt.minute / 60)
    eval_data['minute_cos'] = np.cos(2 * np.pi * eval_data['datetime'].dt.minute / 60)

    # Apply same transformations as training
    eval_data['discharge'] *= 0.0283168
    eval_data['precipitation'] *= 25.4
    eval_data['precipitation'] = np.log1p(eval_data['precipitation'])

    # Normalize features using saved mean and std
    for col in numeric_columns:
        eval_data[col] = (eval_data[col] - mean_values[col]) / std_values[col]

    # Identify rows with missing target values
    missing_mask = eval_data[['channel_depth', 'discharge']].isna()
    missing_indices = missing_mask.any(axis=1)


    sequences, start_indices = create_future_sequences(eval_data, window_size, feature_columns, mean_values, std_values)
    sequences = np.array(sequences, dtype=np.float32)  # Ensure type consistency
    eval_dataset = TensorDataset(torch.tensor(sequences))  # This should no longer throw an error
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Predict missing values
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in eval_loader:
            output = model(features[0])
            predictions.append(output.detach().numpy())

    # Place predictions into DataFrame
    for idx, pred in zip(start_indices, predictions):
        for j, p in enumerate(pred):
            if (idx + j) < len(eval_data):
                eval_data.loc[idx + j, ['channel_depth', 'discharge']] = p


    # Save the completed dataframe
    eval_data.to_csv('Completed_Future_Predictions.csv', index=False)
    return 'Completed_Future_Predictions.csv'

model = PINN(input_dim=20, hidden_dim=64, output_dim=2, num_layers=3)
model.load_state_dict(torch.load('pinn_model.pth'))
completed_csv_path = evaluate_and_complete_missing('Validation_Data.csv', model, feature_columns, location_categories, mean_values, std_values, window_size)
print(f"Predictions completed and saved to {completed_csv_path}")
