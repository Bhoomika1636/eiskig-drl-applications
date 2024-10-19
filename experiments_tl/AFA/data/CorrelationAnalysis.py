"""
This script performs a correlation analysis and visualizes the correlation matrix.

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Construct the full path by combining the working directory with the relative path to the CSV
file_path1 = os.path.join(os.getcwd(), 'experiments_hr/AFA/data/Versuchsreihe_27052024.csv')
file_path2 = os.path.join(os.getcwd(), 'experiments_hr/AFA/data/Sonnenstrahlung_MA_20240527_20240529.csv')

# Load the CSV file using the constructed file path
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# Ensure the 'Timestamp' columns are in datetime format
data1['Timestamp'] = pd.to_datetime(data1['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
data2['Timestamp'] = pd.to_datetime(data2['Timestamp'], format='%Y%m%d%H%M')

# Merge the two DataFrames on the 'Timestamp' column
merged_data = pd.merge(data1, data2, on='Timestamp', suffixes=('_1', '_2'))

# Define the columns for correlation analysis
columns_for_correlation1 = [
    'MAIN.Ambient.localState.fOutsideTemperature', 
    'HKK_KaRo-A.600.ThHy_int.PU600', 
    'HKK_KaRo-A.600.ThHy_ext.PU600', 
    'HKK_KaRo-A.600.ThHy_T.TS660', 
    'HKK_KaRo-A.600.ThHy_T.TS640', 
    'Sonst_WSta.Atmo_v_Luft', 
    'Sonst_WSta.Atmo_RgnS',
    'HKK_KaRo-A.600.ThHy_Q.PU600'
]

columns_for_correlation2 = [
    'GS_10'
]

total_columns = columns_for_correlation1 + columns_for_correlation2

# Check if all columns exist in the DataFrame
for column in total_columns:
    if column not in merged_data.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")

# Compute the correlation matrix
correlation_matrix = merged_data[total_columns].corr()

# Plot the correlation matrix using matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')

# Add color bar
fig.colorbar(cax)

# Set axis labels
ax.set_xticks(np.arange(len(total_columns)))
ax.set_yticks(np.arange(len(total_columns)))
ax.set_xticklabels(total_columns, rotation=45, ha='left')
ax.set_yticklabels(total_columns)

# Annotate the matrix with the correlation values
for i in range(len(total_columns)):
    for j in range(len(total_columns)):
        ax.text(i, j, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()
