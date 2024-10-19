import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Construct the full path by combining the working directory with the relative path to the CSV
file_path1 = os.path.join(os.getcwd(), 'experiments_hr/AFA/data/Versuchsreihe_27052024.csv')
file_path2 = os.path.join(os.getcwd(), 'experiments_hr/AFA/data/Sonnenstrahlung_MA_20240527_20240529.csv')

# Load the CSV file using the constructed file path
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# Convert 'Timestamp' in both datasets to datetime format for easier plotting
data1['Timestamp'] = pd.to_datetime(data1['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
data2['Timestamp'] = pd.to_datetime(data2['Timestamp'], format='%Y%m%d%H%M')

specific_start_time = '2024-05-27 10:14:28'
specific_start_time = pd.to_datetime(specific_start_time)
specific_stop_time = '2024-05-29 16:14:27'
specific_stop_time = pd.to_datetime(specific_stop_time)

# Define the columns to plot
columns_to_plot1 = [
    ('MAIN.Ambient.localState.fOutsideTemperature','HKK_KaRo-A.600.ThHy_int.PU600','HKK_KaRo-A.600.ThHy_ext.PU600'),
    ('HKK_KaRo-A.600.ThHy_T.TS660', 'HKK_KaRo-A.600.ThHy_ext.PU600'),
    ('Sonst_WSta.Atmo_v_Luft','Sonst_WSta.Atmo_RgnS'),
    ('HKK_KaRo-A.600.ThHy_Q.PU600',)
]

columns_to_plot2 = [
    ('GS_10',)
]

# Plot setup
num_plots = len(columns_to_plot1) + len(columns_to_plot2)
fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(14, 6 * num_plots))

if num_plots > 1:
    axes = axes.flatten()
else:
    axes = [axes]

# Plotting data from the first CSV file
for i, column_group in enumerate(columns_to_plot1):
    for column in column_group:
        if column not in data1.columns:
            raise ValueError(f"Column {column} not found in DataFrame 1.")
        axes[i].plot(data1['Timestamp'], data1[column], label=column)
    axes[i].set_xlabel('Timestamp')
    axes[i].set_ylabel('Values')
    axes[i].legend()
    axes[i].set_xlim(specific_start_time, specific_stop_time)

# Plotting data from the second CSV file
for j, column_group in enumerate(columns_to_plot2, start=len(columns_to_plot1)):
    for column in column_group:
        if column not in data2.columns:
            raise ValueError(f"Column {column} not found in DataFrame 2.")
        axes[j].plot(data2['Timestamp'], data2[column], label=column)
    axes[j].set_xlabel('Timestamp')
    axes[j].set_ylabel('Values')
    axes[j].legend()
    axes[j].set_xlim(specific_start_time, specific_stop_time)


for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    #ax.tick_params(axis='x', rotation=45)
    
    

plt.tight_layout()
plt.show()
#plt.xticks([0o5-27-12, 0o5-27-18, 0o5-28-00, 0o5-28-0o6, 0o5-28-12, 0o5-28-18, 0o5-29-00, 0o5-29-0o6, 0o5-29-12, 0o5-29-18])
