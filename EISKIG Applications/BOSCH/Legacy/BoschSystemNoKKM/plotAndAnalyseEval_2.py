from __future__ import annotations

import pathlib

import sys
import numpy as np
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

exclude = []# [13,16]

# Folder path variable
folder_path = r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschSystemNoKKM\results\compare_ppo_noKKm'

# Filenames list
filenames = [
    "rb_0024-01_all-episodes.csv",
    "rb_2018_0024-01_all-episodes.csv",
    "agent_2_0024-01_all-episodes.csv",
    "agent_2_2018_0024-01_all-episodes.csv",
    # Add more filenames as necessary
]

# Initialize an empty list to store dataframes
dataframes = []

# Process each file
for filename in filenames:
    df = pd.read_csv(f"{folder_path}\\{filename}", sep=';')
    df.drop(exclude, inplace=True)  # Exclude some scenarios
    df['datetime_begin'] = pd.to_datetime(df['datetime_begin'])  # Convert 'datetime_begin' column to datetime
    df['Month'] = df['datetime_begin'].dt.month  # Extract the month from the date

    # Extract agent name from filename
    agent_name = filename.split("_0")[0]
    df['Agent'] = agent_name  # Assign agent name based on the filename

    dataframes.append(df)

# Combine all dataframes for analysis
df_combined = pd.concat(dataframes)

columns = ['energy_cost_cooling_eur', 'num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour', 'temperature_range_crossed_per_day']

# Plotting for each column
for column in columns:
    plot_df = df_combined.groupby(['Month', 'Agent'])[column].mean().reset_index()
    fig = px.bar(plot_df, x='Month', y=column, color='Agent', barmode='group', title=f'Average {column.replace("_", " ")} per Month')
    #fig.show()

# full year analysis 
    
# Define your categories of metrics
categories = {
    'columns1': ['CER', 'num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour', 'temperature_range_crossed_per_day'],
    'columns2': ['energy_cost_cooling_eur'],
    'columns3': ['rewards_total', 'rewards_switching', 'rewards_temperature_heat', 'reward_abort', 'reward_policy_overwrite', 'reward_energy_electric'],
    'columns4': ['P_el_KKM_kwh', 'P_el_KTs_kwh', 'P_el_ges_kwh'],
    'columns5': ["Cooling2_low_boundary_crossed_per_day", "Cooling2_high_boundary_crossed_per_day","Cooling1_low_boundary_crossed_per_day", "Cooling1_high_boundary_crossed_per_day"]
}

# Calculate and plot the yearly averages for each category
for category_name, columns in categories.items():
    data = []
    for column in columns:
        for agent in df_combined['Agent'].unique():
            yearly_average = df_combined[df_combined['Agent'] == agent][column].mean()
            data.append({'Category': column, 'Agent': agent, 'Yearly Average': yearly_average})
    df_summary = pd.DataFrame(data)
    
    # Plot
    fig = px.bar(df_summary, x='Category', y='Yearly Average', color='Agent', barmode='group', 
                 title=f'Yearly Average per Category: {category_name}',
                 labels={'Category': 'Metric', 'Yearly Average': 'Yearly Average', 'Agent': 'Agent'})
    fig.show()
