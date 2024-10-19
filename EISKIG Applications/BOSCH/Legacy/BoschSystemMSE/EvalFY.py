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
folder_path = r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschSystemMSE\results\compare_FY'

# Filenames list
AgentNames = [
    "rb",
    "agent_2",
    "agent_4",
    "agent_5"
    # Add more filenames as necessary
]

# Initialize an empty list to store dataframes
dataframes = []

# Process each file
for agent in AgentNames:
    filename = agent + "_0000-01_all-episodes.csv"
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
print(df_combined.columns)
# full year analysis 
    
# Define your categories of metrics
categories = {
    'columns1': ['CER', 'num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour', 'temperature_range_crossed_per_day'],
    'columns2': ['energy_cost_cooling_eur'],
    'columns3': ['rewards_total', 'rewards_switching', 'rewards_temperature_heat', 'reward_abort', 'reward_policy_overwrite', 'reward_energy_electric'],
    'columns4': ['P_el_KKM_kwh', 'P_el_KTs_kwh', 'P_el_ges_kwh'],
    'columns5': ['T_cold_MSE_8','T_cool1_MSE_14', 'T_cool2_MSE_17'],
    'columns5': ["Cooling2_low_boundary_crossed_per_day", "Cooling2_high_boundary_crossed_per_day","Cooling1_low_boundary_crossed_per_day", "Cooling1_high_boundary_crossed_per_day", 'Cold_water_low_boundary_crossed', 'Cold_water_high_boundary_crossed']
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

df_list = []
# Process each file
for agent in AgentNames:
    filename = agent + "_000-01_episode.csv"
    df = pd.read_csv(f"{folder_path}\\{filename}",index_col=None, header=0,sep=";").fillna(0)
    df_list.append(df.fillna(0))
    combined_df = df[['energy_cost_cooling_eur','num_policy_overwrite', 'switches_per_hour','dateTime']]
    combined_df['dateTime']=pd.to_datetime(combined_df['dateTime'])
    combined_df.set_index('dateTime', drop=True,inplace=True)
    for i in combined_df.columns:
        try:
            combined_df[i] = combined_df[i].astype(float)
        except:
            combined_df.drop(columns=i, inplace=True)
    combined_df=combined_df.resample('D').sum().dropna(axis=0)

    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns, title=agent)# print(f"MSE cooling_ag1: {eval_df['SE_T_cooling_ag1'].mean()}")
    #fig.show()