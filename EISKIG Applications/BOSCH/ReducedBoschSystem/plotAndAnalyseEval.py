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

set = 2
# Read the csv files
if set==0: #take from folder compare_ppo_rb
    df_agent1 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb3\agent_8_0024-01_all-episodes.csv', sep=';')
    df_agent2 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb3\rbfb_0024-01_all-episodes.csv', sep=';')
    df_rb = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb3\rb_0024-01_all-episodes.csv', sep=';')
elif set==1: #take from folder compare_ppo_rb_2018
    df_agent1 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb_2018\agent_2_0019-01_all-episodes.csv', sep=';')
    df_agent2 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb_2018\agent_2_0019-01_all-episodes.csv', sep=';')
    df_rb = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_ppo_rb_2018\rb_0019-01_all-episodes.csv', sep=';')
elif set==2:
    df_agent1 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_FY\agent_8_0024-01_all-episodes.csv', sep=';')
    df_agent2 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_FY\rbfb_0024-01_all-episodes.csv', sep=';')
    df_rb = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\compare_FY\rb_0024-01_all-episodes.csv', sep=';')
print(df_agent1.columns)
print(df_rb.columns)
# exclude some scenarios
df_agent1.drop(exclude, inplace=True)
df_agent2.drop(exclude, inplace=True)
df_rb.drop(exclude, inplace=True)
# Convert the 'datetime_begin' column to datetime
df_agent1['datetime_begin'] = pd.to_datetime(df_agent1['datetime_begin'])
df_agent2['datetime_begin'] = pd.to_datetime(df_agent1['datetime_begin'])
df_rb['datetime_begin'] = pd.to_datetime(df_rb['datetime_begin'])

# Extract the month from the date
df_agent1['Month'] = df_agent1['datetime_begin'].dt.month
df_agent2['Month'] = df_agent2['datetime_begin'].dt.month
df_rb['Month'] = df_rb['datetime_begin'].dt.month

# List of columns for which you want to plot graphs
columns = ['energy_cost_cooling_eur','num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour','temperature_range_crossed_per_day' ]
# columns = ['energy_cost_cooling_eur', 'CER', "cost_CER", 'switches_per_hour', 'rewards_temperature_heat','reward_policy_overwrite', "T_cold_MSE_8", "T_cool_MSE_18",]
for column in columns:
    # Group by month and calculate the mean
    monthly_average1 = df_agent1.groupby('Month')[column].mean().reset_index()
    monthly_average2 = df_agent2.groupby('Month')[column].mean().reset_index()
    monthly_averagerb = df_rb.groupby('Month')[column].mean().reset_index()

    # Create a new DataFrame for plotting
    plot_df = pd.concat([monthly_average1.assign(Agent='Agent 1'),monthly_average2.assign(Agent='Agent 2'), monthly_averagerb.assign(Agent='Rule based control')])

    # Plot the bar graph
    fig = px.bar(plot_df, x='Month', y=column, color='Agent', barmode='group', title='Average ' + column.replace("_", " ") + ' per Month')
    # fig.show()

# complete year

columns1 = ['CER','num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour','temperature_range_crossed_per_day']
columns2 = ['energy_cost_cooling_eur']
columns3 = ['rewards_total', 'rewards_switching', 'rewards_temperature_heat', 'reward_abort', 'reward_policy_overwrite', 'reward_energy_electric']
columns4 = ['P_el_KKM_kwh', 'P_el_KTs_kwh', 'P_el_ges_kwh']
for columns in [columns1, columns2, columns3, columns4]:
    data = {
        'Category': columns,
        'Agent 1': [df_agent1[column].mean() for column in columns],
        'Agent 2': [df_agent2[column].mean() for column in columns],
        'Rule-based': [df_rb[column].mean() for column in columns]
    }
    df = pd.DataFrame(data)

    # Create a bar plot
    fig = px.bar(df, x='Category', y=['Agent 1', 'Agent 2', 'Rule-based'],
                title='Comparison: Agents vs. Rule-based Control',
                labels={'value': 'Values', 'variable': 'Method'},
                color_discrete_sequence=['blue', 'green', 'orange'],
                barmode='group')

    # Show the plot
    fig.show()