from __future__ import annotations

import pathlib

import sys
import numpy as np
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

ValidationDates = True
# Read the csv files
if ValidationDates: #take from folder compare_ppo_rb
    df_agent1 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb\agent_5_2_0019-01_all-episodes.csv', sep=';')
    df_agent2 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb\agent_5_2_0019-01_all-episodes.csv', sep=';')
    df_rb = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb\rb_0019-01_all-episodes.csv', sep=';')
else: #take from folder compare_ppo_rb_2018
    df_agent1 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb_2018\agent_1_0019-01_all-episodes.csv', sep=';')
    df_agent2 = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb_2018\agent_2_0019-01_all-episodes.csv', sep=';')
    df_rb = pd.read_csv(r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\BoschCoolingSystem2\results\compare_ppo_rb_2018\rb_0019-01_all-episodes.csv', sep=';')

print(df_agent1.columns)
# Convert the 'datetime_begin' column to datetime
df_agent1['datetime_begin'] = pd.to_datetime(df_agent1['datetime_begin'])
df_agent2['datetime_begin'] = pd.to_datetime(df_agent1['datetime_begin'])
df_rb['datetime_begin'] = pd.to_datetime(df_rb['datetime_begin'])

# Extract the month from the date
df_agent1['Month'] = df_agent1['datetime_begin'].dt.month
df_agent2['Month'] = df_agent2['datetime_begin'].dt.month
df_rb['Month'] = df_rb['datetime_begin'].dt.month

# List of columns for which you want to plot graphs
# columns = ['rewards_total', 'rewards_switching', 'rewards_temperature_heat', 'reward_abort', 'reward_policy_overwrite', 'reward_energy_electric', 'energy_cost_cooling_eur', 'cooling_system_coeff', 'switches_per_hour']
columns = ['energy_cost_cooling_eur', 'CER', "cost_CER", 'switches_per_hour', 'rewards_temperature_heat','reward_policy_overwrite', "T_cold_MSE_8", "T_cool_MSE_18",]
for column in columns:
    # Group by month and calculate the mean
    monthly_average1 = df_agent1.groupby('Month')[column].mean().reset_index()
    monthly_average2 = df_agent2.groupby('Month')[column].mean().reset_index()
    monthly_averagerb = df_rb.groupby('Month')[column].mean().reset_index()

    # Create a new DataFrame for plotting
    plot_df = pd.concat([monthly_average1.assign(Agent='Agent 1'),monthly_average2.assign(Agent='Agent 2'), monthly_averagerb.assign(Agent='Rule based control')])

    # Plot the bar graph
    fig = px.bar(plot_df, x='Month', y=column, color='Agent', barmode='group', title='Average ' + column.replace("_", " ") + ' per Month')
    #fig.show()

# complete year
columns1 = ['CER','num_policy_overwrite_per_day', "cost_CER", 'switches_per_hour', "T_cold_MSE_8", "T_cool_MSE_18","T_cool_ME_18",]
columns2 = ['reward_policy_overwrite','energy_cost_cooling_eur']
columns3 = ['rewards_total', 'rewards_switching', 'rewards_temperature_heat', 'reward_abort', 'reward_policy_overwrite', 'reward_energy_electric']
for columns in [columns1, columns2, columns3]:
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

# df_compare_agents["Agents"] = ["Agent 1", "Rule based control"]
# for column in columns:
#     df_compare_agents[column]=0
#     df_compare_agents[column][0] = df_agent1[column].mean()
#     df_compare_agents[column][1] = df_rb[column].mean()
# print(df_compare_agents.set_index("Agents", drop=True, inplace=True))

# # fig = px.bar(df_compare_agents, x=df_compare_agents.columns, y=df_compare_agents.index, title="Cooling System Metrics")
# fig = px.bar(df_compare_agents, x=df_compare_agents.index, y=columns, orientation="h", title="Cooling System Metrics")

# # Create a long-form DataFrame
# # df_long = pd.melt(df_compare_agents, id_vars="Metric", var_name="Agent", value_name="Value")
# #print(df_long)
# # Create the bar plot
# #fig = px.bar(df_long, x="Value", y="Metric", color="Agent", orientation="h", title="Cooling System Metrics")
# fig.show()