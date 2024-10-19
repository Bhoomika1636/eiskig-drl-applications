from __future__ import annotations

import pathlib

import sys
import numpy as np
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#### second part with time data
import glob
import pprint

# Specify the relative path to the files
path = 'experiments_hr/ReducedBoschSystem/results/compare_ppo_rb3/'
# path = 'experiments_hr/ReducedBoschSystem/results/compare_FY/'

# Get a list of all the csv files in the specified directory
all_files_ag1 = glob.glob(path + 'rb*episode.csv')
# all_files_ag1 = glob.glob(path + 'agent_2*episode.csv')

# Create a list to hold the dataframes
df_list = []

# Loop through the list of files and read each one into a dataframe
i = 0
for filename in all_files_ag1:
    df = pd.read_csv(filename, index_col=None, header=0,sep=";")
    df_list.append(df.fillna(0))

# Concatenate all the dataframes in the list
combined_df = df_list[20]#pd.concat(df_list, axis=0, ignore_index=True)
#print(combined_df.columns)
# build eval df
eval_df = pd.DataFrame()
eval_df['SE_T_cooling_ag1'] = np.square(combined_df['T_in_coolingCircuit'] - 18)
eval_df['switches_per_hour_ag1'] = combined_df['switches_per_hour']
# eval_df.index = combined_df[]
eval_df['difference_cold'] = combined_df['Temperature_cwCircuit_out'] - combined_df['Temperature_cwCircuit_in']
eval_df["Temperature_cwCircuit_in"] = combined_df['Temperature_cwCircuit_in']
eval_df["T_in_coolingCircuit"] = combined_df['T_in_coolingCircuit']
eval_df["Q_removed"] = combined_df['Q_removed']

print(f"T cool: {eval_df['T_in_coolingCircuit'].mean()}")
print(f"T cold: {eval_df['Temperature_cwCircuit_in'].mean()}")
print(f"Q_removed: {eval_df['Q_removed'].sum()/1000}")
print(f"P_elges: {combined_df['P_el_ges_kw'].sum()}")
print(f"CER: {eval_df['Q_removed'].sum()/(1000*combined_df['P_el_ges_kw'].sum())}")
combined_df['dateTime']=pd.to_datetime(combined_df['dateTime'])
combined_df.set_index('dateTime', drop=True,inplace=True)
for i in combined_df.columns:
    try:
        combined_df[i] = combined_df[i].astype(float)
    except:
        combined_df.drop(columns=i, inplace=True)
combined_df=combined_df.resample('H').mean().dropna(axis=0)

fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns)# print(f"MSE cooling_ag1: {eval_df['SE_T_cooling_ag1'].mean()}")
fig.show()

# Separating the plots based on cool1, cool2, and cold temperatures
# Creating separate dataframes for each category based on column names
df_filtered = pd.DataFrame()
df_filtered['T_cool2_supply_is'] = combined_df['T_in_coolingCircuit']
df_filtered['T_cool2_return_is'] = combined_df['T_out_coolingCircuit']
df_filtered['T_cool2_supply_sp'] = 17
df_filtered['T_cool1_supply_is'] = combined_df['T_main_vorlauf']
df_filtered['T_cool1_return_is'] = combined_df['T_main_ruecklauf']
df_filtered['T_cool1_supply_sp'] = 14
df_filtered['Time'] = combined_df.index
df_cool1 = df_filtered[['Time', 'T_cool1_supply_sp', 'T_cool1_supply_is', 'T_cool1_return_is']]
df_cool2 = df_filtered[['Time', 'T_cool2_supply_sp', 'T_cool2_supply_is', 'T_cool2_return_is']]

# Plotting each category in a separate subplot
plt.rcParams.update({'font.size': 21})
fig, axs = plt.subplots(2, 1, figsize=(15, 12))

for column in df_cool1.columns:
    if column != 'Time':
        axs[0].plot(df_cool1['Time'], df_cool1[column], label=column, marker='o')
axs[0].set_title('Cooling water main')
axs[0].set_ylabel('Temperature (°C)')
axs[0].legend()
axs[0].tick_params(axis='x', rotation=45)

# cool2
for column in df_cool2.columns:
    if column != 'Time':
        axs[1].plot(df_cool2['Time'], df_cool2[column], label=column, marker='o')
axs[1].set_title('Cooling water secondary')
axs[1].set_ylabel('Temperature (°C)')
axs[1].legend()
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("temps_sim.pdf")
plt.show()

# start_date = '2019-07-01'
# end_date = '2019-09-01'
# df = combined_df[start_date:end_date]
# # Calculating jumps from 0 to >0 and >0 to 0
# df['shifted'] = df['u_KT_4'].shift(1)  # Shift the 'values' column down to compare previous row values
# # Initial condition: where the first value is non-zero (considered a jump if it starts with non-zero)
# jumps = 1 if df.iloc[0]['u_KT_4'] > 0 else 0
# # Adding jumps where value jumps from 0 to >0 or from >0 to 0
# jumps += ((df['u_KT_4'] > 0) & (df['shifted'] == 0) | (df['u_KT_4'] == 0) & (df['shifted'] > 0)).sum()

# print(jumps) # 81 total