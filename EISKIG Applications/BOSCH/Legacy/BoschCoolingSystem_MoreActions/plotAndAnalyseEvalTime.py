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
path = 'experiments_hr/BoschCoolingSystem2/results/compare_ppo_rb/'

# Get a list of all the csv files in the specified directory
all_files_ag1 = glob.glob(path + 'rb*episode.csv')

# Create a list to hold the dataframes
df_list = []

# Loop through the list of files and read each one into a dataframe
for filename in all_files_ag1:
    df = pd.read_csv(filename, index_col=None, header=0,sep=";")
    df_list.append(df.fillna(0))

# Concatenate all the dataframes in the list
combined_df = pd.concat(df_list, axis=0, ignore_index=True)
print(combined_df.columns)
# build eval df
eval_df = pd.DataFrame()
eval_df['SE_T_cooling_ag1'] = np.square(combined_df['T_in_coolingCircuit'] - 18)
eval_df['switches_per_hour_ag1'] = combined_df['switches_per_hour']
# eval_df.index = combined_df[]
eval_df['difference_cold'] = combined_df['Temperature_cwCircuit_out'] - combined_df['Temperature_cwCircuit_in']
print(f"MSE cooling_ag1: {eval_df['SE_T_cooling_ag1'].mean()}")
print(f"Max Cold difference T: {eval_df['difference_cold'].mean()}")

#print(combined_df)