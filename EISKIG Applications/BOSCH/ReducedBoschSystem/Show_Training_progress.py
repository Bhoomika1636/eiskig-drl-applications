import pandas as pd
import numpy as np
import glob

# Define the agID for which you want to read all files
agID = 12

# Specify the path to the folder containing the files
folder_path = r'C:\Users\Askari-Badouee_David\Documents\GitHub\experimentshr_ast\experiments_hr\ReducedBoschSystem\results\ppo_red_3'

# Pattern to match the files
pattern = f'{folder_path}\\agent_{agID}_*all-episodes.csv'

# Find all matching files
files = glob.glob(pattern)

# Initialize a list to store DataFrames
dfs = []

# Read each file into a DataFrame and append it to the list
for file in files:
    df = pd.read_csv(file, sep=';')
    
    # Select only numeric columns, excluding specific non-relevant ones
    df_numeric = df.select_dtypes(include=['number'])
    df_numeric = df_numeric.drop(columns=['time', 'n_steps'], errors='ignore')
    
    dfs.append(df_numeric)

# Check if we have at least one DataFrame to process
if not dfs:
    raise ValueError("No files found or all files are empty/invalid.")

# Align shapes by ensuring all DataFrames have the same columns in the same order
# This step assumes all DataFrames should have the same set of columns
# If files could have different sets of columns, additional handling is required
common_columns = sorted(dfs[0].columns)
dfs = [df.reindex(columns=common_columns) for df in dfs]

# Stack the DataFrames and calculate the mean along the new axis (0 is rows, so use 2 for depth)
stacked = np.stack([df.values for df in dfs])
mean_values = np.mean(stacked, axis=0)

# Convert the mean values back into a DataFrame
df_mean = pd.DataFrame(mean_values, columns=common_columns)

print(df_mean)

import plotly.express as px
# smoothing 
for column in df_mean.columns:
    try:
        df_mean[column] = pd.to_numeric(df_mean[column])
        df_mean[column] = df_mean[column].rolling(window=6).mean()
    except:
        print("non numeric values")
# Assuming df_mean is your DataFrame and its index represents the x-axis
# and you're plotting all columns in the DataFrame


# For multiple line plots (if you want to plot multiple columns)
# Here, we melt the DataFrame to long format which Plotly handles well for multiple lines
df_long = df_mean.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
fig = px.line(df_long, x='index', y='Value', color='Metric', title='Mean Metric Values Over Time')
fig.update_layout(xaxis_title='Index', yaxis_title='Mean Value')
fig.show()

import os
import matplotlib.pyplot as plt
# Save the DataFrame as a CSV
csv_path = os.path.join(folder_path, f'{agID}mean_values.csv')
df_mean.to_csv(csv_path, index=False)

# Generate and save the Plotly figure as HTML
fig = px.line(df_long, x='index', y='Value', color='Metric', title='Mean Metric Values Over Time')
html_path = os.path.join(folder_path, f'{agID}mean_metrics_plot.html')
fig.write_html(html_path)

# Generate and save the Matplotlib figure as a PDF
plt.figure(figsize=(10, 6))
for column in df_mean.columns:
    if "reward" in column and column != 'rewards_total':
        plt.plot(df_mean.index, df_mean[column], label=column)
plt.title('Reward Metrics Over Time')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(axis="y")
pdf_path = os.path.join(folder_path, f'{agID}reward_metrics_plot.pdf')
plt.savefig(pdf_path)
plt.close()

# Now, you have saved the DataFrame, the Plotly figure, and the Matplotlib figure in the specified directory.
print(f"Saved CSV at: {csv_path}")
print(f"Saved Plotly HTML at: {html_path}")
print(f"Saved Matplotlib PDF at: {pdf_path}")