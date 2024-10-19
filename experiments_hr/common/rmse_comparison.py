import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_rmse(series1, series2):
    return math.sqrt(np.mean((series1 - series2) ** 2))


# Get the directory of the current python script
script_directory = os.path.dirname(os.path.abspath(__file__))
print("Directory of the Python script:", script_directory)

# Path to the P1.csv file
p1_csv_file = os.path.join(script_directory, "P1.csv")

# Load all CSV files whose name ends with _episode.csv
csv_files = glob.glob(os.path.join(script_directory, "*_episode.csv"))

# Specify column names
columns_to_compare = ["s_temp_heat_storage_hi", "s_temp_cold_storage_lo"]  # Add more column names as needed

# Function to read specific columns from a file
def read_columns(file, columns):
    return pd.read_csv(file, delimiter=";", low_memory=False)[columns]


# Read the specific columns from P1.csv
p1_series = {col: read_columns(p1_csv_file, col) for col in columns_to_compare}

# Prepare DataFrame to store results
results_df = pd.DataFrame(columns=["File", "Average RMSE"] + columns_to_compare)

# Calculate RMSE between P1.csv and each other file for each column
for file in tqdm(csv_files, desc="Processing files"):
    file_name = os.path.basename(file)
    rmses = []
    temp_row = {"File": file_name}
    for col in columns_to_compare:
        temp_series = read_columns(file, col)
        rmse = calculate_rmse(p1_series[col], temp_series)
        rmses.append(rmse)
        temp_row[col] = rmse

    avg_rmse = sum(rmses) / len(rmses)
    temp_row["Average RMSE"] = avg_rmse
    results_df = pd.concat([results_df, pd.DataFrame([temp_row])], ignore_index=True)

# Save results to CSV
results_df.to_csv(os.path.join(script_directory, "comparison.csv"), index=False)

# Bar plot
plt.figure(figsize=(10, 6))
plt.barh(results_df["File"], results_df["Average RMSE"], color="black")  # Set bar color to black
plt.xlabel("Average RMSE")
plt.ylabel("CSV File")
plt.title("Average RMSE Comparison")
plt.tight_layout()
plt.savefig(os.path.join(script_directory, "rmse_comparison_plot.png"))
plt.show()
