import json

import pandas as pd

# Load the data from a JSON file
with open("experiments_hr/supplysystem_b/results/parameter_evaluation/PIEvaluation.json") as f:
    data = json.load(f)

# Initialize an empty DataFrame
df = pd.DataFrame()

# Iterate over the first-level keys and values in the dictionary
for key, sub_dict in data.items():
    # Convert each sub-dictionary to a DataFrame and transpose it
    sub_df = pd.DataFrame(sub_dict).T
    # Flatten the DataFrame and create a MultiIndex column
    sub_df = sub_df.stack().unstack(0)
    # Name the column with the first-level key
    sub_df.columns = pd.MultiIndex.from_product([[key], sub_df.columns])
    # Join with the main DataFrame
    df = pd.concat([df, sub_df], axis=1)

# Sort the DataFrame by the index
df = df.sort_index()

# Transpose the DataFrame before saving
df_transposed = df.transpose()

# Save the transposed DataFrame to an Excel file
df_transposed.to_excel(
    "experiments_hr/supplysystem_b/results/parameter_evaluation/PIEvaluation.xlsx", engine="openpyxl"
)
