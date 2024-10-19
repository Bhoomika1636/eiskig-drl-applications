import json
import os

import pandas as pd


def process_json_to_excel(folder_path):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            # Construct the full file path
            json_path = os.path.join(folder_path, filename)

            # Load the JSON file into a Python dictionary
            with open(json_path) as file:
                nested_dict = json.load(file)

            # Convert the nested dictionary to a DataFrame and transpose it
            df = pd.DataFrame(nested_dict).transpose()

            # Transpose the DataFrame so that the first-level keys are columns and the second-level keys are rows
            df_transposed = df.transpose()

            # Reset the index to get the column names right
            df_transposed.reset_index(inplace=True)
            df_transposed.rename(columns={"index": "Parameter"}, inplace=True)

            # Construct the output Excel file path
            excel_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.xlsx")

            # Save the transposed DataFrame to an Excel file
            df_transposed.to_excel(excel_path, index=False)

            print(f"Exported {json_path} to {excel_path}")


# Replace 'your_folder_path' with the path to your folder containing JSON files
process_json_to_excel("experiments_hr/supplysystem_b/results/sensitivity_analysis")
