import pandas as pd

# Define the file path for the 2017 data
file_path_2017 = "experiments_hr/supplysystem_b/data/Factory_2017.csv"

# Load the CSV file
data_2017 = pd.read_csv(file_path_2017, delimiter=';', decimal=',')

# Multiply all entries in the 'Power_Heat' column by 0.6
data_2017['Power_Heat'] *= 0.6

# Define the path for the new CSV file
new_file_path = "experiments_hr/supplysystem_b/data/Modified_Factory_2017.csv"

# Save the modified data to a new CSV file
data_2017.to_csv(new_file_path, sep=';', decimal=',', index=False)

print("Data saved successfully to", new_file_path)
