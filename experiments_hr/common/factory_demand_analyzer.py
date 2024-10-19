import pandas as pd
import matplotlib.pyplot as plt

# Define the file paths
file_path_2017 = "experiments_hr/supplysystem_b/data/Factory_2017.csv"
file_path_2018 = "experiments_hr/supplysystem_b/data/Factory_2018.csv"

# Load the CSV files
data_2017 = pd.read_csv(file_path_2017, delimiter=';', decimal=',')
data_2018 = pd.read_csv(file_path_2018, delimiter=';', decimal=',')

# Prepare the figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid for the histograms

# Histogram for 'Power_Heat'
axs[0, 0].hist(data_2017['Power_Heat'], bins=30, color='grey', alpha=0.7, label='2017')
axs[0, 0].hist(data_2018['Power_Heat'], bins=30, color='green', alpha=0.7, label='2018')
axs[0, 0].set_title('Histogram of Power_Heat for 2017 and 2018')
axs[0, 0].set_xlabel('Power_Heat (units)')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].legend()

# Histogram for 'Power_Electricity'
axs[0, 1].hist(data_2017['Power_Electricity'], bins=30, color='grey', alpha=0.7, label='2017')
axs[0, 1].hist(data_2018['Power_Electricity'], bins=30, color='green', alpha=0.7, label='2018')
axs[0, 1].set_title('Histogram of Power_Electricity for 2017 and 2018')
axs[0, 1].set_xlabel('Power_Electricity (units)')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].legend()

# Histogram for 'Power_Gas'
axs[1, 0].hist(data_2017['Power_Gas'], bins=30, color='grey', alpha=0.7, label='2017')
axs[1, 0].hist(data_2018['Power_Gas'], bins=30, color='green', alpha=0.7, label='2018')
axs[1, 0].set_title('Histogram of Power_Gas for 2017 and 2018')
axs[1, 0].set_xlabel('Power_Gas (units)')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()

# Histogram for 'Power_Cold'
axs[1, 1].hist(data_2017['Power_Cold'], bins=30, color='grey', alpha=0.7, label='2017')
axs[1, 1].hist(data_2018['Power_Cold'], bins=30, color='green', alpha=0.7, label='2018')
axs[1, 1].set_title('Histogram of Power_Cold for 2017 and 2018')
axs[1, 1].set_xlabel('Power_Cold (units)')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Calculate and print averages for each column
print("Averages for 2017:")
for col in ['Power_Heat', 'Power_Electricity', 'Power_Cold', 'Power_Gas']:
    print(f"{col}: {data_2017[col].mean()}")

print("\nAverages for 2018:")
for col in ['Power_Heat', 'Power_Electricity', 'Power_Cold', 'Power_Gas']:
    print(f"{col}: {data_2018[col].mean()}")
