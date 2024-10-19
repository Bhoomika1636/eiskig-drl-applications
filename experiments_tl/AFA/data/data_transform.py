import pandas as pd

# Load the CSV data into a pandas DataFrame
file_path = 'C:\GitHub\experiments_hr\experiments_hr\AFA\data\Mappe1.csv'  # Change to your actual file path
df = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime type
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set 'Timestamp' as the index
df.set_index('Timestamp', inplace=True)

# Resample the data to a 10-minute frequency and interpolate the missing values
df_resampled = df.resample('10T').interpolate(method='linear')

# Reset the index if needed
df_resampled.reset_index(inplace=True)

# Save the resampled DataFrame to a new CSV file if needed
df_resampled.to_csv('C:\GitHub\experiments_hr\experiments_hr\AFA\data\Versuchsreihe_27052024-Kopie.csv', index=False)

print(df_resampled)