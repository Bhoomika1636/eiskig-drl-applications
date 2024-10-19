import pandas as pd
import plotly.express as px
import os
folder_path = "C:\Gitlab\HIST_Export-20231004T170823_Juni 2023"
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files[10:12]:
    file_path = os.path.join(folder_path, file)
    time_index = pd.read_csv(file_path, usecols=[0], skiprows=4, delimiter=';')
    df = pd.read_csv(file_path, skiprows=[1,2,3,4],header=0, delimiter=';', usecols=lambda column: column != 'Name :') 
    df.index = pd.to_datetime(time_index.iloc[:,0], dayfirst=True)
    dfs.append(df)

# Concatenate all DataFrames along axis 1
concatenated_df = pd.concat(dfs, axis=1)
print(concatenated_df)
df = concatenated_df
df.index = pd.to_datetime(df.index)

# df.set_index('Time', inplace=True)
# ids = df.columns.str.extract(r'(.+?)\s+{')
# print(df)

# rowNames = df.columns.str.extract(r'name="([^"]*)"')
# old_colu = df.columns
# df.columns = rowNames.squeeze()
# print(df)

# # Convert column names to strings
# df.columns = df.columns.astype(str)
# df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

# work with data

# drop all columns that only have one value 
columns_to_keep = df.columns[df.nunique(dropna=True) > 1]
df_filtered = df[columns_to_keep]
df_filtered.drop_duplicates(keep='first')
for i in df_filtered.columns:
    try:
        df_filtered[i] = pd.to_numeric(df_filtered[i])
    except:
        df_filtered.drop(i)

df_filtered = df_filtered.bfill(axis=0).ffill(axis=0)
df_filtered = df_filtered.resample('T').mean()
print(df_filtered)
# Calculate percentage of NaN values in each column
nan_percentage = df.isna().mean() * 100
# Drop columns with more than 50% NaN values
columns_to_keep = nan_percentage[nan_percentage <= 50].index
# df_filtered = df[columns_to_keep]
df_filtered.to_csv('Bosch_largeDS_simplified.csv')
# fig = px.line(df,df.index, df.columns[0], markers=False)
# fig.show()