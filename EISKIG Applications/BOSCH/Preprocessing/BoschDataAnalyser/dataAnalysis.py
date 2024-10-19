import pandas as pd
import plotly.express as px
df = pd.read_csv('experiments_hr\BoschData\Daten_eta.csv')
df.set_index('Time', inplace=True)
ids = df.columns.str.extract(r'(.+?)\s+{')
print(df)
# parts = df.columns.str.extract(r'{(.*)}')[0].str.extractall(r'(\w+)="([^"]+)"')
# df2 = pd.DataFrame()
# df2.columns = parts['0'].unique()
rowNames = df.columns.str.extract(r'name="([^"]*)"')
old_colu = df.columns
df.columns = rowNames.squeeze()
print(df)

# Convert column names to strings
df.columns = df.columns.astype(str)
df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

# work with data

# drop all columns that only have one value 
columns_to_keep = df.columns[df.nunique(dropna=True) > 1]
df_filtered = df[columns_to_keep]

# Calculate percentage of NaN values in each column
nan_percentage = df.isna().mean() * 100
# Drop columns with more than 80% NaN values
columns_to_keep = nan_percentage[nan_percentage <= 80].index
df_filtered = df[columns_to_keep]

fig = px.line(df,df.index, df.columns[0], markers=False)
fig.show()