import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('experiments_hr\supplysystem_ETA\data\Weather_Frankfurt_2017_2018.csv', decimal=',', delimiter=';')


data['wind_speed'] = data['wind_speed'].apply(lambda x: 0.5 if pd.notnull(x) and x < 0.5 else x)

data['global_radiation'] = data['global_radiation'].apply(lambda x: 0 if pd.notnull(x) and x < 0 else x)*0.1

data['ts100'] = data['ts100'].apply(lambda x: 10 if pd.notnull(x) and x < 10 else x)

data.to_csv('experiments_hr\supplysystem_ETA\data\Weather_Frankfurt_2017_2018_QSolar.csv', index=False, decimal=',', sep=';')
