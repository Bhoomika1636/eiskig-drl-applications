"""
This script is to test the DWD connector from eta-utility 3.1.0 - version upgrade is needed manually
to upgrade eta-utility also stable-baselines3 v2.2.1 is needed
"""

from eta_utility.connectors.wetterdienst import WetterdienstConnection, WetterdienstPredictionConnection
from eta_utility.connectors.node import NodeWetterdienstObservation, NodeWetterdienstPrediction
from wetterdienst.provider.dwd.observation.api import DwdObservationRequest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ETA_LOCATION = (49.86376654168076, 8.681726558050716)

node = NodeWetterdienstObservation(
    name="Global Radiation",  # Ensure 'name' attribute is present
    url="https://opendata.dwd.de/",  # base URL for the Wetterdienst API
    protocol="wetterdienst_observation",  # protocol used for observation
    #station_id="05906",  # Darmstadt - no global radiation available
    latlon = ETA_LOCATION, # picks closest stations near the ETA location where the parameter is available
    number_of_stations = 3, # how many stations should be used
    parameter="RADIATION_GLOBAL",
    interval="600",  # 10 min in seconds
)

# start connection from one node
connection = WetterdienstConnection.from_node(node)

# Define time interval as datetime values
current_time = datetime.now()
from_datetime = datetime(2024, 5, 27, 10, 0)
to_datetime = datetime(2024, 5, 29, 23, 50)


# read_series will request data from specified connection and time interval
if isinstance(connection, WetterdienstConnection):
    df = connection.read_series(from_time=from_datetime, to_time=to_datetime, interval=600)
else:
    raise TypeError("The connection must be an WetterdienstConnection, to be able to call read_series.")

# print(df)

# Interpolating the columns
df['Interpolated_Global_Radiation'] = df.mean(axis=1)

print(df)
df.to_csv("experiments_hr\AFA\data\Solar_2705_interpolated.csv", index=True)