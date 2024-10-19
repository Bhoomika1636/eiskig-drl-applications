from ctypes import Array
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from dateutil import tz
from eta_utility.connectors.entso_e import ENTSOEConnection
from eta_utility.connectors.node import NodeEntsoE as Node

"""
This script is just for testing purposes. You can test your API-key here!

"""


API_TOKEN = "fcd0ef8b-887a-4c92-8b33-39cc270b664e"  ### INSERT API KEY HERE

# Check out NodeEntsoE documentation for endpoint and bidding zone information
node = Node(
    name="ENTSOE-DEU-LUX",
    url="https://web-api.tp.entsoe.eu/",
    protocol="entsoe",
    endpoint="Price",
    bidding_zone="DEU-LUX",
)

# Start connection from one or multiple nodes
server = ENTSOEConnection.from_node(node, api_token=API_TOKEN)

current_time = datetime.utcnow()  # ask for current time with UTC format
current_time = current_time.replace(second=0, microsecond=0, minute=0)  # round down to full hours

# Add 6 hours to the rounded time to get the target time
six_hours_later = current_time + timedelta(hours=6)

print(current_time)
print(six_hours_later)

to_datetime = current_time - timedelta(hours=9)
from_datetime = to_datetime - timedelta(days=30)

# from_datetime = current_time
# to_datetime = six_hours_later
# interval: interval between time steps. It is interpreted as seconds if given as integer. e. g. interval=60 means one data point per minute
df_energy_price = server.read_series(from_time=from_datetime, to_time=to_datetime, interval=60 * 60)
print(df_energy_price)

price_0h = df_energy_price["ENTSOE-DEU-LUX_60"][0] * 0.001
price_1h = df_energy_price["ENTSOE-DEU-LUX_60"][1] * 0.001
price_3h = df_energy_price["ENTSOE-DEU-LUX_60"][3] * 0.001
price_6h = df_energy_price["ENTSOE-DEU-LUX_60"][6] * 0.001

print(price_0h)
print(price_1h)
print(price_3h)
print(price_6h)

df_energy_price.to_csv(
    r"C:\electricity_price_1month.csv",
    sep=";",
    decimal=",",
    index=True,
)
