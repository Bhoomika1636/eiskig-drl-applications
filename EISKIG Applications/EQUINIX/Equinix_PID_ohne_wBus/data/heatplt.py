import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# CSV-Datei laden
csv_datei = 'C:/Users/lawil/OneDrive/Desktop/experiments_hr/experiments_hr/Equinix_PID_ohne_wBus/data/Heat_2017.csv'
df = pd.read_csv(csv_datei, sep=';', parse_dates=['datetime'], index_col='datetime')

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(df['Heat'], label='Heat')
plt.xlim(df.index[0], df.index[-1])
plt.ylim(600000,1000000)
plt.xlabel('Datum')
plt.ylabel('Wärmelast [W]')
plt.title('Wärmelast 2017')
plt.legend()
plt.grid()

# y-Achse anpassen
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

plt.show()