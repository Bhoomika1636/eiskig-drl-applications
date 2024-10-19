import pandas as pd
import numpy as np
import os

# Erstellen eines Datumsbereichs für das Jahr 2018 mit 15-Minuten-Intervallen
date_rng = pd.date_range(start='1/1/2018', end='1/1/2019', freq='15T')

# Parameter für die Sinuswellen
amplitude = 20000
period = 365.25 / 2 * 24 * 4  # Periodenlänge in 15-Minuten-Intervallen (ein halbes Jahr)
frequency = 2 * np.pi / period  # Frequenz berechnet als 2*pi/Periodenlänge

# Sinusfolgen mit einem Abstand von 100000
sinus_1 = 800000 + amplitude * np.sin(frequency * np.arange(len(date_rng)))
sinus_2 = 800000 + amplitude * np.sin(frequency * np.arange(len(date_rng))) + 50000

# Generieren von zufälligen Werten zwischen den beiden Sinusfunktionen
values = np.random.uniform(sinus_1, sinus_2)

# Erstellen eines DataFrames
df = pd.DataFrame({'datetime': date_rng, 'Heat': values})

# Entfernen der Sekunden aus der Zeit
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# Überprüfen der Spaltennamen im DataFrame
print("Spalten im DataFrame:", df.columns)

# Den aktuellen Ordnerpfad abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# Den vollständigen Pfad für die CSV-Datei erstellen
file_path = os.path.join(current_directory, 'Heat_2018.csv')

# Speichern des DataFrames in eine CSV-Datei mit Semikolon als Trennzeichen
df.to_csv(file_path, index=False, sep=';')

print(f"CSV-Datei wurde erfolgreich erstellt: {file_path}")