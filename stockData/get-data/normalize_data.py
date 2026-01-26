import os
import pandas as pd
from datetime import datetime

# Pfade definieren
SOURCE_DIR = "stockData/sourceData"
NORMALIZED_DIR = "stockData/preprocessedData"

# Stelle sicher, dass das preprocessedData Verzeichnis existiert
os.makedirs(NORMALIZED_DIR, exist_ok=True)

# Liste alle CSV-Dateien im sourceData Verzeichnis
source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]

print(f"Gefundene Dateien: {len(source_files)}")
print("-" * 80)

# Dictionary um die frühesten Daten zu speichern
earliest_dates = {}

# Erste Durchgang: Finde das früheste Datum in jeder Datei
for filename in source_files:
    source_path = os.path.join(SOURCE_DIR, filename)
    
    # Lese die CSV-Datei
    df = pd.read_csv(source_path)
    
    # Überspringe die ersten 2 Zeilen (Header-Zeilen mit Ticker)
    df = df.iloc[2:].reset_index(drop=True)
    
    # Konvertiere die Date-Spalte zu datetime
    df['Price'] = pd.to_datetime(df['Price'], errors='coerce')
    
    # Entferne Zeilen mit ungültigen Daten
    df = df.dropna(subset=['Price'])
    
    # Finde das früheste Datum
    if not df.empty:
        earliest_date = df['Price'].min()
        earliest_dates[filename] = earliest_date
        print(f"{filename}: Frühestes Datum = {earliest_date.strftime('%Y-%m-%d')}")

print("\n" + "=" * 80)

# Finde das späteste "früheste Datum" (gemeinsamer Startpunkt)
common_start_date = max(earliest_dates.values())
print(f"\nGemeinsamer Startpunkt für alle Datensätze: {common_start_date.strftime('%Y-%m-%d')}")
print("=" * 80)

# Zweiter Durchgang: Normalisiere die Daten
for filename in source_files:
    source_path = os.path.join(SOURCE_DIR, filename)
    normalized_path = os.path.join(NORMALIZED_DIR, filename)
    
    print(f"\nVerarbeite: {filename}")
    
    # Lese die CSV-Datei
    df = pd.read_csv(source_path)
    
    # Speichere die ersten 2 Zeilen (Header mit Ticker-Info)
    header_rows = df.iloc[:2].copy()
    
    # Überspringe die ersten 2 Zeilen für die Datenverarbeitung
    df = df.iloc[2:].reset_index(drop=True)
    
    # Konvertiere die Date-Spalte zu datetime
    df['Price'] = pd.to_datetime(df['Price'], errors='coerce')
    
    # Entferne Zeilen mit ungültigen Daten
    df = df.dropna(subset=['Price'])
    
    # Filtere Daten ab dem gemeinsamen Startdatum
    df_filtered = df[df['Price'] >= common_start_date].copy()
    
    # Entferne die Spalten "Open" und "Volume"
    columns_to_keep = [col for col in df_filtered.columns if col not in ['Open', 'Volume']]
    df_normalized = df_filtered[columns_to_keep]
    
    # Aktualisiere die Header-Zeilen (entferne Open und Volume)
    header_normalized = header_rows[columns_to_keep]
    
    # Kombiniere Header und Daten
    df_final = pd.concat([header_normalized, df_normalized], ignore_index=True)
    
    # Speichere die normalisierte Datei
    df_final.to_csv(normalized_path, index=False)
    
    print(f"  Original Datensätze: {len(df)}")
    print(f"  Nach Filterung (ab {common_start_date.strftime('%Y-%m-%d')}): {len(df_normalized)}")
    print(f"  Entfernte Datensätze: {len(df) - len(df_normalized)}")
    print(f"  Gespeichert: {normalized_path}")

print("\n" + "=" * 80)
print("Normalisierung abgeschlossen!")
print("=" * 80)
