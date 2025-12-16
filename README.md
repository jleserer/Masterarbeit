# Masterarbeit - Stock Price Prediction mit Deep Learning

Dieses Repository enthält die Implementierung von Deep Learning Modellen (LSTM und CNN) zur Vorhersage von Aktienkursen basierend auf historischen Daten.

## Repository-Struktur

```
Masterarbeit/
├── models/                   # Deep Learning Modelle und Implementierungen
│   ├── lstm_model.py         # LSTM-Modell (60-Tage floating window)
│   ├── cnn_model.py          # CNN-Modell (60-Tage floating window)
│   ├── data_preparation.py   # Datenaufbereitung und Sequenzerzeugung
│   ├── model_comparison.py   # Vergleich der Modelle
│   ├── sliceWindow.py        # Referenzimplementierung für floating window
│   ├── test_suite.py         # Umfassende Tests
│   └── README.md             # Detaillierte Modell-Dokumentation
│
├── stockData/                # Aktiendaten und Datenverarbeitung
│   ├── sourceData/           # Rohdaten von Yahoo Finance
│   ├── normalizedData/       # Normalisierte CSV-Dateien (für Modelle)
│   ├── excerptData/          # Daten-Auszüge für Tests
│   ├── get-data/             # Scripts zum Datenabruf
│   ├── plots/                # Datenvisualisierungen
│   └── characterize_excerpt_data.py
│
├── lstm_results_SP500/       # LSTM-Trainingsergebnisse für S&P 500
├── cnn_results_SP500/        # CNN-Trainingsergebnisse für S&P 500
├── model_comparison_results.json  # Vergleichsergebnisse
│
├── lehrmaterial/             # Lehrmaterialien und Übungen
└── profCommunication/        # Kommunikation mit Professor

```

## Hauptkomponenten

### 1. Models Directory (`models/`)

Das Herzstück des Projekts mit allen Machine Learning Modellen und Hilfsfunktionen.

**Detaillierte Dokumentation:** Siehe [models/README.md](models/README.md)

Hauptdateien:
- `lstm_model.py`: LSTM-Modell mit 60-Tage Loopback Window
- `cnn_model.py`: CNN-Modell mit 60-Tage Loopback Window
- `data_preparation.py`: Datenaufbereitung, Normalisierung, Train-Val-Test Split (65%-15%-20%)
- `model_comparison.py`: Automatisierter Vergleich beider Modelle
- `sliceWindow.py`: Referenzimplementierung für das floating window Konzept

### 2. Stock Data Directory (`stockData/`)

Enthält alle Aktiendaten in verschiedenen Verarbeitungsstufen:

- **sourceData/**: Originaldaten von Yahoo Finance
- **normalizedData/**: Bereinigte und normalisierte CSV-Dateien
- **excerptData/**: Kleinere Datenauszüge für schnelle Tests
- **get-data/**: Python-Scripts zum Herunterladen neuer Daten
- **plots/**: Visualisierungen der Datenanalyse

### 3. Ergebnisse

- **lstm_results_SP500/**: Trainierte LSTM-Modelle, Plots, Metriken
- **cnn_results_SP500/**: Trainierte CNN-Modelle, Plots, Metriken
- **model_comparison_results.json**: JSON-Datei mit Vergleichsmetriken

## Schnellstart

### Voraussetzungen

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib yfinance
```

### Einzelnes Modell trainieren

```bash
# LSTM trainieren
cd models
python lstm_model.py

# CNN trainieren
python cnn_model.py
```

### Beide Modelle vergleichen

```bash
cd models
python model_comparison.py
```

Dies führt beide Modelle aus und erstellt einen detaillierten Vergleichsbericht in `model_comparison_results.json`.

## Modell-Architektur

Beide Modelle verwenden:
- **Loopback Window**: 60 Handelstage (backward-looking)
- **Train-Val-Test Split**: 65%-15%-20% (nach Goodfellow et al., 2016)
- **Optimizer**: Adam (Learning Rate: 0.001)
- **Loss**: MSE (Mean Squared Error)
- **Target**: Close-Preis Vorhersage

### LSTM
- 2 LSTM-Layer (128, 64 Units)
- Dropout: 0.2
- Dense Layer: 32 Units

### CNN
- 3 Conv1D-Layer (64, 128, 256 Filter)
- Kernel Size: 5
- Max Pooling: 2
- Dense Layers: 64, 32 Units

## Floating Window Konzept

Das Projekt implementiert ein **backward-looking floating window**:

```
Für jeden Zeitpunkt i:
  Input:  Daten von [i-60, i-59, ..., i-1] (60 Tage Historie)
  Output: Preis am Tag i

Das Fenster "gleitet" über die gesamten Trainingsdaten.
```

Siehe [models/sliceWindow.py](models/sliceWindow.py) für die Referenzimplementierung.

## Datensätze

### Testdatensatz
Primärer Testdatensatz: **S&P 500** (^SPX)
- Zeitraum: 2021-01-01 bis 2024-10-26
- Quelle: Yahoo Finance
- Features: Open, High, Low, Close, Volume, Adj Close

### Referenzdatensätze
Das Repository enthält zusätzliche normalisierte Datensätze in `stockData/normalizedData/`:

- **SP500_historical_data.csv** - S&P 500 Index
- **NASDAQ_historical_data.csv** - NASDAQ Composite
- **DAX_historical_data.csv** - Deutscher Aktienindex
- **FTSE100_historical_data.csv** - Financial Times Stock Exchange 100
- **NIKKEI_historical_data.csv** - Nikkei 225 (Japan)
- **HANG_SENG_historical_data.csv** - Hang Seng Index (Hong Kong)
- **10-Year Bond_historical_data.csv** - 10-Jahres Staatsanleihen
- **30 Year Bond_historical_data.csv** - 30-Jahres Staatsanleihen

Alle Datensätze sind vorverarbeitet und normalisiert, bereit für das Training der Modelle.

## Tests

Umfassende Testsuite:

```bash
cd models
python test_suite.py
```

Tests umfassen:
- Datenaufbereitung
- Sequenzerzeugung
- Modellarchitektur
- Training
- Vorhersagen

## Dokumentation

- **Hauptdokumentation**: [models/README.md](models/README.md)
- **Implementierungs-Details**: [models/IMPLEMENTATION_SUMMARY.txt](models/IMPLEMENTATION_SUMMARY.txt)
- **Quick Reference**: [models/QUICK_REFERENCE.txt](models/QUICK_REFERENCE.txt)
- **Test-Ergebnisse**: [models/TEST_RESULTS.txt](models/TEST_RESULTS.txt)

## Wissenschaftliche Grundlagen

- **Data Split**: Goodfellow, Bengio, Courville (2016) - "Deep Learning"
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **CNN for Time Series**: LeCun, Bengio, Hinton (2015)
