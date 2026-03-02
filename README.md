# Masterarbeit - Stock Price Prediction mit Deep Learning

Dieses Repository enthält die Implementierung von Deep Learning Modellen (LSTM, CNN, GRU und Informer) zur Vorhersage von Aktienkursen basierend auf historischen Daten.

## Repository-Struktur

```
Masterarbeit/
├── models/                          # Deep Learning Modelle und Implementierungen
│   ├── lstm_model.py                # LSTM-Modell (60-Tage floating window, TensorFlow)
│   ├── cnn_model.py                 # CNN-Modell (60-Tage floating window, TensorFlow)
│   ├── gru_model.py                 # GRU-Modell (60-Tage floating window, TensorFlow)
│   ├── informer_model.py            # Informer-Modell (60-Tage floating window, PyTorch)
│   ├── data_preparation.py          # Datenaufbereitung und Sequenzerzeugung
│   ├── model_comparison.py          # Vergleich aller Modelle
│   ├── hyperparameter_tuning.py     # Full Grid Search (LSTM: 48, CNN: 48, GRU: 48, Informer: 64 Configs)
│   ├── evaluate_lookback_window_sweep.py  # Lookback-Window Sweep (L=1..60)
│   └── README.md                    # Detaillierte Modell-Dokumentation
│
├── stockData/                # Aktiendaten und Datenverarbeitung
│   ├── sourceData/           # Rohdaten von Yahoo Finance
│   ├── preprocessedData/     # Bereinigte CSV-Dateien (absolute Preise)
│   ├── get-data/             # Scripts zum Datenabruf
│   └── plots/                # Datenvisualisierungen
│
├── results/                  # Alle Modell-Ergebnisse
│   ├── tuning/               # Hyperparameter-Tuning-Ergebnisse (8 Indizes)
│   │   ├── LSTM/{INDEX}/     # LSTM-Tuning (48 Configs pro Index)
│   │   ├── CNN/{INDEX}/      # CNN-Tuning (48 Configs pro Index)
│   │   ├── GRU/{INDEX}/      # GRU-Tuning (48 Configs pro Index)
│   │   └── INFORMER/{INDEX}/ # Informer-Tuning (64 Configs pro Index)
│   ├── lookback_sweep/       # Lookback-Window Sweep (L=1..60, 8 Indizes)
│   ├── best_configurations.json   # Beste Konfiguration pro Index
│   └── cross_index_comparison.json # Cross-Index Vergleich
│
└── run_pipeline.py           # Parallelisierte Pipeline (Tuning + Sweep)
```

## Hauptkomponenten

### 1. Models Directory (`models/`)

Das Herzstück des Projekts mit allen Machine Learning Modellen und Hilfsfunktionen.

**Detaillierte Dokumentation:** Siehe [models/README.md](models/README.md)

Hauptdateien:
- `lstm_model.py`: LSTM-Modell mit konfigurierbarem Lookback Window (TensorFlow)
- `cnn_model.py`: CNN-Modell mit konfigurierbarem Lookback Window (TensorFlow)
- `gru_model.py`: GRU-Modell mit konfigurierbarem Lookback Window (TensorFlow)
- `informer_model.py`: Informer-Modell mit ProbSparse Self-Attention (PyTorch)
- `data_preparation.py`: Datenaufbereitung, Log-Returns, Train-Val-Test Split (65%-15%-20%)
- `model_comparison.py`: Automatisierter Vergleich aller Modelle
- `hyperparameter_tuning.py`: Full Grid Search (LSTM: 48, CNN: 48, GRU: 48, Informer: 64 = 208 Konfigurationen)
- `evaluate_lookback_window_sweep.py`: Lookback-Window Sweep (L=1..60) mit per-Index bester Konfiguration

### 2. Stock Data Directory (`stockData/`)

Enthält alle Aktiendaten in verschiedenen Verarbeitungsstufen:

- **sourceData/**: Originaldaten von Yahoo Finance
- **preprocessedData/**: Bereinigte CSV-Dateien (absolute Preise, Returns werden zur Laufzeit berechnet)
- **get-data/**: Python-Scripts zum Herunterladen neuer Daten
- **plots/**: Visualisierungen der Datenanalyse

### 3. Pipeline (`run_pipeline.py`)

Maximal parallelisierte Pipeline für das gesamte Experiment:
1. **Hyperparameter-Tuning**: 8 Indizes × 208 Configs = 1664 Tasks parallel
2. **Ergebnisse sammeln**: Beste Konfiguration pro Index ermitteln
3. **Lookback Sweep**: 8 Indizes × 60 L-Werte = 480 Tasks parallel
4. **Cross-Index Vergleich**: Konfigurationen und Ergebnisse vergleichen

### 4. Ergebnisse

- **results/tuning/{LSTM,CNN,GRU,INFORMER}/{INDEX}/**: Hyperparameter-Tuning-Ergebnisse pro Index
- **results/lookback_sweep/{INDEX}/**: Lookback-Window Sweep (L=1..60, 8 Indizes)
- **results/best_configurations.json**: Beste Konfiguration pro Index und Modelltyp
- **results/cross_index_comparison.json**: Vergleich über alle Indizes

## Schnellstart

### Voraussetzungen

```bash
pip install tensorflow pandas numpy matplotlib yfinance torch
```

### Einzelnes Modell trainieren

```bash
cd models
python lstm_model.py
python cnn_model.py
python gru_model.py
```

### Alle drei Modelle vergleichen

```bash
cd models
python model_comparison.py
```

### Gesamte Pipeline ausführen

```bash
python run_pipeline.py                # Alles parallel (8 Workers)
python run_pipeline.py --workers 16   # Mehr Parallelität
python run_pipeline.py --skip-tuning  # Tuning überspringen
```

## Modell-Architektur

Alle vier Modelle verwenden:
- **Lookback Window**: 60 Handelstage (backward-looking, konfigurierbar)
- **Train-Val-Test Split**: 65%-15%-20% (nach Goodfellow et al., 2016)
- **Optimizer**: Adam
- **Loss**: MSE (Mean Squared Error)
- **Input**: Log-Returns r(t) = ln(P(t) / P(t-1))
- **Target**: Close-Log-Return (Single-Output)

### LSTM (TensorFlow)
- 2 LSTM-Layer (128, 64 Units)
- Dropout: 0.2
- Dense Layer: 32 Units
- Learning Rate: 0.001

### GRU (TensorFlow)
- 2 GRU-Layer (128, 64 Units)
- Dropout: 0.2
- Dense Layer: 32 Units
- Learning Rate: 0.001

### CNN (TensorFlow)
- 3 Conv1D-Layer (64, 128, 256 Filter)
- Kernel Size: 5, Max Pooling: 2
- Dense Layers: 64, 32 Units
- Learning Rate: 0.001

### Informer (PyTorch)
- ProbSparse Self-Attention (O(L log L) statt O(L²))
- Self-Attention Distilling (ConvPool zwischen Encoder-Layers)
- Encoder-Decoder Architektur (2 Encoder-Layer, 1 Decoder-Layer)
- d_model: 64, n_heads: 8, d_ff: 256
- Dropout: 0.05, Learning Rate: 0.0001
- Single-step Prediction (pred_len=1, label_len=L/2)

## Floating Window Konzept

Das Projekt implementiert ein **backward-looking floating window**:

```
Für jeden Zeitpunkt i:
  Input:  Daten von [i-L, i-L+1, ..., i-1] (L Tage Historie)
  Output: Return am Tag i

Das Fenster "gleitet" über die gesamten Trainingsdaten.
```

## Datensätze

Das Repository enthält 8 Datensätze in `stockData/preprocessedData/`:

- **SP500_historical_data.csv** - S&P 500 Index
- **NASDAQ_historical_data.csv** - NASDAQ Composite
- **DAX_historical_data.csv** - Deutscher Aktienindex
- **FTSE100_historical_data.csv** - Financial Times Stock Exchange 100
- **NIKKEI_historical_data.csv** - Nikkei 225 (Japan)
- **HANG_SENG_historical_data.csv** - Hang Seng Index (Hong Kong)
- **10-Year Bond_historical_data.csv** - 10-Jahres Staatsanleihen
- **30 Year Bond_historical_data.csv** - 30-Jahres Staatsanleihen

Alle Datensätze enthalten absolute Preise. Log-Returns werden zur Laufzeit in `data_preparation.py` berechnet.

## Wissenschaftliche Grundlagen

- **Data Split**: Goodfellow, Bengio, Courville (2016) - "Deep Learning"
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"
- **CNN for Time Series**: LeCun, Bengio, Hinton (2015)
- **Informer**: Zhou et al. (2021) - "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper)
