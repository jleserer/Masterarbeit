# Masterarbeit - Stock Price Prediction mit Deep Learning

Vorhersage von Aktienkursen mit LSTM, CNN, GRU und Informer auf Basis von Log-Returns für 8 Finanzindizes.

## Projektstruktur

```
Masterarbeit/
├── models/                              # Modell-Implementierungen
│   ├── lstm_model.py                    # LSTM (TensorFlow)
│   ├── cnn_model.py                     # CNN (TensorFlow)
│   ├── gru_model.py                     # GRU (TensorFlow)
│   ├── informer_model.py               # Informer (PyTorch)
│   └── README.md                        # Modell-Dokumentation
│
├── data_preparation/                    # Schritt 1: Datenaufbereitung
│   └── data_preparation.py             # Log-Returns, Splits, Sequenzen
│
├── parameter_tuning/                    # Schritt 2: Hyperparameter-Optimierung
│   ├── parameter_tuning.py             # Full Grid Search (208 Configs)
│   └── results/                         # Tuning-Ergebnisse pro Index
│       ├── {LSTM,CNN,GRU,INFORMER}/{INDEX}/
│       └── best_configurations.json
│
├── lookback_window_sweep/               # Schritt 3: Lookback-Window Sweep
│   ├── lookback_window_sweep.py
│   ├── rerun_sp500.py                   # Helper: SP500 nachlaufen
│   └── results/{INDEX}/                 # Sweep-Ergebnisse (L=1..60)
│
├── comparison/                          # Schritt 4: Vergleich & Auswertung
│   ├── model_comparison.py              # Vergleich aller 4 Modelle
│   ├── post_processing.py              # Erweiterte Metriken (DA%, R²)
│   └── results/                         # Vergleichsergebnisse
│       ├── {INDEX}/                     # Per-Index Plots & Metriken
│       └── cross_index_comparison.json
│
├── stockData/                           # Aktiendaten
│   ├── sourceData/                      # Rohdaten von Yahoo Finance
│   ├── preprocessedData/                # Bereinigte CSV-Dateien
│   ├── get-data/                        # Scripts zum Datenabruf
│   └── plots/                           # Datenvisualisierungen
│
├── run_pipeline.py                      # Parallelisierte Pipeline (Orchestrator)
└── quick_verify.py                      # Schnelltest aller Modelle
```

## Pipeline

Die Pipeline (`run_pipeline.py`) führt das gesamte Experiment parallelisiert aus:

```
Schritt 1: Parameter-Tuning         8 Indizes × 208 Configs = 1664 Tasks
Schritt 2: Ergebnisse sammeln       Beste Config pro Index ermitteln
Schritt 3: Lookback Sweep           8 Indizes × 60 L-Werte = 480 Tasks
Schritt 4: Cross-Index Vergleich    Konfigurationen vergleichen
```

```bash
python run_pipeline.py                  # Alles parallel (8 Workers)
python run_pipeline.py --workers 16     # Mehr Parallelität
python run_pipeline.py --skip-tuning    # Tuning überspringen
python run_pipeline.py --clean-only     # Nur aufräumen
```

Nach Pipeline-Abschluss:

```bash
python comparison/post_processing.py    # Erweiterte Metriken berechnen
```

## Schnellstart

### Voraussetzungen

```bash
pip install tensorflow pandas numpy matplotlib seaborn yfinance torch
```

### Einzelnes Modell trainieren

```bash
cd models
python lstm_model.py
python cnn_model.py
python gru_model.py
python informer_model.py
```

### Modelle vergleichen

```bash
python comparison/model_comparison.py
```

## Modelle

Alle vier Modelle verwenden dasselbe Interface und identischen Datenfluss:

```
preprocessedData/*.csv
    -> Log-Returns: r(t) = ln(P(t) / P(t-1))
    -> 65%-15%-20% Split (Goodfellow et al., 2016)
    -> Sequences (Lookback Window L=1..60)
    -> Training (feste Epochenzahl, kein EarlyStopping)
    -> Predictions (Returns -> Preise via Inverse Transform)
```

| Modell | Framework | Architektur | Tuning-Configs |
|--------|-----------|-------------|----------------|
| LSTM | TensorFlow | 2× LSTM (128, 64) + Dense(32) | 48 |
| CNN | TensorFlow | 3× Conv1D (64, 128, 256) + Dense(64, 32) | 48 |
| GRU | TensorFlow | 2× GRU (128, 64) + Dense(32) | 48 |
| Informer | PyTorch | ProbSparse Attention + Encoder-Decoder | 64 |

Detaillierte Modell-Dokumentation: [models/README.md](models/README.md)

## Datenaufbereitung (data_preparation/)

- Laden von CSV-Daten (absolute Preise)
- Berechnung von Log-Returns: r(t) = ln(P(t) / P(t-1))
- Train-Val-Test Split: 65%-15%-20% (Goodfellow et al., 2016)
- Sequenzerzeugung mit konfigurierbarem Lookback Window
- Inverse Transform: Rücktransformation von Returns zu Preisen

```python
from data_preparation import DataPreparator, create_sequences

preparator = DataPreparator('stockData/preprocessedData/SP500_historical_data.csv')
train, val, test = preparator.load_and_prepare()
X_train, y_train = create_sequences(train, lookback=30)
```

## Parameter-Tuning (parameter_tuning/)

Full Grid Search über alle Kombinationen:
- LSTM: 48 Configs (dropout × dense_units × lr × batch × epochs)
- CNN: 48 Configs (kernel × pool × dropout × batch × epochs)
- GRU: 48 Configs (dropout × dense_units × lr × batch × epochs)
- Informer: 64 Configs (d_model × n_heads × dropout × lr × batch × epochs)
- Resume-Support für unterbrochene Durchläufe
- Ergebnisse in `parameter_tuning/results/{MODEL}/{INDEX}/`

## Lookback-Window Sweep (lookback_window_sweep/)

- Evaluiert L=1 bis L=60 für alle 4 Modelle
- Nutzt per-Index beste Konfiguration aus `parameter_tuning/results/best_configurations.json`
- Erzeugt Plots (MAE vs L, Predictions, Scatter) und JSON pro Index
- Ergebnisse in `lookback_window_sweep/results/{INDEX}/`

## Vergleich & Post-Processing (comparison/)

Trainiert die 32 besten Modelle (8 Indizes × 4 Modelle) mit jeweils bester Config und bestem L:
- **Directional Accuracy (DA%)**: Anteil korrekt vorhergesagter Kursrichtungen
- **Naive Baseline MAE**: MAE wenn immer 0 vorhergesagt wird = mean(|y_actual|)
- **MAE Improvement**: (baseline_mae - model_mae) / baseline_mae × 100
- **R²**: Bestimmtheitsmaß
- Cross-Index Heatmaps (DA%, MAE Improvement, R²)
- Ergebnisse in `comparison/results/`

## Datensätze

8 Indizes in `stockData/preprocessedData/`:

| Index | Datei |
|-------|-------|
| S&P 500 | SP500_historical_data.csv |
| NASDAQ | NASDAQ_historical_data.csv |
| DAX | DAX_historical_data.csv |
| FTSE 100 | FTSE100_historical_data.csv |
| Nikkei 225 | NIKKEI_historical_data.csv |
| Hang Seng | HANG_SENG_historical_data.csv |
| 10Y Bond | 10-Year Bond_historical_data.csv |
| 30Y Bond | 30 Year Bond_historical_data.csv |

## Referenzen

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8).
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *arXiv:1406.1078*.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553).
- Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021* (Best Paper).
