# Stock Price Prediction Models

Implementierung von LSTM, CNN und GRU Modellen zur Vorhersage von Aktienkursen basierend auf historischen Daten.

## Quick Start

```bash
# LSTM trainieren
python lstm_model.py

# CNN trainieren
python cnn_model.py

# GRU trainieren
python gru_model.py

# Alle drei vergleichen
python model_comparison.py

# Hyperparameter-Tuning (Grid Search)
python hyperparameter_tuning.py

# Lookback-Window-Sweep (L=1..60)
python evaluate_lookback_window_sweep.py
```

## Projektstruktur

```
models/
├── data_preparation.py             ← Data loading & 65%-15%-20% Split (Goodfellow)
├── lstm_model.py                   ← LSTM-Implementierung (128→64 Units)
├── cnn_model.py                    ← CNN-Implementierung (64→128→256 Filter)
├── gru_model.py                    ← GRU-Implementierung (128→64 Units)
├── model_comparison.py             ← Training & Vergleich aller drei Modelle
├── hyperparameter_tuning.py        ← Full Grid Search (CNN: 48, LSTM: 72, GRU: 72 Configs)
├── evaluate_lookback_window_sweep.py ← Lookback-Window-Sweep L=1..60
└── README.md                       ← Dieses Dokument

results/
├── tuning/
│   ├── CNN/{INDEX}/                ← CNN Tuning-Ergebnisse (48 Configs pro Index)
│   ├── LSTM/{INDEX}/               ← LSTM Tuning-Ergebnisse (72 Configs pro Index)
│   └── GRU/{INDEX}/                ← GRU Tuning-Ergebnisse (72 Configs pro Index)
├── lookback_sweep/{INDEX}/         ← Window-Sweep-Analyse pro Index
│   ├── 01_mae_vs_lookback_window.png
│   ├── 02_{MODEL}_predictions_vs_actual_timeseries.png
│   ├── 03_{MODEL}_predictions_vs_actual_original_scale.png
│   ├── 04_{MODEL}_scatter_predictions.png
│   └── lookback_evaluation_results.json
├── best_configurations.json        ← Beste Konfiguration pro Index
└── cross_index_comparison.json     ← Cross-Index Vergleich
```

## Überblick

Dieses Projekt implementiert drei Deep Learning Modelle:
- **LSTM** (Long Short-Term Memory): Zur Erfassung von zeitlichen Abhängigkeiten
- **CNN** (Convolutional Neural Network): Zur räumlichen Merkmalserkennung in Zeitreihen
- **GRU** (Gated Recurrent Unit): Leichtgewichtige Alternative zu LSTM

### Data Split (Goodfellow et al., 2016)

Die Datensätze werden nach dem Standard aus "Deep Learning" (Goodfellow, Bengio, Courville, 2016) aufgeteilt:

| Set | Anteil | Zweck |
|-----|--------|-------|
| Training | 65% | Modelltraining |
| Validation | 15% | Hyperparameter-Optimierung & Early Stopping |
| Test | 20% | Unabhängige Bewertung |

### Lookback-Window (Sequenzen)

Alle Modelle verwenden eine **Lookback-Window** Strategie zur Sequenzerzeugung:

```
L = 60 (Lookback-Window Länge, konfigurierbar)

Input:  X[i:i+L] = Sequenz von L aufeinanderfolgenden Zeitpunkten
Output: y[i+L]   = Zielwert zum Zeitpunkt (i+L)

Feature pro Zeitpunkt: [Close-Return]
```

Dies erzeugt Sequenzen der Form:
- **X shape**: (n_sequences, L, 1) - L Zeitschritte × 1 Feature
- **y shape**: (n_sequences, 1) - Vorhersage des Close-Returns

## LSTM-Modell

### Architektur

```
Input Layer
    ↓
LSTM(128) [return_sequences=True]
    ↓
Dropout(0.2)
    ↓
LSTM(64) [return_sequences=False]
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='linear') [Output: Close-Return]
```

### Hyperparameter

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Lookback Window | 60 | Ca. 3 Monate Trading Days |
| LSTM Units | 128, 64 | Progressiv abnehmende Komplexität |
| Dropout Rate | 0.2 | Regularisierung zur Vermeidung von Overfitting |
| Learning Rate | 0.001 | Standard für Adam Optimizer |
| Batch Size | 32 | Balance zwischen Stabilität und Effizienz |
| Epochs | 100 | Mit Early Stopping bei Übertraining |

## GRU-Modell

### Architektur

```
Input Layer
    ↓
GRU(128) [return_sequences=True]
    ↓
Dropout(0.2)
    ↓
GRU(64) [return_sequences=False]
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='linear') [Output: Close-Return]
```

Gleiche Hyperparameter wie LSTM für faire Vergleichbarkeit.

## CNN-Modell

### Architektur

```
Input Layer (L, 1)
    ↓
Conv1D(64, kernel=5, padding='same') + ReLU
    ↓
MaxPooling1D(2)
    ↓
Dropout(0.2)
    ↓
Conv1D(128, kernel=5, padding='same') + ReLU
    ↓
MaxPooling1D(2)
    ↓
Dropout(0.2)
    ↓
Conv1D(256, kernel=5, padding='same') + ReLU
    ↓
Dropout(0.2)
    ↓
Flatten
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dense(1, activation='linear') [Output: Close-Return]
```

### Hyperparameter

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Lookback Window | 60 | Gleich wie LSTM/GRU für faire Vergleichbarkeit |
| Conv Filter | 64, 128, 256 | Progressive Tiefe (Feature-Hierarchie) |
| Kernel Size | 5 | Größeres Fenster für Mustererkennung |
| Pool Size | 2 | Dimensionsreduktion und Abstraktionen |
| Dropout Rate | 0.2 | Regularisierung |
| Learning Rate | 0.001 | Standard für Adam Optimizer |
| Batch Size | 32 | Konsistent mit LSTM/GRU |
| Epochs | 100 | Mit Early Stopping |

## Module

### `data_preparation.py`

**DataPreparator Klasse:**
- Laden von CSV-Daten
- Berechnung prozentualer Returns
- 65%-15%-20% Split nach Goodfellow

**create_sequences() Funktion:**
- Erzeugt Lookback-Window Sequenzen
- Input: Raw Data + Lookback Length
- Output: (X, y) Paare für Training

Beispiel:
```python
preparator = DataPreparator('data.csv')
train, val, test = preparator.load_and_prepare()
X_train, y_train = create_sequences(train, lookback=30)
```

### `lstm_model.py` / `gru_model.py`

**LSTMModel / GRUModel Klasse:**

Konfigurierbare Parameter:
```python
LOOKBACK_WINDOW = 60
LSTM_UNITS = [128, 64]  # bzw. GRU_UNITS
DROPOUT_RATE = 0.2
DENSE_UNITS = 32
LEARNING_RATE = 0.001
```

Hauptmethoden:
- `prepare_data()`: Laden und Sequenzerzeugung
- `build_model()`: Modellarchitektur
- `train()`: Trainieren mit Callbacks (EarlyStopping, ReduceLROnPlateau)
- `evaluate()`: Test-Performance
- `predict()`: Vorhersagen
- `plot_results()`: Visualisierung
- `save_model()`: Persistierung

### `cnn_model.py`

**CNNModel Klasse:**

Konfigurierbare Parameter:
```python
LOOKBACK_WINDOW = 60
CONV_FILTERS = [64, 128, 256]
KERNEL_SIZE = 5
POOL_SIZE = 2
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
```

Selbes Interface wie LSTM/GRU für Vergleichbarkeit.

### `model_comparison.py`

**Vergleichs-Script:**
- Führt LSTM, CNN und GRU nacheinander aus
- Misst Trainingszeit
- Erstellt JSON-Report

### `hyperparameter_tuning.py`

**FullGridSearchTuner Klasse:**
- Vollständiger Grid Search über alle Hyperparameter-Kombinationen
- CNN: 48 Konfigurationen (kernel × pool × dropout × batch × epochs)
- LSTM: 72 Konfigurationen (dropout × dense_units × lr × batch × epochs)
- GRU: 72 Konfigurationen (dropout × dense_units × lr × batch × epochs)
- Resume-Support für unterbrochene Durchläufe
- Ergebnisse in `results/tuning/{MODEL}/{INDEX}/`

### `evaluate_lookback_window_sweep.py`

**Lookback-Window Sweep:**
- Evaluiert Lookback-Windows von L=1 bis L=60
- Nutzt per-Index beste Konfiguration aus `best_configurations.json`
- Erzeugt Plots und JSON-Ergebnisse pro Index in `results/lookback_sweep/{INDEX}/`

## Callbacks

Alle Modelle verwenden:

1. **EarlyStopping**: Stoppt Training wenn Val Loss nicht mehr sinkt
   - Monitor: val_loss
   - Patience: 10 Epochen
   - Restore best weights

2. **ReduceLROnPlateau**: Reduziert Learning Rate bei Plateau
   - Factor: 0.5
   - Patience: 5 Epochen
   - Min LR: 1e-6

## Datenfluss

```
preprocessedData/*.csv
        ↓
  [Data Preparation]
        ↓
    +--------+--------+--------+
    |        |        |        |
   65%      15%      20%
  Train    Val      Test
    |        |        |
    +--------+--------+
           ↓
    [Sequences L=1..60]
           ↓
    +--LSTM--+--GRU--+
    |        |       |
    +--CNN---+-------+
           ↓
      [Predictions]
           ↓
    [Original Scale]
```

## Referenzen

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *arXiv:1406.1078*.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
