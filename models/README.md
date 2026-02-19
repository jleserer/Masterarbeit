# Stock Price Prediction Models

Implementierung von LSTM und CNN Modellen zur Vorhersage von Aktienkursen basierend auf historischen Daten.

## Quick Start

```bash
# 1. Alles testen
python test_models.py

# 2. LSTM trainieren
python lstm_model.py

# 3. CNN trainieren
python cnn_model.py

# 4. Beide vergleichen
python model_comparison.py

# 5. Hyperparameter-Tuning (Grid Search)
python hyperparameter_tuning.py

# 6. Beste Modelle evaluieren
python evaluate_best_models.py

# 7. Lookback-Window-Sweep (L=1..60)
python evaluate_lookback_window_sweep.py
```

## Projektstruktur

```
models/
├── data_preparation.py             ← Data loading & 65%-15%-20% Split (Goodfellow)
├── lstm_model.py                   ← LSTM-Implementierung (128→64 Units)
├── cnn_model.py                    ← CNN-Implementierung (64→128→256 Filter)
├── model_comparison.py             ← Training & Vergleich beider Modelle
├── hyperparameter_tuning.py        ← Full Grid Search (CNN: 48, LSTM: 72 Configs)
├── evaluate_best_models.py         ← Beste Modelle auf Test-Set evaluieren
├── evaluate_lookback_window_sweep.py ← Lookback-Window-Sweep L=1..60
├── test_models.py                  ← Schnelltest aller Module
├── test_suite.py                   ← Umfassende Unit-Test-Suite (28 Tests)
└── README.md                       ← Dieses Dokument

results/
├── CNN/SP500/                      ← CNN Tuning-Ergebnisse (48 Experimente) + Modell
├── LSTM/SP500/                     ← LSTM Tuning-Ergebnisse (72 Experimente) + Modell
├── evaluation/SP500/               ← Modellvergleich & Evaluierungs-Plots
└── lookback_sweep/lookback_evaluation_sweep/  ← Window-Sweep-Analyse
    ├── 01_mae_vs_lookback_window.png
    ├── 02/03/04_LSTM_*.png
    ├── 02/03/04_CNN_*.png
    └── lookback_evaluation_results.json
```

## Überblick

Dieses Projekt implementiert zwei Deep Learning Modelle:
- **LSTM** (Long Short-Term Memory): Zur Erfassung von zeitlichen Abhängigkeiten
- **CNN** (Convolutional Neural Network): Zur räumlichen Merkmalserkennung in Zeitreihen

### Data Split (Goodfellow et al., 2016)

Die Datensätze werden nach dem Standard aus "Deep Learning" (Goodfellow, Bengio, Courville, 2016) aufgeteilt:

| Set | Anteil | Zweck |
|-----|--------|-------|
| Training | 65% | Modelltraining |
| Validation | 15% | Hyperparameter-Optimierung & Early Stopping |
| Test | 20% | Unabhängige Bewertung |

### Loopback-Window (Sequenzen)

Beide Modelle verwenden eine **Loopback-Window** Strategie zur Sequenzerzeugung:

```
L = 60 (Loopback-Window Länge)

Input:  X[i:i+L] = Sequenz von L aufeinanderfolgenden Zeitpunkten
Output: y[i+L]   = Zielwert zum Zeitpunkt (i+L)

Feature pro Zeitpunkt: [Close]
```

Dies erzeugt Sequenzen der Form:
- **X shape**: (n_sequences, 60, 1) - 60 Zeitschritte × 1 Feature
- **y shape**: (n_sequences, 1) - Vorhersage des Close-Preises

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
Dense(1, activation='linear') [Output: Close]
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

### Rationale

- **2 LSTM-Layer**: Erfasst mehrstufige zeitliche Muster
- **Dropout 0.2**: Verhindert Overfitting ohne zu aggressiv zu sein
- **Return Sequences=True bei Schicht 1**: Ermöglicht Durchgang an nächste LSTM-Schicht
- **Linear Aktivierung Output**: Für kontinuierliche Vorhersagen

## CNN-Modell

### Architektur

```
Input Layer (60, 1)
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
Dense(1, activation='linear') [Output: Close]
```

### Hyperparameter

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Lookback Window | 60 | Gleich wie LSTM für faire Vergleichbarkeit |
| Conv Filter | 64, 128, 256 | Progressive Tiefe (Feature-Hierarchie) |
| Kernel Size | 5 | Größeres Fenster für Mustererkennung |
| Pool Size | 2 | Dimensionsreduktion und Abstraktionen |
| Dropout Rate | 0.2 | Regularisierung |
| Learning Rate | 0.001 | Standard für Adam Optimizer |
| Batch Size | 32 | Konsistent mit LSTM |
| Epochs | 100 | Mit Early Stopping |

### Rationale

- **3 Conv-Layer**: Progressive Feature-Hierarchie
- **Progressive Filter**: 64 → 128 → 256 (tiefere Abstraktionen)
- **Kernel Size 5**: Erfasst lokale zeitliche Muster
- **Max Pooling**: Dimensionsreduktion und Hervorhebung wichtiger Features
- **Padding='same'**: Erhält zeitliche Auflösung

## Module

### `data_preparation.py`

**DataPreparator Klasse:**
- Laden von CSV-Daten
- Normalisierung auf [0, 1] mit MinMaxScaler
- 65%-15%-20% Split nach Goodfellow
- Inverse Transform für Ausgaben

**create_sequences() Funktion:**
- Erzeugt Loopback-Window Sequenzen
- Input: Raw Data + Lookback Length
- Output: (X, y) Paare für Training

Beispiel:
```python
preparator = DataPreparator('data.csv')
train, val, test = preparator.load_and_prepare()
X_train, y_train = create_sequences(train, lookback=30)
```

### `lstm_model.py`

**LSTMModel Klasse:**

Konfigurierbare Parameter:
```python
LOOKBACK_WINDOW = 60
LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.2
DENSE_UNITS = 32
LEARNING_RATE = 0.001
```

Hauptmethoden:
- `prepare_data()`: Laden und Sequenzerzeugung
- `build_model()`: Modellarchitektur
- `train()`: Trainieren mit Callbacks (EarlyStopping, ReduceLROnPlateau)
- `evaluate()`: Test-Performance
- `predict()`: Vorhersagen (normalisiert und original)
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

Selbe Interface wie LSTM für Vergleichbarkeit.

### `model_comparison.py`

**Vergleichs-Script:**
- Führt LSTM und CNN nacheinander aus
- Misst Trainingszeit
- Erstellt JSON-Report
- Speichert Ergebnisse in separaten Ordnern

### `hyperparameter_tuning.py`

**FullGridSearchTuner Klasse:**
- Vollständiger Grid Search über alle Hyperparameter-Kombinationen
- CNN: 48 Konfigurationen (kernel × pool × dropout × batch × epochs)
- LSTM: 72 Konfigurationen (dropout × dense_units × lr × batch × epochs)
- Resume-Support für unterbrochene Durchläufe
- Ergebnisse in `results/CNN/SP500/` und `results/LSTM/SP500/`

### `evaluate_best_models.py`

**Evaluierungs-Script:**
- Lädt die besten CNN- und LSTM-Konfigurationen
- Evaluiert auf dem Test-Set
- Erzeugt Vergleichs-Visualisierungen in `results/evaluation/SP500/`

### `evaluate_lookback_window.py`

- Evaluiert verschiedene Lookback-Window-Größen
- Vergleicht Modellperformance bei unterschiedlichen Zeitfenstern

## Verwendung

### Einzelnes Modell trainieren

```bash
# LSTM
python lstm_model.py

# CNN
python cnn_model.py
```

### Beide Modelle vergleichen

```bash
python model_comparison.py
```

Dies erzeugt:
- `results/LSTM/SP500/`: LSTM-Modell und Plots
- `results/CNN/SP500/`: CNN-Modell und Plots
- `results/evaluation/SP500/model_comparison_results.json`: Vergleichsergebnisse

### Hyperparameter-Tuning (Grid Search)

```bash
python hyperparameter_tuning.py
```

Dies führt einen vollständigen Grid Search durch und speichert alle Experiment-Ergebnisse.

### Beste Modelle evaluieren

```bash
python evaluate_best_models.py
```

Erzeugt Evaluierungs-Plots in `results/evaluation/SP500/`.

## Ausgaben

### results/CNN/SP500/
```
├── all_results.json              ← Alle Grid-Search-Ergebnisse
├── cnn_model.h5                  ← Trainiertes Modell
├── cnn_training_history.png      ← Loss/MAE Plots
└── k*_p*_d*_b*_e*/              ← Einzelne Experiment-Ergebnisse
```

### results/LSTM/SP500/
```
├── all_results.json              ← Alle Grid-Search-Ergebnisse
├── lstm_model.h5                 ← Trainiertes Modell
├── lstm_training_history.png     ← Loss/MAE Plots
└── d*_u*_lr*_b*_e*/             ← Einzelne Experiment-Ergebnisse
```

### results/evaluation/SP500/
```
├── model_comparison.png          ← Side-by-Side Modellvergleich
├── model_comparison_zoomed.png   ← Gezoomter Vergleich
├── model_comparison_results.json ← Performance-Vergleich
├── predictions_vs_actual.png     ← Vorhersage-Plots
└── training_history.png          ← Trainings-Verlauf Vergleich
```

## Hyperparameter-Anpassung

Zur Experimentation mit verschiedenen Architekturen:

### LSTM Variationen
```python
# Tieferes Modell
LSTM_UNITS = [256, 128, 64]

# Weniger Dropout
DROPOUT_RATE = 0.1

# Höhere Learning Rate
LEARNING_RATE = 0.005
```

### CNN Variationen
```python
# Mehr Filter
CONV_FILTERS = [128, 256, 512]

# Kleinerer Kernel
KERNEL_SIZE = 3

# Mehr Dense Layer
# (füge Dense Layer in der Klasse hinzu)
```

## Inverse Transform

Die normalisierten Vorhersagen werden mittels Inverse Transform zurück in die ursprüngliche Preiseskala konvertiert:

```python
predictions_original = preparator.inverse_transform(predictions_normalized)
```

## Callbacks

Beide Modelle verwenden:

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
    [Sequences L=60]
           ↓
    +--LSTM--+
    |        |
    +--CNN---+
           ↓
      [Predictions]
           ↓
    [Inverse Transform]
           ↓
    [Original Price Scale]
```

## Test-Ergebnisse

**Datum**: 2025-12-15 | **Status**: ALLE 28/28 TESTS BESTANDEN

| Kategorie | Tests | Status |
|-----------|-------|--------|
| Data Preparation | 6/6 | ✅ |
| Sequence Creation | 4/4 | ✅ |
| LSTM Model | 7/7 | ✅ |
| CNN Model | 7/7 | ✅ |
| Model Comparison | 4/4 | ✅ |

**Verifizierte Daten:**

| Set | Samples | Anteil |
|-----|---------|--------|
| Gesamt | 9.562 | 100% |
| Train | 6.215 | 65% |
| Validation | 1.434 | 15% |
| Test | 1.913 | 20% |

```
# Tests ausführen:
python test_suite.py
python -m unittest test_suite.TestLSTMModel
python -m unittest test_suite.TestCNNModel
python -m unittest test_suite.TestDataPreparation
```

## Referenzen

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
