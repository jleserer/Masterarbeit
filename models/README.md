# Stock Price Prediction Models

Implementierung von LSTM und CNN Modellen zur Vorhersage von Aktienkursen basierend auf historischen Daten.

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
L = 30 (Loopback-Window Länge)

Input:  X[i:i+L] = Sequenz von L aufeinanderfolgenden Zeitpunkten
Output: y[i+L]   = Zielwert zum Zeitpunkt (i+L)

Features pro Zeitpunkt: [Close, High, Low]
```

Dies erzeugt Sequenzen der Form:
- **X shape**: (n_sequences, 30, 3) - 30 Zeitschritte × 3 Features
- **y shape**: (n_sequences, 3) - Vorhersage der 3 Features

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
Dense(3, activation='linear') [Output: Close, High, Low]
```

### Hyperparameter

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Lookback Window | 30 | 1 Monat Trading (ca. 20 Business Days) |
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
Input Layer (30, 3)
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
Dense(3, activation='linear') [Output: Close, High, Low]
```

### Hyperparameter

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| Lookback Window | 30 | Gleich wie LSTM für faire Vergleichbarkeit |
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
LOOKBACK_WINDOW = 30
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
LOOKBACK_WINDOW = 30
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
- `lstm_results_SP500/`: LSTM-Modell und Plots
- `cnn_results_SP500/`: CNN-Modell und Plots
- `model_comparison_results.json`: Vergleichsergebnisse

## Ausgaben

Jedes Modell erzeugt:
- `*_model.h5`: Trainiertes Modell (Keras format)
- `*_training_history.png`: Loss und MAE Plots
- Vorhersagen (normalisiert und original)

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

Da das Zielvektor multi-dimensional ist (Close, High, Low), wird für alle Komponenten die Inverse Transform angewendet:

```python
predictions_original = preparator.inverse_transform(predictions_normalized)
```

Dies konvertiert die normalisierten Vorhersagen zurück in die ursprüngliche Preiseskala.

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

## Referenzen

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

## Zukünftige Verbesserungen

- [ ] GridSearch für Hyperparameter
- [ ] Ensemble-Modelle (Kombination LSTM+CNN)
- [ ] Attention-Mechanismen
- [ ] Multi-Step Ahead Vorhersagen
- [ ] Cross-Validation über mehrere Stocks
