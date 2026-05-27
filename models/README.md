# Modell-Implementierungen

Vier Deep Learning Modelle zur Vorhersage von Log-Returns auf Basis historischer Kursdaten.

## Dateien

| Datei | Framework | Beschreibung |
|-------|-----------|-------------|
| `base_keras_model.py` | TensorFlow | `BaseKerasTimeSeriesModel` — gemeinsame Pipeline-Methoden für LSTM/GRU/CNN |
| `lstm_model.py` | TensorFlow | `LSTMModel(BaseKerasTimeSeriesModel)` — 2 Recurrent-Layer aus `config.LSTM_UNITS` |
| `cnn_model.py` | TensorFlow | `CNNModel(BaseKerasTimeSeriesModel)` — 3 Conv1D-Layer aus `config.CONV_FILTERS` |
| `gru_model.py` | TensorFlow | `GRUModel(BaseKerasTimeSeriesModel)` — 2 Recurrent-Layer aus `config.GRU_UNITS` |
| `informer_model.py` | PyTorch | Informer mit ProbSparse Self-Attention (eigene Klasse, kein Erbe) |

Die drei Keras-Subklassen überschreiben nur `build_model()`. `prepare_data`, `train`,
`evaluate`, `predict`, `plot_results`, `save_model`, `run_full_pipeline` sind alle
in `BaseKerasTimeSeriesModel` einmalig implementiert. Standalone-Aufruf: jede Datei
kann mit `python models/<datei>.py` direkt ausgeführt werden.

Architektur-Konstanten (`LSTM_UNITS`, `GRU_UNITS`, `CONV_FILTERS`, `DENSE_UNITS_CNN`,
`LEARNING_RATE_CNN`, `LOOKBACK_WINDOW`) kommen aus [`config.py`](../config.py) — kein
Drift-Risiko mehr zwischen Pipeline und Standalone-Pfad.

Modell-Speicherformat: `model.keras` (Keras 3 Standard). Phase 3 hat einen bilateralen
Loader — alte `.h5`-Modelle aus früheren Pipeline-Läufen werden weiterhin akzeptiert.

## LSTM-Architektur

```
Input (L, 1)
    -> LSTM(128, return_sequences=True) -> Dropout
    -> LSTM(64) -> Dropout
    -> Dense(32, relu) -> Dropout
    -> Dense(1, linear)
```

| Parameter | Default | Tuning-Bereich |
|-----------|---------|----------------|
| Lookback Window | 60 | Coarse: {1, 10, 20, 30, 40, 50, 60} |
| LSTM Units | 128, 64 | fest |
| Dropout | 0.2 | {0.2, 0.4} |
| Dense Units | 32 | {16, 32} |
| Learning Rate | 0.001 | {0.001, 0.005} |
| Batch Size | 32 | {8, 32} |
| Epochs | 100 | {50, 100} |

## GRU-Architektur

```
Input (L, 1)
    -> GRU(128, return_sequences=True) -> Dropout
    -> GRU(64) -> Dropout
    -> Dense(32, relu) -> Dropout
    -> Dense(1, linear)
```

Gleicher Tuning-Bereich wie LSTM für faire Vergleichbarkeit.

## CNN-Architektur

```
Input (L, 1)
    -> Conv1D(64, kernel, padding='same') -> MaxPool -> Dropout
    -> Conv1D(128, kernel, padding='same') -> MaxPool -> Dropout
    -> Conv1D(256, kernel, padding='same') -> Dropout
    -> Flatten -> Dense(64) -> Dropout -> Dense(32) -> Dense(1)
```

| Parameter | Default | Tuning-Bereich |
|-----------|---------|----------------|
| Lookback Window | 60 | Coarse: {1, 10, 20, 30, 40, 50, 60} |
| Conv Filter | 64, 128, 256 | fest |
| Kernel Size | 5 | {3, 5} |
| Pool Size | 2 | {2, 4} |
| Dropout | 0.2 | {0.2, 0.4} |
| Batch Size | 32 | {8, 32} |
| Epochs | 100 | {50, 100} |

## Informer-Architektur (PyTorch)

```
Input (L, n_features)
    -> Data Embedding (Linear + Positional Encoding)
    -> Encoder Layer 1 (ProbSparse Attention + FFN + LayerNorm)
    -> Distilling (Conv1D + MaxPool: L -> L/2)
    -> Encoder Layer 2 (ProbSparse Attention + FFN + LayerNorm)
    -> Decoder (Self-Attention + Cross-Attention + FFN)
    -> Linear Projection -> (1,)
```

### ProbSparse Self-Attention

Kern-Innovation des Informers (Zhou et al., 2021):
- Misst die "Sparsity" jeder Query via KL-Divergenz zur Gleichverteilung
- Wählt nur die Top-u (u = c * ln(L)) aktivsten Queries aus
- Komplexität: O(L log L) statt O(L²) bei Standard-Attention

| Parameter | Default | Tuning-Bereich |
|-----------|---------|----------------|
| Lookback Window | 60 | Coarse: {1, 10, 20, 30, 40, 50, 60} |
| d_model | 64 | {32, 64} |
| n_heads | 8 | {4, 8} |
| e_layers | 2 | fest |
| d_layers | 1 | fest |
| d_ff | 256 | 4 * d_model |
| Dropout | 0.05 | {0.05, 0.1} |
| Learning Rate | 0.0001 | {0.0001, 0.001} |
| Batch Size | 32 | {8, 32} |
| Epochs | 100 | {50, 100} |
| factor | 5 | fest |

## Helper-Funktionen (informer_model.py)

Für die Nutzung in der Pipeline exportiert `informer_model.py` zusätzliche Funktionen:

```python
from models.informer_model import build_informer, train_informer, evaluate_informer, predict_informer

model = build_informer(lookback_window, n_features, config_dict)
train_informer(model, X_train, y_train, config_dict, lookback_window,
               X_val=X_val, y_val=y_val)   # X_val/y_val optional für Per-Epoche Val-Loss
mse, mae = evaluate_informer(model, X_test, y_test, lookback_window)
mse, mae, mare = evaluate_informer(model, X_test, y_test, lookback_window, return_mare=True)
predictions = predict_informer(model, X_test, lookback_window)
```

### P1-8-Fix: `label_len = max(1, L // 2)`

Bei sehr kleinen L (z.B. L=1) hatte der Decoder vorher `label_len = 0` und damit nur Null-Padding als Eingabe — die Architektur degenerierte. Mit `max(1, ...)` bekommt der Decoder jederzeit mindestens ein reales Token.

## Einzelnes Modell ausführen

```bash
cd models
python lstm_model.py
python cnn_model.py
python gru_model.py
python informer_model.py
```

Jedes Modell kann auch direkt instanziiert werden:

```python
from models.lstm_model import LSTMModel

model = LSTMModel('stockData/preprocessedData/SP500_historical_data.csv',
                   output_dir='parameter_tuning/results/LSTM/SP500')
model.run_full_pipeline()
```

## Gemeinsame Eigenschaften

- **Input**: Log-Returns r(t) = ln(P(t) / P(t-1))
- **Target**: Close-Log-Return (Single-Output)
- **Optimizer**: Adam
- **Loss**: MSE (Mean Squared Error) auf Log-Returns
- **Reporting-Metrik**: MAPE (Mean Absolute Percentage Error) **auf Preisen** nach Inverse-Transform — `MAPE = 100 · mean(|p_pred - p_true| / (|p_true| + ε))`. Intern (`results.json`) wird der Rohwert als `mare` in Dezimalform abgelegt; `MAPE = mare · 100`
- **Training**: Feste Epochenzahl (kein EarlyStopping)
- **Data Split**: 65%-15%-20% (Goodfellow et al., 2016), chronologisch (kein Shuffle)
- **Inverse Transform**: Rücktransformation von Returns zu Preisen via `DataPreparator`
  `P(t) = base_price(t-1) * exp(predicted_log_return(t))`
- **Reproduzierbarkeit**: Seeds für `random`, `numpy`, `tf.random`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all`, `PYTHONHASHSEED` werden in jedem Subprozess gesetzt
