# Modell-Implementierungen

Vier Deep Learning Modelle zur Vorhersage von Log-Returns auf Basis historischer Kursdaten.

## Dateien

| Datei | Framework | Beschreibung |
|-------|-----------|-------------|
| `lstm_model.py` | TensorFlow | LSTM mit 2 Recurrent-Layern (128, 64 Units) |
| `cnn_model.py` | TensorFlow | CNN mit 3 Conv1D-Layern (64, 128, 256 Filter) |
| `gru_model.py` | TensorFlow | GRU mit 2 Recurrent-Layern (128, 64 Units) |
| `informer_model.py` | PyTorch | Informer mit ProbSparse Self-Attention |

Alle Modelle implementieren dasselbe Interface: `prepare_data()`, `build_model()`, `train()`, `evaluate()`, `predict()`, `plot_results()`, `save_model()`, `run_full_pipeline()`.

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
| Lookback Window | 60 | 1..60 (Sweep) |
| LSTM Units | 128, 64 | fest |
| Dropout | 0.2 | {0.2, 0.4} |
| Dense Units | 32 | {16, 32} |
| Learning Rate | 0.001 | {0.001, 0.005} |
| Batch Size | 32 | {8, 16, 32} |
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
| Lookback Window | 60 | 1..60 (Sweep) |
| Conv Filter | 64, 128, 256 | fest |
| Kernel Size | 5 | {3, 5} |
| Pool Size | 2 | {2, 4} |
| Dropout | 0.2 | {0.2, 0.4} |
| Batch Size | 32 | {8, 16, 32} |
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
| Lookback Window | 60 | 1..60 (Sweep) |
| d_model | 64 | {32, 64} |
| n_heads | 8 | {4, 8} |
| e_layers | 2 | fest |
| d_layers | 1 | fest |
| d_ff | 256 | 4 * d_model |
| Dropout | 0.05 | {0.05, 0.1} |
| Learning Rate | 0.0001 | {0.0001, 0.001} |
| Batch Size | 32 | {16, 32} |
| Epochs | 100 | {50, 100} |
| factor | 5 | fest |

## Helper-Funktionen (informer_model.py)

Für die Nutzung in Tuning und Sweep exportiert `informer_model.py` zusätzliche Funktionen:

```python
from models.informer_model import build_informer, train_informer, evaluate_informer, predict_informer

model = build_informer(lookback_window, n_features, config_dict)
train_informer(model, X_train, y_train, config_dict, lookback_window)
mse, mae = evaluate_informer(model, X_test, y_test, lookback_window)
predictions = predict_informer(model, X_test, lookback_window)
```

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
- **Loss**: MSE (Mean Squared Error)
- **Training**: Feste Epochenzahl (kein EarlyStopping)
- **Data Split**: 65%-15%-20% (Goodfellow et al., 2016)
- **Inverse Transform**: Rücktransformation von Returns zu Preisen via `DataPreparator`
