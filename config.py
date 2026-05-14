"""
Zentrale Konfiguration für die Masterarbeit-Pipeline.

Enthält:
- INDICES: Mapping Index-Name -> CSV-Dateiname
- ALL_INDICES: Liste der Indizes
- Modell-Architektur-Konstanten (LSTM_UNITS, GRU_UNITS, CNN-Filter, Informer-Layer)
- LOOKBACK_WINDOW (Default für Full-Grid-Tuning)

Alle Submodule (parameter_tuning, comparison, quick_verify)
importieren von hier — verhindert Drift zwischen Kopien.
"""

INDICES = {
    'SP500':     'SP500_historical_data.csv',
    'DAX':       'DAX_historical_data.csv',
    'NASDAQ':    'NASDAQ_historical_data.csv',
    'HANG_SENG': 'HANG_SENG_historical_data.csv',
    'NIKKEI':    'NIKKEI_historical_data.csv',
    '10Y_Bond':  '10-Year Bond_historical_data.csv',
}

ALL_INDICES = list(INDICES.keys())

# Default lookback window (wird für Grid-Tuning genutzt; Sweep variiert L=1..60)
LOOKBACK_WINDOW = 60

# LSTM / GRU
LSTM_UNITS = [128, 64]
GRU_UNITS = [128, 64]

# CNN
CONV_FILTERS = [64, 128, 256]
DENSE_UNITS_CNN = 64
LEARNING_RATE_CNN = 0.001

# Informer (feste Architektur-Parameter, nicht getunt)
INFORMER_E_LAYERS = 2
INFORMER_D_LAYERS = 1
INFORMER_FACTOR = 5

# Random Seed (einheitlich für Tuning + Sweep + Reproduzierbarkeit)
RANDOM_SEED = 42


def apply_determinism(set_tf=True, set_torch=True):
    """Setzt deterministische Flags für TF/PyTorch.

    Ruft *nur* die Flags, die ohne Import-Seiteneffekte sicher sind.
    Seeds werden separat gesetzt (siehe set_all_seeds).
    """
    import os
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    if set_tf:
        try:
            import tensorflow as tf
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception:
            pass

    if set_torch:
        try:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        except Exception:
            pass


def set_all_seeds(seed=None):
    """Setzt Seeds für random, numpy, tensorflow, torch."""
    import os, random
    s = seed if seed is not None else RANDOM_SEED
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    try:
        import numpy as np
        np.random.seed(s)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(s)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass
