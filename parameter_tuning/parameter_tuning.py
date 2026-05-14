"""
Full Hyperparameter Grid Search for LSTM, CNN, GRU and Informer Models
======================================================================
Tests ALL possible combinations of hyperparameters.
Skips already completed configurations.

batch ∈ {8, 32} für alle Modelle (16 wurde nach Auswertung der ersten Indizes
entfernt — verlor durchgängig in allen drei Keras-Modellen).

LSTM:     2 × 2 × 2 × 2 × 2 = 32 combinations
CNN:      2 × 2 × 2 × 2 × 2 = 32 combinations
GRU:      2 × 2 × 2 × 2 × 2 = 32 combinations
Informer: 2 × 2 × 2 × 2 × 2 × 2 = 64 combinations
Total: 160 combinations
"""

import sys
import os
STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from data_preparation import DataPreparator, create_sequences
from config import (
    INDICES, LOOKBACK_WINDOW, LSTM_UNITS, GRU_UNITS,
    CONV_FILTERS, DENSE_UNITS_CNN, LEARNING_RATE_CNN,
    INFORMER_E_LAYERS, INFORMER_D_LAYERS, INFORMER_FACTOR,
)


# MARE wird hier NICHT mehr als Keras-Metric berechnet, sondern post-training
# auf der Preis-Ebene (via DataPreparator.inverse_transform). Siehe
# FullGridSearchTuner._mare_on_prices(). Grund: Log-Returns schwanken um 0 —
# eine relative Fehler-Metrik ist auf dieser Skala instabil und interpretatorisch
# wenig aussagekräftig. Preis-basierte MARE ist das, was Gutachter erwarten.

# Fixe y-Achse für die Pro-Config-Balken-Plots, damit alle 48-64 PNGs eines
# Modells visuell direkt vergleichbar sind. 0-2 % deckt typische tägliche
# Kursänderungen (SP500 ~0.8 %, DAX ~1 %, Bond ~0.5 %) mit Reserve ab.
MARE_PLOT_YLIM = (0.0, 0.02)


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# LSTM parameters
LSTM_DROPOUT = [0.2, 0.4]
LSTM_DENSE_UNITS = [16, 32]
LSTM_LEARNING_RATE = [0.001, 0.005]
LSTM_BATCH_SIZE = [8, 32]
LSTM_EPOCHS = [50, 100]

# CNN parameters
CNN_KERNEL_SIZE = [3, 5]
CNN_POOL_SIZE = [2, 4]
CNN_DROPOUT = [0.2, 0.4]
CNN_BATCH_SIZE = [8, 32]
CNN_EPOCHS = [50, 100]

# GRU parameters (same search space as LSTM)
GRU_DROPOUT = [0.2, 0.4]
GRU_DENSE_UNITS = [16, 32]
GRU_LEARNING_RATE = [0.001, 0.005]
GRU_BATCH_SIZE = [8, 32]
GRU_EPOCHS = [50, 100]

# Informer parameters
INFORMER_D_MODEL = [32, 64]
INFORMER_N_HEADS = [4, 8]
INFORMER_DROPOUT = [0.05, 0.1]
INFORMER_LEARNING_RATE = [0.0001, 0.001]
INFORMER_BATCH_SIZE = [8, 32]
INFORMER_EPOCHS = [50, 100]

# Model-Architektur-Konstanten, INDICES und LOOKBACK_WINDOW aus config.py (oben importiert)


def generate_lstm_configs():
    """Generate all LSTM configurations."""
    configs = []
    for dropout, dense, lr, batch, epochs in itertools.product(
        LSTM_DROPOUT, LSTM_DENSE_UNITS, LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_EPOCHS
    ):
        name = f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"
        configs.append({
            'name': name,
            'dropout': dropout,
            'dense_units': dense,
            'lr': lr,
            'batch': batch,
            'epochs': epochs
        })
    return configs


def generate_cnn_configs():
    """Generate all CNN configurations."""
    configs = []
    for kernel, pool, dropout, batch, epochs in itertools.product(
        CNN_KERNEL_SIZE, CNN_POOL_SIZE, CNN_DROPOUT, CNN_BATCH_SIZE, CNN_EPOCHS
    ):
        name = f"k{kernel}_p{pool}_d{dropout}_b{batch}_e{epochs}"
        configs.append({
            'name': name,
            'kernel': kernel,
            'pool': pool,
            'dropout': dropout,
            'batch': batch,
            'epochs': epochs
        })
    return configs


def generate_gru_configs():
    """Generate all GRU configurations."""
    configs = []
    for dropout, dense, lr, batch, epochs in itertools.product(
        GRU_DROPOUT, GRU_DENSE_UNITS, GRU_LEARNING_RATE, GRU_BATCH_SIZE, GRU_EPOCHS
    ):
        name = f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"
        configs.append({
            'name': name,
            'dropout': dropout,
            'dense_units': dense,
            'lr': lr,
            'batch': batch,
            'epochs': epochs
        })
    return configs


def generate_informer_configs():
    """Generate all Informer configurations."""
    configs = []
    for d_model, n_heads, dropout, lr, batch, epochs in itertools.product(
        INFORMER_D_MODEL, INFORMER_N_HEADS, INFORMER_DROPOUT,
        INFORMER_LEARNING_RATE, INFORMER_BATCH_SIZE, INFORMER_EPOCHS
    ):
        name = f"dm{d_model}_h{n_heads}_d{dropout}_lr{lr}_b{batch}_e{epochs}"
        configs.append({
            'name': name,
            'd_model': d_model,
            'n_heads': n_heads,
            'dropout': dropout,
            'lr': lr,
            'batch': batch,
            'epochs': epochs
        })
    return configs


class FullGridSearchTuner:
    """Manages full grid search for LSTM, CNN, GRU and Informer models.

    Mit der neuen L-aware Pipeline wird pro L ein eigener Tuner instanziiert
    und die Results landen unter .../<MODEL>/<INDEX>/L<L>/<config>/results.json.
    Wenn `lookback` None ist, wird LOOKBACK_WINDOW aus config.py verwendet
    (legacy / Standalone-Modus).
    """

    def __init__(self, data_path, output_dir=os.path.join(STEP_DIR, 'results'),
                 index_name='SP500', lookback=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.index_name = index_name
        self.lookback = lookback if lookback is not None else LOOKBACK_WINDOW

        # Pro L eigener Unterordner (L<L>). Bei legacy-Modus (kein --lookback) wird
        # ebenfalls L<LOOKBACK_WINDOW> verwendet — konsistent und keine Altlasten.
        l_subdir = f'L{self.lookback:02d}'
        self.lstm_dir = os.path.join(output_dir, 'LSTM', index_name, l_subdir)
        self.cnn_dir = os.path.join(output_dir, 'CNN', index_name, l_subdir)
        self.gru_dir = os.path.join(output_dir, 'GRU', index_name, l_subdir)
        self.informer_dir = os.path.join(output_dir, 'INFORMER', index_name, l_subdir)

        # Data containers
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.preparator = None

        # Results
        self.lstm_results = []
        self.cnn_results = []
        self.gru_results = []
        self.informer_results = []

        os.makedirs(self.lstm_dir, exist_ok=True)
        os.makedirs(self.cnn_dir, exist_ok=True)
        os.makedirs(self.gru_dir, exist_ok=True)
        os.makedirs(self.informer_dir, exist_ok=True)

        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results to skip completed configs."""
        lstm_results_file = os.path.join(self.lstm_dir, 'all_results.json')
        cnn_results_file = os.path.join(self.cnn_dir, 'all_results.json')

        if os.path.exists(lstm_results_file):
            with open(lstm_results_file, 'r') as f:
                self.lstm_results = json.load(f)
            print(f"Loaded {len(self.lstm_results)} existing LSTM results")

        if os.path.exists(cnn_results_file):
            with open(cnn_results_file, 'r') as f:
                self.cnn_results = json.load(f)
            print(f"Loaded {len(self.cnn_results)} existing CNN results")

        gru_results_file = os.path.join(self.gru_dir, 'all_results.json')
        if os.path.exists(gru_results_file):
            with open(gru_results_file, 'r') as f:
                self.gru_results = json.load(f)
            print(f"Loaded {len(self.gru_results)} existing GRU results")

        informer_results_file = os.path.join(self.informer_dir, 'all_results.json')
        if os.path.exists(informer_results_file):
            with open(informer_results_file, 'r') as f:
                self.informer_results = json.load(f)
            print(f"Loaded {len(self.informer_results)} existing Informer results")

    def _get_completed_configs(self, results):
        """Get set of completed config names."""
        return {r['config_name'] for r in results if 'metrics' in r}

    def prepare_data(self):
        """Load and prepare data once for all experiments."""
        print("=" * 70)
        print("PREPARING DATA")
        print("=" * 70)

        self.preparator = DataPreparator(self.data_path)
        train_data, val_data, test_data = self.preparator.load_and_prepare()

        self.X_train, self.y_train = create_sequences(train_data, self.lookback)
        self.X_val, self.y_val = create_sequences(val_data, self.lookback)
        self.X_test, self.y_test = create_sequences(test_data, self.lookback)

        print(f"\nData prepared:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")

    def _mare_on_prices(self, y_true_returns, y_pred_returns, split):
        """MARE auf Preis-Ebene — NICHT auf Log-Returns.

        Transformiert via self.preparator.inverse_transform beide Arrays zurück
        auf echte Preise und berechnet dann MARE = mean(|p_pred - p_true| / |p_true|).

        Das ist die physikalisch sinnvolle relative Fehler-Metrik. Auf Log-Returns
        wäre MARE numerisch instabil (Returns schwanken um null).
        """
        y_true_prices = self.preparator.inverse_transform(
            np.asarray(y_true_returns).reshape(-1, 1),
            split=split, lookback=self.lookback
        ).flatten()
        y_pred_prices = self.preparator.inverse_transform(
            np.asarray(y_pred_returns).reshape(-1, 1),
            split=split, lookback=self.lookback
        ).flatten()
        eps = 1e-9
        return float(np.mean(np.abs(y_pred_prices - y_true_prices)
                             / (np.abs(y_true_prices) + eps)))

    def build_lstm_model(self, config):
        """Build LSTM model with given configuration."""
        n_features = self.X_train.shape[2]

        model = Sequential([
            LSTM(LSTM_UNITS[0], input_shape=(self.lookback, n_features),
                 return_sequences=True, name='LSTM_1'),
            Dropout(config['dropout']),
            LSTM(LSTM_UNITS[1], return_sequences=False, name='LSTM_2'),
            Dropout(config['dropout']),
            Dense(config['dense_units'], activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(1, activation='linear', name='Output')
        ])

        optimizer = Adam(learning_rate=config['lr'])
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def build_cnn_model(self, config):
        """Build CNN model with given configuration.

        Pool-Size muss dynamisch begrenzt werden: bei sehr kurzen Sequenzen (z.B. L=1)
        kann MaxPool1D mit pool_size=2 oder 4 nicht angewandt werden — die Sequence-
        Länge würde 0. Hier: pool effektiv min(pool, current_seq_len). Wenn
        current_seq_len <= 1, wird der MaxPool-Layer komplett übersprungen.
        """
        n_features = self.X_train.shape[2]
        pool = config['pool']

        layers = [
            Conv1D(CONV_FILTERS[0], kernel_size=config['kernel'], activation='relu',
                   input_shape=(self.lookback, n_features), padding='same', name='Conv1D_1'),
        ]
        seq = self.lookback  # padding='same' erhält seq_len

        if seq >= pool and pool > 1:
            layers.append(MaxPooling1D(pool_size=pool, name='MaxPool_1'))
            seq = seq // pool
        layers.append(Dropout(config['dropout']))

        layers.append(Conv1D(CONV_FILTERS[1], kernel_size=config['kernel'],
                             activation='relu', padding='same', name='Conv1D_2'))
        if seq >= pool and pool > 1:
            layers.append(MaxPooling1D(pool_size=pool, name='MaxPool_2'))
            seq = seq // pool
        layers.append(Dropout(config['dropout']))

        layers.append(Conv1D(CONV_FILTERS[2], kernel_size=config['kernel'],
                             activation='relu', padding='same', name='Conv1D_3'))
        layers.append(Dropout(config['dropout']))

        layers.extend([
            Flatten(name='Flatten'),
            Dense(DENSE_UNITS_CNN, activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(DENSE_UNITS_CNN // 2, activation='relu', name='Dense_2'),
            Dense(1, activation='linear', name='Output')
        ])

        model = Sequential(layers)
        optimizer = Adam(learning_rate=LEARNING_RATE_CNN)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def build_gru_model(self, config):
        """Build GRU model with given configuration."""
        n_features = self.X_train.shape[2]

        model = Sequential([
            GRU(GRU_UNITS[0], input_shape=(self.lookback, n_features),
                 return_sequences=True, name='GRU_1'),
            Dropout(config['dropout']),
            GRU(GRU_UNITS[1], return_sequences=False, name='GRU_2'),
            Dropout(config['dropout']),
            Dense(config['dense_units'], activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(1, activation='linear', name='Output')
        ])

        optimizer = Adam(learning_rate=config['lr'])
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def build_informer_model(self, config):
        """Build Informer model with given configuration (PyTorch)."""
        from models.informer_model import build_informer
        n_features = self.X_train.shape[2]
        informer_config = {**config,
                           'e_layers': INFORMER_E_LAYERS,
                           'd_layers': INFORMER_D_LAYERS,
                           'factor': INFORMER_FACTOR}
        model = build_informer(self.lookback, n_features, informer_config)
        return model

    def train_informer_model(self, model, config, verbose=0):
        """Train Informer model with PyTorch training loop. Returns history dict.

        Übergibt X_val/y_val für per-Epoche Val-Loss (analog zu Keras
        validation_data), damit die History direkt mit Keras-History
        vergleichbar ist.
        """
        from models.informer_model import train_informer
        history = train_informer(model, self.X_train, self.y_train,
                                 config, self.lookback, verbose=verbose,
                                 X_val=self.X_val, y_val=self.y_val)
        return history

    def evaluate_informer_model(self, model, save_predictions_dir=None):
        """Evaluate Informer model — NUR auf Train + Val (kein Test).

        Wenn save_predictions_dir gesetzt: schreibt pred_train.npz + pred_val.npz.
        """
        from models.informer_model import evaluate_informer, predict_informer

        train_loss, _train_mae = evaluate_informer(
            model, self.X_train, self.y_train, self.lookback)
        val_loss, _val_mae = evaluate_informer(
            model, self.X_val, self.y_val, self.lookback)

        y_train_pred = predict_informer(model, self.X_train, self.lookback)
        y_val_pred = predict_informer(model, self.X_val, self.lookback)

        if save_predictions_dir is not None:
            self._save_split_pred(self.y_train, y_train_pred, 'train',
                                  os.path.join(save_predictions_dir, 'pred_train.npz'))
            self._save_split_pred(self.y_val, y_val_pred, 'val',
                                  os.path.join(save_predictions_dir, 'pred_val.npz'))

        return {
            'train_loss': float(train_loss),
            'train_mare': self._mare_on_prices(self.y_train, y_train_pred, 'train'),
            'val_loss': float(val_loss),
            'val_mare': self._mare_on_prices(self.y_val, y_val_pred, 'val'),
        }

    def train_model(self, model, config, verbose=0):
        """Train Keras model and return history."""
        # With percentage returns, val data has a similar distribution to training data.
        # Fixed epochs ensure fair comparison across all configs.
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config['epochs'],
            batch_size=config['batch'],
            verbose=verbose
        )

        return history

    def evaluate_model(self, model, save_predictions_dir=None):
        """Evaluate Keras model — NUR auf Train + Val.

        Test wird im Tuning bewusst NICHT berührt (Data-Leakage-Vermeidung).
        Wenn save_predictions_dir gesetzt ist, werden pred_train.npz und
        pred_val.npz dorthin geschrieben (Returns + Preise per Inverse-Transform).
        """
        train_loss = model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss = model.evaluate(self.X_val, self.y_val, verbose=0)

        y_train_pred = model.predict(self.X_train, verbose=0).flatten()
        y_val_pred = model.predict(self.X_val, verbose=0).flatten()

        if save_predictions_dir is not None:
            self._save_split_pred(self.y_train, y_train_pred, 'train',
                                  os.path.join(save_predictions_dir, 'pred_train.npz'))
            self._save_split_pred(self.y_val, y_val_pred, 'val',
                                  os.path.join(save_predictions_dir, 'pred_val.npz'))

        return {
            'train_loss': float(train_loss),
            'train_mare': self._mare_on_prices(self.y_train, y_train_pred, 'train'),
            'val_loss': float(val_loss),
            'val_mare': self._mare_on_prices(self.y_val, y_val_pred, 'val'),
        }

    def _save_split_pred(self, y_true_returns, y_pred_returns, split, npz_path):
        """Speichert npz mit y_actual + y_pred jeweils in Returns UND Preisen."""
        y_actual_prices = self.preparator.inverse_transform(
            np.asarray(y_true_returns).reshape(-1, 1),
            split=split, lookback=self.lookback).flatten()
        y_pred_prices = self.preparator.inverse_transform(
            np.asarray(y_pred_returns).reshape(-1, 1),
            split=split, lookback=self.lookback).flatten()
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path,
                 y_actual_returns=np.asarray(y_true_returns),
                 y_pred_returns=np.asarray(y_pred_returns),
                 y_actual_prices=y_actual_prices,
                 y_pred_prices=y_pred_prices)

    def save_training_plot(self, history, config_name, model_type, output_dir, metrics=None):
        """Speichert pro Config EINE Zusammenfassung: zwei Balken (Train / Val MARE).

        Test wird im Tuning nicht berührt (Data-Leakage-Vermeidung), also auch
        nicht geplottet. `history` wird ignoriert, bleibt für Signatur-Compat.
        """
        if not metrics:
            return
        labels = ['Train', 'Val']
        keys = ['train_mare', 'val_mare']
        vals = [float(metrics.get(k, float('nan'))) for k in keys]
        colors = ['#7CB342', '#1E88E5']

        plt.figure(figsize=(4.5, 3.8))
        ax = plt.gca()
        bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.6)
        for bar, v in zip(bars, vals):
            # Label auf die Säulenspitze; bei Werten > YLIM oberhalb clipping
            y_text = min(v, MARE_PLOT_YLIM[1]) if np.isfinite(v) else 0
            ax.text(bar.get_x() + bar.get_width() / 2, y_text,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('MARE  (|pred-y|/|y|)  auf Preisebene', fontsize=10)
        ax.set_title(f'{model_type} — {config_name}', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(*MARE_PLOT_YLIM)  # fixe Skala für Config-Vergleichbarkeit
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_name}_mare.png'), dpi=90)
        plt.close()

    def _save_history(self, history, config_dir):
        """Save training history to JSON with einheitlichen Keys für Keras + Informer.

        Keras benutzt 'loss' / 'val_loss', Informer 'train_loss' / 'val_loss'.
        Hier wird alles auf 'train_loss' / 'val_loss' normalisiert.
        """
        if hasattr(history, 'history'):
            src = history.history
        else:
            src = history  # Informer dict

        data = {}
        for k, vals in src.items():
            key = 'train_loss' if k == 'loss' else k
            data[key] = [float(v) for v in vals]

        with open(os.path.join(config_dir, 'history.json'), 'w') as f:
            json.dump(data, f)

    def _save_results(self, results, results_file):
        """Save results to JSON file."""
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def run_lstm_tuning(self):
        """Run full grid search for LSTM."""
        all_configs = generate_lstm_configs()
        completed = self._get_completed_configs(self.lstm_results)
        pending = [c for c in all_configs if c['name'] not in completed]

        print("\n" + "=" * 70)
        print("LSTM FULL GRID SEARCH")
        print("=" * 70)
        print(f"Total configs: {len(all_configs)}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {len(pending)}")

        for i, config in enumerate(pending, 1):
            print(f"\n[{len(completed) + i}/{len(all_configs)}] Testing: {config['name']}")
            print(f"  dropout={config['dropout']}, dense={config['dense_units']}, "
                  f"lr={config['lr']}, batch={config['batch']}, epochs={config['epochs']}")

            config_dir = os.path.join(self.lstm_dir, config['name'])
            os.makedirs(config_dir, exist_ok=True)

            start_time = time.time()

            try:
                model = self.build_lstm_model(config)
                history = self.train_model(model, config, verbose=0)
                metrics = self.evaluate_model(model, save_predictions_dir=config_dir)
                training_time = time.time() - start_time

                result = {
                    'config_name': config['name'],
                    'config': config,
                    'metrics': metrics,
                    'training_time': training_time,
                    'epochs_trained': len(history.history['loss'])
                }

                self.save_training_plot(history, config['name'], 'LSTM', config_dir, metrics)
                self._save_history(history, config_dir)
                model.save(os.path.join(config_dir, 'model.keras'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Val MARE: {metrics['val_mare']:.6f}, "
                      f"Train MARE: {metrics['train_mare']:.6f}, Time: {training_time:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    'config_name': config['name'],
                    'config': config,
                    'error': str(e)
                }

            self.lstm_results.append(result)

            # Save after each config
            self._save_results(self.lstm_results, os.path.join(self.lstm_dir, 'all_results.json'))

    def run_cnn_tuning(self):
        """Run full grid search for CNN."""
        all_configs = generate_cnn_configs()
        completed = self._get_completed_configs(self.cnn_results)
        pending = [c for c in all_configs if c['name'] not in completed]

        print("\n" + "=" * 70)
        print("CNN FULL GRID SEARCH")
        print("=" * 70)
        print(f"Total configs: {len(all_configs)}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {len(pending)}")

        for i, config in enumerate(pending, 1):
            print(f"\n[{len(completed) + i}/{len(all_configs)}] Testing: {config['name']}")
            print(f"  kernel={config['kernel']}, pool={config['pool']}, "
                  f"dropout={config['dropout']}, batch={config['batch']}, epochs={config['epochs']}")

            config_dir = os.path.join(self.cnn_dir, config['name'])
            os.makedirs(config_dir, exist_ok=True)

            start_time = time.time()

            try:
                model = self.build_cnn_model(config)
                history = self.train_model(model, config, verbose=0)
                metrics = self.evaluate_model(model, save_predictions_dir=config_dir)
                training_time = time.time() - start_time

                result = {
                    'config_name': config['name'],
                    'config': config,
                    'metrics': metrics,
                    'training_time': training_time,
                    'epochs_trained': len(history.history['loss'])
                }

                self.save_training_plot(history, config['name'], 'CNN', config_dir, metrics)
                self._save_history(history, config_dir)
                model.save(os.path.join(config_dir, 'model.keras'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Val MARE: {metrics['val_mare']:.6f}, "
                      f"Train MARE: {metrics['train_mare']:.6f}, Time: {training_time:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    'config_name': config['name'],
                    'config': config,
                    'error': str(e)
                }

            self.cnn_results.append(result)

            # Save after each config
            self._save_results(self.cnn_results, os.path.join(self.cnn_dir, 'all_results.json'))

    def run_gru_tuning(self):
        """Run full grid search for GRU."""
        all_configs = generate_gru_configs()
        completed = self._get_completed_configs(self.gru_results)
        pending = [c for c in all_configs if c['name'] not in completed]

        print("\n" + "=" * 70)
        print("GRU FULL GRID SEARCH")
        print("=" * 70)
        print(f"Total configs: {len(all_configs)}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {len(pending)}")

        for i, config in enumerate(pending, 1):
            print(f"\n[{len(completed) + i}/{len(all_configs)}] Testing: {config['name']}")
            print(f"  dropout={config['dropout']}, dense={config['dense_units']}, "
                  f"lr={config['lr']}, batch={config['batch']}, epochs={config['epochs']}")

            config_dir = os.path.join(self.gru_dir, config['name'])
            os.makedirs(config_dir, exist_ok=True)

            start_time = time.time()

            try:
                model = self.build_gru_model(config)
                history = self.train_model(model, config, verbose=0)
                metrics = self.evaluate_model(model, save_predictions_dir=config_dir)
                training_time = time.time() - start_time

                result = {
                    'config_name': config['name'],
                    'config': config,
                    'metrics': metrics,
                    'training_time': training_time,
                    'epochs_trained': len(history.history['loss'])
                }

                self.save_training_plot(history, config['name'], 'GRU', config_dir, metrics)
                self._save_history(history, config_dir)
                model.save(os.path.join(config_dir, 'model.keras'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Val MARE: {metrics['val_mare']:.6f}, "
                      f"Train MARE: {metrics['train_mare']:.6f}, Time: {training_time:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    'config_name': config['name'],
                    'config': config,
                    'error': str(e)
                }

            self.gru_results.append(result)

            # Save after each config
            self._save_results(self.gru_results, os.path.join(self.gru_dir, 'all_results.json'))

    def run_informer_tuning(self):
        """Run full grid search for Informer (PyTorch)."""
        all_configs = generate_informer_configs()
        completed = self._get_completed_configs(self.informer_results)
        pending = [c for c in all_configs if c['name'] not in completed]

        print("\n" + "=" * 70)
        print("INFORMER FULL GRID SEARCH")
        print("=" * 70)
        print(f"Total configs: {len(all_configs)}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {len(pending)}")

        for i, config in enumerate(pending, 1):
            print(f"\n[{len(completed) + i}/{len(all_configs)}] Testing: {config['name']}")
            print(f"  d_model={config['d_model']}, n_heads={config['n_heads']}, "
                  f"dropout={config['dropout']}, lr={config['lr']}, "
                  f"batch={config['batch']}, epochs={config['epochs']}")

            config_dir = os.path.join(self.informer_dir, config['name'])
            os.makedirs(config_dir, exist_ok=True)

            start_time = time.time()

            try:
                model = self.build_informer_model(config)
                history = self.train_informer_model(model, config, verbose=0)
                metrics = self.evaluate_informer_model(model, save_predictions_dir=config_dir)
                training_time = time.time() - start_time

                result = {
                    'config_name': config['name'],
                    'config': config,
                    'metrics': metrics,
                    'training_time': training_time,
                    'epochs_trained': len(history['train_loss'])
                }

                # Pro Config: EIN Wert je Split (train/val/test MARE) — keine Epoch-Historie
                self.save_training_plot(history, config['name'], 'INFORMER', config_dir, metrics)

                # Keep lightweight history JSON + model weights
                self._save_history(history, config_dir)
                import torch
                torch.save(model.state_dict(), os.path.join(config_dir, 'model.pt'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Val MARE: {metrics['val_mare']:.6f}, "
                      f"Train MARE: {metrics['train_mare']:.6f}, Time: {training_time:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    'config_name': config['name'],
                    'config': config,
                    'error': str(e)
                }

            self.informer_results.append(result)

            # Save after each config
            self._save_results(self.informer_results, os.path.join(self.informer_dir, 'all_results.json'))

    def _find_best(self, results):
        """Find best config from results by val_loss. Returns None if no valid results."""
        valid = [r for r in results if 'metrics' in r]
        if not valid:
            return None, []
        return min(valid, key=lambda x: x['metrics']['val_loss']), valid

    def find_best_configs(self):
        """Find and save the best configurations."""
        print("\n" + "=" * 70)
        print("FINDING BEST CONFIGURATIONS (by Val Loss)")
        print("=" * 70)

        best_lstm, valid_lstm = self._find_best(self.lstm_results)
        best_cnn, valid_cnn = self._find_best(self.cnn_results)
        best_gru, valid_gru = self._find_best(self.gru_results)
        best_informer, valid_informer = self._find_best(self.informer_results)

        best_configs = {
            'timestamp': datetime.now().isoformat(),
            'total_lstm_configs': len(valid_lstm),
            'total_cnn_configs': len(valid_cnn),
            'total_gru_configs': len(valid_gru),
            'total_informer_configs': len(valid_informer),
        }

        models = [
            ('LSTM', best_lstm),
            ('CNN', best_cnn),
            ('GRU', best_gru),
            ('INFORMER', best_informer),
        ]

        for name, best in models:
            if best is None:
                continue
            best_configs[f'best_{name.lower()}'] = {
                'config_name': best['config_name'],
                'config': best['config'],
                'metrics': best['metrics'],
                'training_time': best['training_time']
            }
            print(f"\n  BEST {name}: {best['config_name']}")
            print(f"    Parameters: {', '.join(f'{k}={v}' for k, v in best['config'].items() if k != 'name')}")
            print(f"    Train Loss: {best['metrics']['train_loss']:.6f}  |  Train MARE: {best['metrics']['train_mare']:.6f}")
            print(f"    Val   Loss: {best['metrics']['val_loss']:.6f}  |  Val   MARE: {best['metrics']['val_mare']:.6f}")

        # Summary table
        print("\n" + "-" * 70)
        print("BEST MODEL COMPARISON (selected by Val Loss) — NO TEST (Test only at end of pipeline):")
        print("-" * 90)
        print(f"{'Model':<12} {'Config':<40} {'TrainLoss':<12} {'TrainMARE':<12} {'ValLoss':<12} {'ValMARE':<12}")
        print("-" * 90)
        for name, best in models:
            if best is None:
                continue
            m = best['metrics']
            print(f"{name:<12} {best['config_name']:<40} {m['train_loss']:<12.6f} {m['train_mare']:<12.6f} {m['val_loss']:<12.6f} {m['val_mare']:<12.6f}")

        # Save best configs
        best_configs_path = os.path.join(self.output_dir, 'best_configurations.json')
        with open(best_configs_path, 'w') as f:
            json.dump(best_configs, f, indent=2)

        return best_configs

    def _print_top_table(self, results, model_name, top_n=3):
        """Print top N results for a model type, sorted by Val Loss."""
        valid = [r for r in results if 'metrics' in r]
        if not valid:
            return
        sorted_results = sorted(valid, key=lambda x: x['metrics']['val_loss'])[:top_n]

        print(f"\n{model_name} TOP {top_n} (sorted by Val Loss):")
        print("-" * 115)
        print(f"{'Rank':<5} {'Config':<35} {'TrainLoss':<12} {'TrainMARE':<12} {'ValLoss':<12} {'ValMARE':<12} {'Time':<10}")
        print("-" * 115)

        for i, r in enumerate(sorted_results, 1):
            m = r['metrics']
            print(f"{i:<5} {r['config_name']:<35} {m['train_loss']:<12.6f} {m['train_mare']:<12.6f} {m['val_loss']:<12.6f} {m['val_mare']:<12.6f} {r['training_time']:<10.1f}s")

    def create_comparison_table(self):
        """Create a comparison table of all results."""
        print("\n" + "=" * 70)
        print("TOP 3 RESULTS PER MODEL")
        print("=" * 70)

        self._print_top_table(self.lstm_results, 'LSTM')
        self._print_top_table(self.cnn_results, 'CNN')
        self._print_top_table(self.gru_results, 'GRU')
        self._print_top_table(self.informer_results, 'INFORMER')

    def analyze_hyperparameters(self):
        """Analyze which hyperparameters perform best on average."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER ANALYSIS")
        print("=" * 70)

        # LSTM Analysis
        valid_lstm = [r for r in self.lstm_results if 'metrics' in r]

        if valid_lstm:
            print("\nLSTM - Average Val Loss by Parameter:")
            print("-" * 50)

            # Dropout
            for dropout in LSTM_DROPOUT:
                subset = [r for r in valid_lstm if r['config']['dropout'] == dropout]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Dropout={dropout}: {avg:.6f}")

            # Dense Units
            for units in LSTM_DENSE_UNITS:
                subset = [r for r in valid_lstm if r['config']['dense_units'] == units]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Dense={units}: {avg:.6f}")

            # Learning Rate
            for lr in LSTM_LEARNING_RATE:
                subset = [r for r in valid_lstm if r['config']['lr'] == lr]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  LR={lr}: {avg:.6f}")

            # Batch Size
            for batch in LSTM_BATCH_SIZE:
                subset = [r for r in valid_lstm if r['config']['batch'] == batch]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Batch={batch}: {avg:.6f}")

        # CNN Analysis
        valid_cnn = [r for r in self.cnn_results if 'metrics' in r]

        if valid_cnn:
            print("\nCNN - Average Val Loss by Parameter:")
            print("-" * 50)

            # Kernel Size
            for kernel in CNN_KERNEL_SIZE:
                subset = [r for r in valid_cnn if r['config']['kernel'] == kernel]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Kernel={kernel}: {avg:.6f}")

            # Pool Size
            for pool in CNN_POOL_SIZE:
                subset = [r for r in valid_cnn if r['config']['pool'] == pool]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Pool={pool}: {avg:.6f}")

            # Dropout
            for dropout in CNN_DROPOUT:
                subset = [r for r in valid_cnn if r['config']['dropout'] == dropout]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Dropout={dropout}: {avg:.6f}")

            # Batch Size
            for batch in CNN_BATCH_SIZE:
                subset = [r for r in valid_cnn if r['config']['batch'] == batch]
                if subset:
                    avg = np.mean([r['metrics']['val_loss'] for r in subset])
                    print(f"  Batch={batch}: {avg:.6f}")

    def run(self):
        """Run complete full grid search."""
        print("\n" + "=" * 70)
        print("FULL HYPERPARAMETER GRID SEARCH")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"LSTM total configs: {len(generate_lstm_configs())}")
        print(f"CNN total configs: {len(generate_cnn_configs())}")
        print(f"GRU total configs: {len(generate_gru_configs())}")
        print(f"Informer total configs: {len(generate_informer_configs())}")

        total_start = time.time()

        # Prepare data
        self.prepare_data()

        # Run tuning
        self.run_lstm_tuning()
        self.run_cnn_tuning()
        self.run_gru_tuning()
        self.run_informer_tuning()

        # Find best configs
        best = self.find_best_configs()

        # Create comparison table
        self.create_comparison_table()

        # Analyze hyperparameters
        self.analyze_hyperparameters()

        total_time = time.time() - total_start

        print("\n" + "=" * 70)
        print("FULL GRID SEARCH COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"Results saved to: {self.output_dir}/")
        print(f"Best configurations: {self.output_dir}/best_configurations.json")

        return best


def run_single_config(data_path, model_type, config_name, output_dir=os.path.join(STEP_DIR, 'results'),
                      index_name='SP500', lookback=None):
    """
    Train and evaluate a single config for a specific Lookback L.

    Each call is independent: loads data, builds model, trains, saves results.
    Wird als Subprocess pro (index, model, config, L) dispatched.
    """
    # Seeds pro Task neu setzen. TF-Threading wird NICHT mehr hier gesetzt —
    # das macht der _pool_initializer in run_pipeline.py einmal pro Worker.
    # Ein erneuter tf.config.threading.set_* nach der ersten TF-Op wirft
    # RuntimeError im wiederverwendeten Pool-Worker. Torch-Threading darf
    # idempotent gesetzt werden, also bleibt das hier (Informer-Pfad).
    import random
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    if model_type != 'informer':
        import tensorflow as tf
        tf.random.set_seed(42)
    else:
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(int(os.environ.get('TF_WORKER_THREADS', '4')))

    tuner = FullGridSearchTuner(data_path, output_dir=output_dir,
                                index_name=index_name, lookback=lookback)
    tuner.prepare_data()

    # Find the config by name
    if model_type == 'lstm':
        all_configs = generate_lstm_configs()
        config_dir_base = tuner.lstm_dir
    elif model_type == 'cnn':
        all_configs = generate_cnn_configs()
        config_dir_base = tuner.cnn_dir
    elif model_type == 'gru':
        all_configs = generate_gru_configs()
        config_dir_base = tuner.gru_dir
    elif model_type == 'informer':
        all_configs = generate_informer_configs()
        config_dir_base = tuner.informer_dir
    else:
        print(f"ERROR: Unknown model type '{model_type}'")
        return None

    config = next((c for c in all_configs if c['name'] == config_name), None)
    if config is None:
        print(f"ERROR: Config '{config_name}' not found for {model_type}")
        return None

    config_dir = os.path.join(config_dir_base, config['name'])
    os.makedirs(config_dir, exist_ok=True)

    start_time = time.time()
    model = None  # damit finally del greifen kann auch bei Exception
    try:
        if model_type == 'informer':
            model = tuner.build_informer_model(config)
            history = tuner.train_informer_model(model, config, verbose=0)
            metrics = tuner.evaluate_informer_model(model, save_predictions_dir=config_dir)
            training_time = time.time() - start_time

            result = {
                'config_name': config['name'],
                'config': config,
                'metrics': metrics,
                'training_time': training_time,
                'epochs_trained': len(history['train_loss'])
            }

            # Pro Config: EIN Wert je Split (train/val/test MARE) — keine Epoch-Historie
            tuner.save_training_plot(history, config['name'], 'INFORMER', config_dir, metrics)

            # Save history JSON + model weights
            tuner._save_history(history, config_dir)
            import torch
            torch.save(model.state_dict(), os.path.join(config_dir, 'model.pt'))
        else:
            if model_type == 'lstm':
                model = tuner.build_lstm_model(config)
            elif model_type == 'cnn':
                model = tuner.build_cnn_model(config)
            else:
                model = tuner.build_gru_model(config)

            history = tuner.train_model(model, config, verbose=0)
            metrics = tuner.evaluate_model(model, save_predictions_dir=config_dir)
            training_time = time.time() - start_time

            result = {
                'config_name': config['name'],
                'config': config,
                'metrics': metrics,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss'])
            }

            tuner.save_training_plot(history, config['name'], model_type.upper(), config_dir, metrics)
            tuner._save_history(history, config_dir)
            # Keras 3: .keras Format ist Standard. Phase-3 Bilateral-Loader probiert
            # zuerst .keras, dann .h5 — alte Phase-1-Outputs bleiben kompatibel.
            model.save(os.path.join(config_dir, 'model.keras'))

        with open(os.path.join(config_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  [{model_type.upper()}] {config['name']}: "
              f"Val Loss={metrics['val_loss']:.6f}, Val MARE={metrics['val_mare']:.6f}, "
              f"Train MARE={metrics['train_mare']:.6f}, "
              f"Time={training_time:.1f}s")

    except Exception as e:
        print(f"  [{model_type.upper()}] {config['name']}: ERROR: {e}")
        result = {
            'config_name': config['name'],
            'config': config,
            'error': str(e)
        }
        with open(os.path.join(config_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
    finally:
        # Memory-Cleanup nach jedem Task. PyTorch-Informer-Modelle akkumulieren
        # in wiederverwendeten Pool-Workern leicht 100+ MB pro Task. Mit dem
        # max_tasks_per_child=3 im Pool plus explizitem del + gc kommen wir
        # auch ohne ROCm-empty_cache aus (gibt's auf CPU eh nicht).
        try:
            del model
            del tuner
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

    return result


def run_best_per_L_test(data_path, model_type, index_name, lookback,
                        output_dir=os.path.join(STEP_DIR, 'results')):
    """Phase 3: predicted die in Phase 2 ausgewählte global-best Config bei dem
    gegebenen `lookback` auf Train+Val+Test.

    Bevorzugt wird die GLOBAL beste Config (gewählt über alle Coarse L) aus
    parameter_tuning/results/<MODEL>/<INDEX>/global_best.json. So nutzt der
    Sweep über L=1..60 IMMER die gleiche Config.

    Wenn keine global_best.json existiert, wird die beste Config aus dem
    L<L>/ Ordner gewählt (legacy / Per-L-Best Verhalten).

    Modell-Beschaffung:
      - Wenn unter <l_dir>/<best_cfg_name>/{model.keras|model.h5} (bzw. model.pt
        für Informer) ein in Phase 1 trainiertes Modell vorliegt -> LOAD
        (kein Re-Training). Bilateral Loader: .keras zuerst, .h5 als Fallback.
      - Sonst: build_model + train mit best_cfg bei sequence_length=lookback.
        Das ist der Normalfall für L ausserhalb der COARSE_LOOKBACKS.

    Output im L<L>-Unterordner:
      - best_test_metrics.json
      - pred_test.npz, pred_train.npz, pred_val.npz
    """
    # Seeds pro Task neu setzen. TF-Threading wird NICHT mehr hier gesetzt —
    # das macht der _pool_initializer in run_pipeline.py einmal pro Worker.
    # Torch-Threading darf idempotent gesetzt werden, also bleibt das hier.
    import random
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    if model_type != 'informer':
        import tensorflow as tf
        tf.random.set_seed(42)
    else:
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(int(os.environ.get('TF_WORKER_THREADS', '4')))

    tuner = FullGridSearchTuner(data_path, output_dir=output_dir,
                                index_name=index_name, lookback=lookback)
    tuner.prepare_data()

    model_dir_map = {'lstm': tuner.lstm_dir, 'cnn': tuner.cnn_dir,
                     'gru': tuner.gru_dir, 'informer': tuner.informer_dir}
    l_dir = model_dir_map[model_type]
    # global_best.json liegt eine Ebene höher (auf <MODEL>/<INDEX>/-Ebene)
    global_best_path = os.path.join(os.path.dirname(l_dir), 'global_best.json')

    # === Auswahl der Config ===
    best_cfg = None
    best_meta = None
    selection_source = None

    if os.path.isfile(global_best_path):
        try:
            with open(global_best_path) as f:
                gb = json.load(f)
            best_cfg = gb.get('config')
            best_meta = {
                'config_name': gb.get('config_name'),
                'val_loss_at_selection_L': gb.get('val_loss'),
                'val_mare_at_selection_L': gb.get('val_mare'),
                'selected_at_L': gb.get('selected_at_L'),
            }
            selection_source = 'global_best'
        except Exception as e:
            print(f"  WARN: global_best.json unleserlich ({e}), fallback per-L-best")

    if best_cfg is None:
        all_results = []
        if os.path.isdir(l_dir):
            for cd in sorted(os.listdir(l_dir)):
                rj = os.path.join(l_dir, cd, 'results.json')
                if os.path.isfile(rj):
                    try:
                        with open(rj) as f:
                            r = json.load(f)
                        if 'metrics' in r and 'val_loss' in r['metrics']:
                            all_results.append(r)
                    except Exception:
                        continue
        if not all_results:
            print(f"  [{model_type.upper()}] L={lookback} {index_name}: "
                  f"weder global_best noch per-L Tuning-Ergebnisse gefunden")
            return None
        best = min(all_results, key=lambda x: x['metrics']['val_loss'])
        best_cfg = best['config']
        best_meta = {
            'config_name': best['config_name'],
            'val_loss_at_selection_L': best['metrics']['val_loss'],
            'val_mare_at_selection_L': best['metrics'].get('val_mare'),
            'selected_at_L': lookback,
        }
        selection_source = 'per_L_best'

    # Fallback für Output-Kompatibilität
    best = {'config_name': best_meta['config_name'], 'config': best_cfg,
            'metrics': {
                'val_loss': best_meta.get('val_loss_at_selection_L'),
                'val_mare': best_meta.get('val_mare_at_selection_L'),
            }}

    # Modell beschaffen: aus Phase-1-Speicherort laden, falls vorhanden — sonst
    # mit best_cfg bei sequence_length=lookback neu trainieren. Phase 1 hat
    # best_cfg bei den COARSE_LOOKBACKS bereits trainiert; für L ausserhalb
    # davon existiert kein Modell-File -> Re-Training nötig.
    #
    # Bilateral-Loader: .keras (neuer Standard) zuerst, .h5 als Fallback für
    # bereits existierende Phase-1-Outputs aus dem alten Save-Format.
    saved_cfg_dir = os.path.join(l_dir, best_meta['config_name'])
    keras_files = [os.path.join(saved_cfg_dir, 'model.keras'),
                   os.path.join(saved_cfg_dir, 'model.h5')]
    torch_model_file = os.path.join(saved_cfg_dir, 'model.pt')
    loaded_from_disk = False

    def _find_keras_file():
        for p in keras_files:
            if os.path.isfile(p):
                return p
        return None

    if model_type in ('lstm', 'cnn', 'gru'):
        builder = {'lstm': tuner.build_lstm_model,
                   'cnn':  tuner.build_cnn_model,
                   'gru':  tuner.build_gru_model}[model_type]
        kf = _find_keras_file()
        if kf is not None:
            model = load_model(kf, compile=False)
            loaded_from_disk = True
        else:
            model = builder(best_cfg)
            tuner.train_model(model, best_cfg, verbose=0)
        y_train_pred_returns = model.predict(tuner.X_train, verbose=0).flatten()
        y_val_pred_returns   = model.predict(tuner.X_val,   verbose=0).flatten()
        y_test_pred_returns  = model.predict(tuner.X_test,  verbose=0).flatten()
    elif model_type == 'informer':
        from models.informer_model import predict_informer
        model = tuner.build_informer_model(best_cfg)
        if os.path.isfile(torch_model_file):
            import torch
            model.load_state_dict(torch.load(torch_model_file, map_location='cpu'))
            model.eval()
            loaded_from_disk = True
        else:
            tuner.train_informer_model(model, best_cfg, verbose=0)
        y_train_pred_returns = predict_informer(model, tuner.X_train, lookback)
        y_val_pred_returns   = predict_informer(model, tuner.X_val,   lookback)
        y_test_pred_returns  = predict_informer(model, tuner.X_test,  lookback)
    else:
        return None

    # Test MSE + Test MARE
    y_test_true = tuner.y_test
    test_loss = float(np.mean((y_test_pred_returns - y_test_true) ** 2))
    test_mare = tuner._mare_on_prices(y_test_true, y_test_pred_returns, 'test')

    test_metrics = {
        'best_config_name': best['config_name'],
        'best_config': best_cfg,
        'val_loss_used_for_selection': best['metrics']['val_loss'],
        'val_mare': best['metrics'].get('val_mare'),
        'test_loss': test_loss,
        'test_mare': test_mare,
        'test_rmse': float(np.sqrt(test_loss)),
        'lookback': lookback,
        'config_selection_source': selection_source,
        'config_selected_at_L': best_meta.get('selected_at_L'),
        'model_loaded_from_disk': loaded_from_disk,
    }
    with open(os.path.join(l_dir, 'best_test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Pred-NPZ für alle drei Splits — komplette Reproduktion aller Plots möglich
    tuner._save_split_pred(tuner.y_train, y_train_pred_returns, 'train',
                           os.path.join(l_dir, 'pred_train.npz'))
    tuner._save_split_pred(tuner.y_val, y_val_pred_returns, 'val',
                           os.path.join(l_dir, 'pred_val.npz'))
    tuner._save_split_pred(y_test_true, y_test_pred_returns, 'test',
                           os.path.join(l_dir, 'pred_test.npz'))

    src_tag = 'LOAD' if loaded_from_disk else 'TRAIN'
    print(f"  [{model_type.upper()}] L={lookback:2d} {index_name:10s}: {src_tag} "
          f"best={best['config_name']}, test_loss={test_loss:.6f}, "
          f"test_mare={test_mare:.6f}")
    return test_metrics


if __name__ == "__main__":
    import argparse
    import random
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    try:
        import tensorflow as _tf
        _tf.random.set_seed(42)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(42)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    parser.add_argument('--model', choices=['lstm', 'cnn', 'gru', 'informer', 'all'], default='all',
                        help='Which model type to tune (default: all)')
    parser.add_argument('--config', type=str, default=None,
                        help='Run single config by name (e.g. d0.2_u16_lr0.001_b8_e50)')
    parser.add_argument('--index', type=str, default='SP500',
                        help='Index to tune on (default: SP500)')
    parser.add_argument('--lookback', type=int, default=None,
                        help='Lookback window L. Default: LOOKBACK_WINDOW (=60) aus config.py. '
                             'In der L-aware Pipeline wird pro L ein eigener Subprocess '
                             'mit --lookback dispatched.')
    parser.add_argument('--mode', choices=['tune', 'best-predict'], default='tune',
                        help='tune = Hyperparameter-Tuning (Train + Val). '
                             'best-predict = Liest best config aus L<L>/ Ordner und '
                             'predicted auf Test (Phase 2 der neuen Pipeline).')
    args = parser.parse_args()

    if args.index not in INDICES:
        print(f"Error: Unknown index '{args.index}'. Available: {list(INDICES.keys())}")
        sys.exit(1)

    data_path = os.path.join(
        PROJECT_ROOT,
        'stockData', 'preprocessedData', INDICES[args.index]
    )

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Phase-2 Modus: best config aus L<L>/ lesen und auf Test predicten
    if args.mode == 'best-predict':
        if args.lookback is None:
            print("Error: --mode best-predict benötigt --lookback L")
            sys.exit(1)
        if args.model not in ('lstm', 'cnn', 'gru', 'informer'):
            print("Error: --mode best-predict benötigt --model <lstm|cnn|gru|informer>")
            sys.exit(1)
        run_best_per_L_test(data_path, args.model, args.index, args.lookback)
        sys.exit(0)

    # Single config mode (for parallel dispatch from pipeline)
    if args.config:
        run_single_config(data_path, args.model, args.config,
                          index_name=args.index, lookback=args.lookback)
        sys.exit(0)

    # Full grid search mode
    tuner = FullGridSearchTuner(data_path, output_dir=os.path.join(STEP_DIR, 'results'),
                                index_name=args.index, lookback=args.lookback)
    tuner.prepare_data()

    if args.model in ('lstm', 'all'):
        tuner.run_lstm_tuning()
    if args.model in ('cnn', 'all'):
        tuner.run_cnn_tuning()
    if args.model in ('gru', 'all'):
        tuner.run_gru_tuning()

    # Only run analysis when all models are available
    if args.model == 'all':
        tuner.find_best_configs()
        tuner.create_comparison_table()
        tuner.analyze_hyperparameters()
