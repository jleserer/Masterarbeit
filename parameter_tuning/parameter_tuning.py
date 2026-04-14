"""
Full Hyperparameter Grid Search for LSTM, CNN and GRU Models
==============================================================
Tests ALL possible combinations of hyperparameters.
Skips already completed configurations.

LSTM:     2 × 2 × 2 × 3 × 2 = 48 combinations
CNN:      2 × 2 × 2 × 3 × 2 = 48 combinations
GRU:      2 × 2 × 2 × 3 × 2 = 48 combinations
Informer: 2 × 2 × 2 × 2 × 2 × 2 = 64 combinations
Total: 208 combinations
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from data_preparation import DataPreparator, create_sequences


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# LSTM parameters
LSTM_DROPOUT = [0.2, 0.4]
LSTM_DENSE_UNITS = [16, 32]
LSTM_LEARNING_RATE = [0.001, 0.005]
LSTM_BATCH_SIZE = [8, 16, 32]
LSTM_EPOCHS = [50, 100]

# CNN parameters
CNN_KERNEL_SIZE = [3, 5]
CNN_POOL_SIZE = [2, 4]
CNN_DROPOUT = [0.2, 0.4]
CNN_BATCH_SIZE = [8, 16, 32]
CNN_EPOCHS = [50, 100]

# GRU parameters (same search space as LSTM)
GRU_DROPOUT = [0.2, 0.4]
GRU_DENSE_UNITS = [16, 32]
GRU_LEARNING_RATE = [0.001, 0.005]
GRU_BATCH_SIZE = [8, 16, 32]
GRU_EPOCHS = [50, 100]

# Informer parameters
INFORMER_D_MODEL = [32, 64]
INFORMER_N_HEADS = [4, 8]
INFORMER_DROPOUT = [0.05, 0.1]
INFORMER_LEARNING_RATE = [0.0001, 0.001]
INFORMER_BATCH_SIZE = [16, 32]
INFORMER_EPOCHS = [50, 100]

# Fixed Informer architecture
INFORMER_E_LAYERS = 2
INFORMER_D_LAYERS = 1
INFORMER_FACTOR = 5

# Fixed parameters
LOOKBACK_WINDOW = 60
LSTM_UNITS = [128, 64]
GRU_UNITS = [128, 64]
CONV_FILTERS = [64, 128, 256]
DENSE_UNITS_CNN = 64
LEARNING_RATE_CNN = 0.001

# All available indices
INDICES = {
    'SP500':     'SP500_historical_data.csv',
    'DAX':       'DAX_historical_data.csv',
    'NASDAQ':    'NASDAQ_historical_data.csv',
    'FTSE100':   'FTSE100_historical_data.csv',
    'HANG_SENG': 'HANG_SENG_historical_data.csv',
    'NIKKEI':    'NIKKEI_historical_data.csv',
    '10Y_Bond':  '10-Year Bond_historical_data.csv',
    '30Y_Bond':  '30 Year Bond_historical_data.csv',
}


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
    """Manages full grid search for LSTM, CNN, GRU and Informer models."""

    def __init__(self, data_path, output_dir=os.path.join(STEP_DIR, 'results'), index_name='SP500'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.index_name = index_name
        self.lstm_dir = os.path.join(output_dir, 'LSTM', index_name)
        self.cnn_dir = os.path.join(output_dir, 'CNN', index_name)
        self.gru_dir = os.path.join(output_dir, 'GRU', index_name)
        self.informer_dir = os.path.join(output_dir, 'INFORMER', index_name)

        # Data containers
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        # Results
        self.lstm_results = []
        self.cnn_results = []
        self.gru_results = []
        self.informer_results = []

        # Create directories
        os.makedirs(self.lstm_dir, exist_ok=True)
        os.makedirs(self.cnn_dir, exist_ok=True)
        os.makedirs(self.gru_dir, exist_ok=True)
        os.makedirs(self.informer_dir, exist_ok=True)

        # Load existing results
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

        preparator = DataPreparator(self.data_path)
        train_data, val_data, test_data = preparator.load_and_prepare()

        self.X_train, self.y_train = create_sequences(train_data, LOOKBACK_WINDOW)
        self.X_val, self.y_val = create_sequences(val_data, LOOKBACK_WINDOW)
        self.X_test, self.y_test = create_sequences(test_data, LOOKBACK_WINDOW)

        print(f"\nData prepared:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val:   {self.X_val.shape}")
        print(f"  Test:  {self.X_test.shape}")

    def build_lstm_model(self, config):
        """Build LSTM model with given configuration."""
        n_features = self.X_train.shape[2]

        model = Sequential([
            LSTM(LSTM_UNITS[0], input_shape=(LOOKBACK_WINDOW, n_features),
                 return_sequences=True, name='LSTM_1'),
            Dropout(config['dropout']),
            LSTM(LSTM_UNITS[1], return_sequences=False, name='LSTM_2'),
            Dropout(config['dropout']),
            Dense(config['dense_units'], activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(1, activation='linear', name='Output')
        ])

        optimizer = Adam(learning_rate=config['lr'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_cnn_model(self, config):
        """Build CNN model with given configuration."""
        n_features = self.X_train.shape[2]

        model = Sequential([
            Conv1D(CONV_FILTERS[0], kernel_size=config['kernel'], activation='relu',
                   input_shape=(LOOKBACK_WINDOW, n_features), padding='same', name='Conv1D_1'),
            MaxPooling1D(pool_size=config['pool'], name='MaxPool_1'),
            Dropout(config['dropout']),

            Conv1D(CONV_FILTERS[1], kernel_size=config['kernel'], activation='relu',
                   padding='same', name='Conv1D_2'),
            MaxPooling1D(pool_size=config['pool'], name='MaxPool_2'),
            Dropout(config['dropout']),

            Conv1D(CONV_FILTERS[2], kernel_size=config['kernel'], activation='relu',
                   padding='same', name='Conv1D_3'),
            Dropout(config['dropout']),

            Flatten(name='Flatten'),
            Dense(DENSE_UNITS_CNN, activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(DENSE_UNITS_CNN // 2, activation='relu', name='Dense_2'),
            Dense(1, activation='linear', name='Output')
        ])

        optimizer = Adam(learning_rate=LEARNING_RATE_CNN)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_gru_model(self, config):
        """Build GRU model with given configuration."""
        n_features = self.X_train.shape[2]

        model = Sequential([
            GRU(GRU_UNITS[0], input_shape=(LOOKBACK_WINDOW, n_features),
                 return_sequences=True, name='GRU_1'),
            Dropout(config['dropout']),
            GRU(GRU_UNITS[1], return_sequences=False, name='GRU_2'),
            Dropout(config['dropout']),
            Dense(config['dense_units'], activation='relu', name='Dense_1'),
            Dropout(config['dropout']),
            Dense(1, activation='linear', name='Output')
        ])

        optimizer = Adam(learning_rate=config['lr'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_informer_model(self, config):
        """Build Informer model with given configuration (PyTorch)."""
        from models.informer_model import build_informer
        n_features = self.X_train.shape[2]
        informer_config = {**config,
                           'e_layers': INFORMER_E_LAYERS,
                           'd_layers': INFORMER_D_LAYERS,
                           'factor': INFORMER_FACTOR}
        model = build_informer(LOOKBACK_WINDOW, n_features, informer_config)
        return model

    def train_informer_model(self, model, config, verbose=0):
        """Train Informer model with PyTorch training loop. Returns history dict."""
        from models.informer_model import train_informer
        history = train_informer(model, self.X_train, self.y_train,
                                 config, LOOKBACK_WINDOW, verbose=verbose)
        return history

    def evaluate_informer_model(self, model):
        """Evaluate Informer model on all splits. Returns metrics dict."""
        from models.informer_model import evaluate_informer

        train_loss, train_mae = evaluate_informer(model, self.X_train, self.y_train, LOOKBACK_WINDOW)
        val_loss, val_mae = evaluate_informer(model, self.X_val, self.y_val, LOOKBACK_WINDOW)
        test_loss, test_mae = evaluate_informer(model, self.X_test, self.y_test, LOOKBACK_WINDOW)

        return {
            'train_loss': float(train_loss),
            'train_mae': float(train_mae),
            'train_rmse': float(np.sqrt(train_loss)),
            'val_loss': float(val_loss),
            'val_mae': float(val_mae),
            'val_rmse': float(np.sqrt(val_loss)),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_rmse': float(np.sqrt(test_loss))
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

    def evaluate_model(self, model):
        """Evaluate Keras model and return metrics."""
        train_loss, train_mae = model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss, val_mae = model.evaluate(self.X_val, self.y_val, verbose=0)
        test_loss, test_mae = model.evaluate(self.X_test, self.y_test, verbose=0)

        return {
            'train_loss': float(train_loss),
            'train_mae': float(train_mae),
            'train_rmse': float(np.sqrt(train_loss)),
            'val_loss': float(val_loss),
            'val_mae': float(val_mae),
            'val_rmse': float(np.sqrt(val_loss)),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_rmse': float(np.sqrt(test_loss))
        }

    def save_training_plot(self, history, config_name, model_type, output_dir, metrics=None):
        """Save training history plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        if metrics:
            axes[0].axhline(y=metrics['test_loss'], color='green', linestyle='--', label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'{model_type} - Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        if metrics:
            axes[1].axhline(y=metrics['test_mae'], color='green', linestyle='--', label='Test MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{model_type} - MAE')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_name}_history.png'), dpi=80)
        plt.close()

    def _save_history(self, history, config_dir):
        """Save training history to JSON for later plot regeneration."""
        if hasattr(history, 'history'):
            # Keras history object
            data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        else:
            # Dict (e.g. Informer)
            data = history
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
                metrics = self.evaluate_model(model)
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
                model.save(os.path.join(config_dir, 'model.h5'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"RMSE: {metrics['test_rmse']:.6f}, Time: {training_time:.1f}s")

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
                metrics = self.evaluate_model(model)
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
                model.save(os.path.join(config_dir, 'model.h5'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"RMSE: {metrics['test_rmse']:.6f}, Time: {training_time:.1f}s")

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
                metrics = self.evaluate_model(model)
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
                model.save(os.path.join(config_dir, 'model.h5'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"RMSE: {metrics['test_rmse']:.6f}, Time: {training_time:.1f}s")

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
                metrics = self.evaluate_informer_model(model)
                training_time = time.time() - start_time

                result = {
                    'config_name': config['name'],
                    'config': config,
                    'metrics': metrics,
                    'training_time': training_time,
                    'epochs_trained': len(history['train_loss'])
                }

                # Save training plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.plot(history['train_loss'], label='Train Loss')
                ax.axhline(y=metrics['test_loss'], color='green', linestyle='--', label='Test Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title(f'INFORMER - {config["name"]}')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(config_dir, f'{config["name"]}_history.png'), dpi=80)
                plt.close()

                # Save history and model
                self._save_history(history, config_dir)
                import torch
                torch.save(model.state_dict(), os.path.join(config_dir, 'model.pt'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"RMSE: {metrics['test_rmse']:.6f}, Time: {training_time:.1f}s")

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
            print(f"    Val Loss:  {best['metrics']['val_loss']:.6f}  |  Val MAE:  {best['metrics']['val_mae']:.6f}")
            print(f"    Test Loss: {best['metrics']['test_loss']:.6f}  |  Test MAE: {best['metrics']['test_mae']:.6f}")

        # Summary table
        print("\n" + "-" * 70)
        print("BEST MODEL COMPARISON (selected by Val Loss):")
        print("-" * 70)
        print(f"{'Model':<12} {'Config':<40} {'ValLoss':<12} {'ValMAE':<12} {'TestMAE':<12}")
        print("-" * 70)
        for name, best in models:
            if best is None:
                continue
            m = best['metrics']
            print(f"{name:<12} {best['config_name']:<40} {m['val_loss']:<12.6f} {m['val_mae']:<12.6f} {m['test_mae']:<12.6f}")

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
        print(f"{'Rank':<5} {'Config':<35} {'ValLoss':<12} {'ValMAE':<12} {'TestMAE':<12} {'Time':<10}")
        print("-" * 115)

        for i, r in enumerate(sorted_results, 1):
            m = r['metrics']
            print(f"{i:<5} {r['config_name']:<35} {m['val_loss']:<12.6f} {m['val_mae']:<12.6f} {m['test_mae']:<12.6f} {r['training_time']:<10.1f}s")

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


def run_single_config(data_path, model_type, config_name, output_dir=os.path.join(STEP_DIR, 'results'), index_name='SP500'):
    """
    Train and evaluate a single config. Designed for parallel dispatch.
    Each call is independent: loads data, builds model, trains, saves results.
    """
    if model_type != 'informer':
        import tensorflow as tf
        # Limit TF threads to avoid oversubscription when running in parallel
        n_threads = int(os.environ.get('TF_WORKER_THREADS', '4'))
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    else:
        import torch
        torch.set_num_threads(int(os.environ.get('TF_WORKER_THREADS', '4')))

    tuner = FullGridSearchTuner(data_path, output_dir=output_dir, index_name=index_name)
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
    try:
        if model_type == 'informer':
            model = tuner.build_informer_model(config)
            history = tuner.train_informer_model(model, config, verbose=0)
            metrics = tuner.evaluate_informer_model(model)
            training_time = time.time() - start_time

            result = {
                'config_name': config['name'],
                'config': config,
                'metrics': metrics,
                'training_time': training_time,
                'epochs_trained': len(history['train_loss'])
            }

            # Save training plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(history['train_loss'], label='Train Loss')
            ax.axhline(y=metrics['test_loss'], color='green', linestyle='--', label='Test Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title(f'INFORMER - {config["name"]}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(config_dir, f'{config["name"]}_history.png'), dpi=80)
            plt.close()

            # Save history and model
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
            metrics = tuner.evaluate_model(model)
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
            model.save(os.path.join(config_dir, 'model.h5'))

        with open(os.path.join(config_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  [{model_type.upper()}] {config['name']}: "
              f"Val Loss={metrics['val_loss']:.6f}, Test MAE={metrics['test_mae']:.6f}, "
              f"RMSE={metrics['test_rmse']:.6f}, Time={training_time:.1f}s")

    except Exception as e:
        print(f"  [{model_type.upper()}] {config['name']}: ERROR: {e}")
        result = {
            'config_name': config['name'],
            'config': config,
            'error': str(e)
        }
        with open(os.path.join(config_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    parser.add_argument('--model', choices=['lstm', 'cnn', 'gru', 'informer', 'all'], default='all',
                        help='Which model type to tune (default: all)')
    parser.add_argument('--config', type=str, default=None,
                        help='Run single config by name (e.g. d0.2_u16_lr0.001_b8_e50)')
    parser.add_argument('--index', type=str, default='SP500',
                        help='Index to tune on (default: SP500)')
    args = parser.parse_args()

    # Resolve data path from index
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

    # Single config mode (for parallel dispatch from pipeline)
    if args.config:
        run_single_config(data_path, args.model, args.config, index_name=args.index)
        sys.exit(0)

    # Full grid search mode
    tuner = FullGridSearchTuner(data_path, output_dir=os.path.join(STEP_DIR, 'results'), index_name=args.index)
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
