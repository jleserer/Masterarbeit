"""
Full Hyperparameter Grid Search for LSTM and CNN Models
=========================================================
Tests ALL possible combinations of hyperparameters.
Skips already completed configurations.

LSTM: 2 × 3 × 2 × 3 × 2 = 72 combinations
CNN:  2 × 2 × 2 × 3 × 2 = 48 combinations
Total: 120 combinations
"""

import os
import sys
import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import DataPreparator, create_sequences


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# LSTM parameters
LSTM_DROPOUT = [0.2, 0.4]
LSTM_DENSE_UNITS = [16, 32, 64]
LSTM_LEARNING_RATE = [0.001, 0.005]
LSTM_BATCH_SIZE = [8, 16, 32]
LSTM_EPOCHS = [50, 100]

# CNN parameters
CNN_KERNEL_SIZE = [3, 5]
CNN_POOL_SIZE = [2, 4]
CNN_DROPOUT = [0.2, 0.4]
CNN_BATCH_SIZE = [8, 16, 32]
CNN_EPOCHS = [50, 100]

# Fixed parameters
LOOKBACK_WINDOW = 60
LSTM_UNITS = [128, 64]
CONV_FILTERS = [64, 128, 256]
DENSE_UNITS_CNN = 64
LEARNING_RATE_CNN = 0.001


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


class FullGridSearchTuner:
    """Manages full grid search for LSTM and CNN models."""

    def __init__(self, data_path, output_dir='results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.lstm_dir = os.path.join(output_dir, 'LSTM', 'SP500')
        self.cnn_dir = os.path.join(output_dir, 'CNN', 'SP500')

        # Data containers
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        # Results
        self.lstm_results = []
        self.cnn_results = []

        # Create directories
        os.makedirs(self.lstm_dir, exist_ok=True)
        os.makedirs(self.cnn_dir, exist_ok=True)

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

    def train_model(self, model, config, verbose=0):
        """Train model and return history."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config['epochs'],
            batch_size=config['batch'],
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def evaluate_model(self, model):
        """Evaluate model and return metrics."""
        train_loss, train_mae = model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss, val_mae = model.evaluate(self.X_val, self.y_val, verbose=0)
        test_loss, test_mae = model.evaluate(self.X_test, self.y_test, verbose=0)

        return {
            'train_loss': float(train_loss),
            'train_mae': float(train_mae),
            'val_loss': float(val_loss),
            'val_mae': float(val_mae),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae)
        }

    def save_training_plot(self, history, config_name, model_type, output_dir):
        """Save training history plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'{model_type} - Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{model_type} - MAE')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_name}_history.png'), dpi=80)
        plt.close()

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

                self.save_training_plot(history, config['name'], 'LSTM', config_dir)
                model.save(os.path.join(config_dir, 'model.h5'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"Time: {training_time:.1f}s")

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

                self.save_training_plot(history, config['name'], 'CNN', config_dir)
                model.save(os.path.join(config_dir, 'model.h5'))

                with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  Val Loss: {metrics['val_loss']:.6f}, Test MAE: {metrics['test_mae']:.6f}, "
                      f"Time: {training_time:.1f}s")

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

    def find_best_configs(self):
        """Find and save the best configurations."""
        print("\n" + "=" * 70)
        print("FINDING BEST CONFIGURATIONS")
        print("=" * 70)

        # Find best LSTM
        valid_lstm = [r for r in self.lstm_results if 'metrics' in r]
        if valid_lstm:
            best_lstm = min(valid_lstm, key=lambda x: x['metrics']['val_loss'])
        else:
            best_lstm = None

        # Find best CNN
        valid_cnn = [r for r in self.cnn_results if 'metrics' in r]
        if valid_cnn:
            best_cnn = min(valid_cnn, key=lambda x: x['metrics']['val_loss'])
        else:
            best_cnn = None

        best_configs = {
            'timestamp': datetime.now().isoformat(),
            'total_lstm_configs': len(valid_lstm),
            'total_cnn_configs': len(valid_cnn),
        }

        if best_lstm:
            best_configs['best_lstm'] = {
                'config_name': best_lstm['config_name'],
                'config': best_lstm['config'],
                'metrics': best_lstm['metrics'],
                'training_time': best_lstm['training_time']
            }
            print("\n" + "-" * 70)
            print("BEST LSTM CONFIGURATION:")
            print("-" * 70)
            print(f"  Name: {best_lstm['config_name']}")
            print(f"  Dropout: {best_lstm['config']['dropout']}")
            print(f"  Dense Units: {best_lstm['config']['dense_units']}")
            print(f"  Learning Rate: {best_lstm['config']['lr']}")
            print(f"  Batch Size: {best_lstm['config']['batch']}")
            print(f"  Epochs: {best_lstm['config']['epochs']}")
            print(f"  Val Loss: {best_lstm['metrics']['val_loss']:.6f}")
            print(f"  Test Loss: {best_lstm['metrics']['test_loss']:.6f}")
            print(f"  Test MAE: {best_lstm['metrics']['test_mae']:.6f}")

        if best_cnn:
            best_configs['best_cnn'] = {
                'config_name': best_cnn['config_name'],
                'config': best_cnn['config'],
                'metrics': best_cnn['metrics'],
                'training_time': best_cnn['training_time']
            }
            print("\n" + "-" * 70)
            print("BEST CNN CONFIGURATION:")
            print("-" * 70)
            print(f"  Name: {best_cnn['config_name']}")
            print(f"  Kernel Size: {best_cnn['config']['kernel']}")
            print(f"  Pool Size: {best_cnn['config']['pool']}")
            print(f"  Dropout: {best_cnn['config']['dropout']}")
            print(f"  Batch Size: {best_cnn['config']['batch']}")
            print(f"  Epochs: {best_cnn['config']['epochs']}")
            print(f"  Val Loss: {best_cnn['metrics']['val_loss']:.6f}")
            print(f"  Test Loss: {best_cnn['metrics']['test_loss']:.6f}")
            print(f"  Test MAE: {best_cnn['metrics']['test_mae']:.6f}")

        # Save best configs
        with open(os.path.join(self.output_dir, 'best_configurations.json'), 'w') as f:
            json.dump(best_configs, f, indent=2)

        return best_configs

    def create_comparison_table(self):
        """Create a comparison table of all results."""
        print("\n" + "=" * 70)
        print("TOP 10 RESULTS PER MODEL")
        print("=" * 70)

        # LSTM Top 10
        valid_lstm = [r for r in self.lstm_results if 'metrics' in r]
        sorted_lstm = sorted(valid_lstm, key=lambda x: x['metrics']['val_loss'])[:10]

        print("\nLSTM TOP 10 (sorted by Val Loss):")
        print("-" * 100)
        print(f"{'Rank':<5} {'Config':<35} {'ValLoss':<12} {'TestMAE':<12} {'Time':<10}")
        print("-" * 100)

        for i, r in enumerate(sorted_lstm, 1):
            m = r['metrics']
            print(f"{i:<5} {r['config_name']:<35} {m['val_loss']:<12.6f} {m['test_mae']:<12.6f} {r['training_time']:<10.1f}s")

        # CNN Top 10
        valid_cnn = [r for r in self.cnn_results if 'metrics' in r]
        sorted_cnn = sorted(valid_cnn, key=lambda x: x['metrics']['val_loss'])[:10]

        print("\nCNN TOP 10 (sorted by Val Loss):")
        print("-" * 100)
        print(f"{'Rank':<5} {'Config':<35} {'ValLoss':<12} {'TestMAE':<12} {'Time':<10}")
        print("-" * 100)

        for i, r in enumerate(sorted_cnn, 1):
            m = r['metrics']
            print(f"{i:<5} {r['config_name']:<35} {m['val_loss']:<12.6f} {m['test_mae']:<12.6f} {r['training_time']:<10.1f}s")

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

        total_start = time.time()

        # Prepare data
        self.prepare_data()

        # Run tuning
        self.run_lstm_tuning()
        self.run_cnn_tuning()

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


if __name__ == "__main__":
    # Data path
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'stockData', 'preprocessedData', 'SP500_historical_data.csv'
    )

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Run full grid search
    tuner = FullGridSearchTuner(data_path, output_dir='results')
    tuner.run()
