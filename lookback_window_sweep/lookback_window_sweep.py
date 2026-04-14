"""
Lookback Window Sweep Analysis (L=1..60)
==========================================
Parameter sweep analysis of LSTM, CNN, GRU and Informer models across lookback window sizes L = 1..60.
Uses BEST configurations from hyperparameter tuning.

Generates:
1. MAE vs L plot (Parameter sweep visualization)
2. Prediction plots for best L value(s)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from data_preparation import DataPreparator, create_sequences


# =============================================================================
# BEST CONFIGURATIONS (from hyperparameter tuning)
# =============================================================================

BEST_LSTM_CONFIG = {
    'name': 'd0.4_u16_lr0.001_b16_e50',
    'dropout': 0.4,
    'dense_units': 16,
    'lr': 0.001,
    'batch': 16,
    'epochs': 50
}

BEST_CNN_CONFIG = {
    'name': 'k5_p4_d0.4_b8_e50',
    'kernel': 5,
    'pool': 4,
    'dropout': 0.4,
    'batch': 8,
    'epochs': 50
}

BEST_GRU_CONFIG = {
    'name': 'd0.4_u16_lr0.001_b16_e50',
    'dropout': 0.4,
    'dense_units': 16,
    'lr': 0.001,
    'batch': 16,
    'epochs': 50
}

BEST_INFORMER_CONFIG = {
    'name': 'dm64_h8_d0.05_lr0.0001_b32_e50',
    'd_model': 64,
    'n_heads': 8,
    'dropout': 0.05,
    'lr': 0.0001,
    'batch': 32,
    'epochs': 50
}

# Lookback windows to test: L = 1..60
LOOKBACK_WINDOWS = list(range(1, 61))

# Fixed LSTM/CNN/GRU architecture parameters
LSTM_UNITS = [128, 64]
GRU_UNITS = [128, 64]
CONV_FILTERS = [64, 128, 256]
DENSE_UNITS_CNN = 64


def build_lstm_model(lookback_window, n_features, config):
    """Build LSTM model with best configuration."""
    model = Sequential([
        Input(shape=(lookback_window, n_features)),
        LSTM(LSTM_UNITS[0], return_sequences=True, name='LSTM_1'),
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


def build_cnn_model(lookback_window, n_features, config):
    """Build CNN model with best configuration."""
    # Determine pool size based on lookback window
    min_lookback = config['pool'] * config['pool'] * 2
    pool_size = config['pool'] if lookback_window >= min_lookback else 1

    model = Sequential([
        Input(shape=(lookback_window, n_features)),
        Conv1D(CONV_FILTERS[0], kernel_size=config['kernel'], activation='relu', 
               padding='same', name='Conv1D_1'),
        MaxPooling1D(pool_size=pool_size, name='MaxPool_1'),
        Dropout(config['dropout']),
        Conv1D(CONV_FILTERS[1], kernel_size=config['kernel'], activation='relu', 
               padding='same', name='Conv1D_2'),
        MaxPooling1D(pool_size=pool_size, name='MaxPool_2'),
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

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_gru_model(lookback_window, n_features, config):
    """Build GRU model with best configuration."""
    model = Sequential([
        Input(shape=(lookback_window, n_features)),
        GRU(GRU_UNITS[0], return_sequences=True, name='GRU_1'),
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


def train_and_evaluate(model, X_train, y_train, X_test, y_test, config):
    """Train model and return test MAE."""
    # With percentage returns, val data has a similar distribution to training data.
    # Fixed epochs ensure fair comparison across all L values.
    model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch'],
        verbose=0
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    return test_mae, test_loss


def build_informer_sweep_model(lookback_window, n_features, config):
    """Build Informer model for sweep (PyTorch)."""
    from models.informer_model import build_informer
    informer_config = {**config, 'e_layers': 2, 'd_layers': 1, 'factor': 5}
    return build_informer(lookback_window, n_features, informer_config)


def train_and_evaluate_informer(model, X_train, y_train, X_test, y_test, config, lookback_window):
    """Train and evaluate Informer model. Returns (test_mae, test_loss)."""
    from models.informer_model import train_informer, evaluate_informer

    train_informer(model, X_train, y_train, config, lookback_window, verbose=0)
    test_loss, test_mae = evaluate_informer(model, X_test, y_test, lookback_window)

    return test_mae, test_loss


def evaluate_lookback_windows(data_path, output_dir):
    """Evaluate models for different lookback windows L=1..60."""
    print("=" * 70)
    print("COMPREHENSIVE LOOKBACK WINDOW (L) EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing L = {LOOKBACK_WINDOWS[0]} to {LOOKBACK_WINDOWS[-1]}")
    print(f"\nBest LSTM Config:     {BEST_LSTM_CONFIG['name']}")
    print(f"Best CNN Config:      {BEST_CNN_CONFIG['name']}")
    print(f"Best GRU Config:      {BEST_GRU_CONFIG['name']}")
    print(f"Best Informer Config: {BEST_INFORMER_CONFIG['name']}")
    print()

    # Load and prepare data once
    print("Loading and preparing data...")
    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    results = {
        'lookback_windows': LOOKBACK_WINDOWS,
        'lstm_mae': [],
        'cnn_mae': [],
        'gru_mae': [],
        'informer_mae': [],
        'lstm_loss': [],
        'cnn_loss': [],
        'gru_loss': [],
        'informer_loss': [],
        'lstm_rmse': [],
        'cnn_rmse': [],
        'gru_rmse': [],
        'informer_rmse': []
    }

    # Evaluate each lookback window
    for idx, L in enumerate(LOOKBACK_WINDOWS, 1):
        print(f"\r[{idx}/{len(LOOKBACK_WINDOWS)}] Testing L = {L:3d}...", end=" ", flush=True)

        try:
            # Create sequences
            X_train, y_train = create_sequences(train_data, L)
            X_test, y_test = create_sequences(test_data, L)

            n_features = X_train.shape[2]

            # Reset seeds before each L to ensure fair comparison
            np.random.seed(42)
            tf.random.set_seed(42)

            # LSTM
            lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
            lstm_mae, lstm_loss = train_and_evaluate(
                lstm_model, X_train, y_train, X_test, y_test,
                BEST_LSTM_CONFIG
            )
            results['lstm_mae'].append(lstm_mae)
            results['lstm_loss'].append(lstm_loss)
            results['lstm_rmse'].append(np.sqrt(lstm_loss))

            # CNN
            cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
            cnn_mae, cnn_loss = train_and_evaluate(
                cnn_model, X_train, y_train, X_test, y_test,
                BEST_CNN_CONFIG
            )
            results['cnn_mae'].append(cnn_mae)
            results['cnn_loss'].append(cnn_loss)
            results['cnn_rmse'].append(np.sqrt(cnn_loss))

            # GRU
            gru_model = build_gru_model(L, n_features, BEST_GRU_CONFIG)
            gru_mae, gru_loss = train_and_evaluate(
                gru_model, X_train, y_train, X_test, y_test,
                BEST_GRU_CONFIG
            )
            results['gru_mae'].append(gru_mae)
            results['gru_loss'].append(gru_loss)
            results['gru_rmse'].append(np.sqrt(gru_loss))

            # Clear TF models to free memory
            del lstm_model, cnn_model, gru_model
            tf.keras.backend.clear_session()

            # Informer (PyTorch)
            import torch
            torch.manual_seed(42)
            informer_model = build_informer_sweep_model(L, n_features, BEST_INFORMER_CONFIG)
            informer_mae, informer_loss = train_and_evaluate_informer(
                informer_model, X_train, y_train, X_test, y_test,
                BEST_INFORMER_CONFIG, L
            )
            results['informer_mae'].append(informer_mae)
            results['informer_loss'].append(informer_loss)
            results['informer_rmse'].append(np.sqrt(informer_loss))
            del informer_model

            print(f"LSTM MAE: {lstm_mae:.6f}, CNN MAE: {cnn_mae:.6f}, "
                  f"GRU MAE: {gru_mae:.6f}, Informer MAE: {informer_mae:.6f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results['lstm_mae'].append(np.nan)
            results['cnn_mae'].append(np.nan)
            results['gru_mae'].append(np.nan)
            results['informer_mae'].append(np.nan)
            results['lstm_loss'].append(np.nan)
            results['cnn_loss'].append(np.nan)
            results['gru_loss'].append(np.nan)
            results['informer_loss'].append(np.nan)
            results['lstm_rmse'].append(np.nan)
            results['cnn_rmse'].append(np.nan)
            results['gru_rmse'].append(np.nan)
            results['informer_rmse'].append(np.nan)

    print("\n")
    return results, preparator, train_data, val_data, test_data


def predict_single_L(data_path, output_dir, L):
    """Train all 4 models for a single L and save predictions as .npz."""
    import tensorflow as tf
    n_threads = int(os.environ.get('TF_WORKER_THREADS', '4'))
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    np.random.seed(42)
    tf.random.set_seed(42)

    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    X_train, y_train = create_sequences(train_data, L)
    X_test, y_test = create_sequences(test_data, L)
    n_features = X_train.shape[2]

    # LSTM
    lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
    lstm_model.fit(X_train, y_train, epochs=BEST_LSTM_CONFIG['epochs'],
                   batch_size=BEST_LSTM_CONFIG['batch'], verbose=0)
    lstm_predictions = lstm_model.predict(X_test, verbose=0).flatten()
    del lstm_model
    tf.keras.backend.clear_session()

    # CNN
    cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
    cnn_model.fit(X_train, y_train, epochs=BEST_CNN_CONFIG['epochs'],
                  batch_size=BEST_CNN_CONFIG['batch'], verbose=0)
    cnn_predictions = cnn_model.predict(X_test, verbose=0).flatten()
    del cnn_model
    tf.keras.backend.clear_session()

    # GRU
    gru_model = build_gru_model(L, n_features, BEST_GRU_CONFIG)
    gru_model.fit(X_train, y_train, epochs=BEST_GRU_CONFIG['epochs'],
                  batch_size=BEST_GRU_CONFIG['batch'], verbose=0)
    gru_predictions = gru_model.predict(X_test, verbose=0).flatten()
    del gru_model
    tf.keras.backend.clear_session()

    # Informer
    from models.informer_model import predict_informer, train_informer
    import torch
    torch.manual_seed(42)
    informer_model = build_informer_sweep_model(L, n_features, BEST_INFORMER_CONFIG)
    informer_config_full = {**BEST_INFORMER_CONFIG, 'e_layers': 2, 'd_layers': 1, 'factor': 5}
    train_informer(informer_model, X_train, y_train, informer_config_full, L, verbose=0)
    informer_predictions = predict_informer(informer_model, X_test, L)
    del informer_model

    # Inverse transform
    y_test_original = preparator.inverse_transform(
        y_test.reshape(-1, 1), split='test', lookback=L).flatten()
    lstm_pred_original = preparator.inverse_transform(
        lstm_predictions.reshape(-1, 1), split='test', lookback=L).flatten()
    cnn_pred_original = preparator.inverse_transform(
        cnn_predictions.reshape(-1, 1), split='test', lookback=L).flatten()
    gru_pred_original = preparator.inverse_transform(
        gru_predictions.reshape(-1, 1), split='test', lookback=L).flatten()
    informer_pred_original = preparator.inverse_transform(
        informer_predictions.reshape(-1, 1), split='test', lookback=L).flatten()

    # Save as .npz
    npz_path = os.path.join(output_dir, f'pred_L_{L:02d}.npz')
    np.savez(npz_path,
             y_actual_normalized=y_test,
             lstm_pred_normalized=lstm_predictions,
             cnn_pred_normalized=cnn_predictions,
             gru_pred_normalized=gru_predictions,
             informer_pred_normalized=informer_predictions,
             y_actual_original=y_test_original,
             lstm_pred_original=lstm_pred_original,
             cnn_pred_original=cnn_pred_original,
             gru_pred_original=gru_pred_original,
             informer_pred_original=informer_pred_original,
             test_indices=np.arange(L + 1, len(test_data) + 1))
    print(f"  pred L={L:2d} saved", flush=True)


def generate_predictions_multiple_L(preparator, train_data, val_data, test_data,
                                     best_L_values, output_dir,
                                     data_path=None, max_workers=32):
    """Generate predictions for multiple best L values (parallel via subprocesses)."""
    print(f"\n{'='*70}")
    print(f"GENERATING PREDICTIONS FOR BEST L VALUES: {best_L_values}")
    print(f"{'='*70}")

    # Try parallel subprocess dispatch if data_path is available
    if data_path is not None:
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        script = os.path.abspath(__file__)
        # Build env with reduced threads per worker
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        env['TF_WORKER_THREADS'] = '2'

        # Find best-configs path (passed via CLI)
        best_configs = env.get('BEST_CONFIGS_PATH', '')

        index_name = None
        for idx, csv in INDICES.items():
            if csv in data_path:
                index_name = idx
                break

        def run_predict_one(L):
            cmd = [sys.executable, '-u', script,
                   '--index', index_name, '--predict-one', str(L)]
            if best_configs:
                cmd += ['--best-configs', best_configs]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    env=env, cwd=PROJECT_ROOT)
            elapsed = time.time() - start
            status = 'OK' if result.returncode == 0 else 'FAIL'
            if status == 'FAIL':
                print(f"  FAIL pred L={L}: {result.stderr[-300:]}", flush=True)
            return L, status, elapsed

        print(f"Dispatching {len(best_L_values)} predictions with {max_workers} workers...",
              flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_predict_one, L): L for L in best_L_values}
            for future in as_completed(futures):
                L, status, elapsed = future.result()
                print(f"  L={L:2d} {status} ({elapsed:.0f}s)", flush=True)

    # Load results from .npz files
    predictions_data = {}
    for L in best_L_values:
        npz_path = os.path.join(output_dir, f'pred_L_{L:02d}.npz')
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            predictions_data[L] = {
                'y_actual_normalized': d['y_actual_normalized'],
                'lstm_pred_normalized': d['lstm_pred_normalized'],
                'cnn_pred_normalized': d['cnn_pred_normalized'],
                'gru_pred_normalized': d['gru_pred_normalized'],
                'informer_pred_normalized': d['informer_pred_normalized'],
                'y_actual_original': d['y_actual_original'],
                'lstm_pred_original': d['lstm_pred_original'],
                'cnn_pred_original': d['cnn_pred_original'],
                'gru_pred_original': d['gru_pred_original'],
                'informer_pred_original': d['informer_pred_original'],
                'test_indices': d['test_indices']
            }
            d.close()
            try:
                os.remove(npz_path)
            except PermissionError:
                pass  # Windows file lock — will be cleaned up later
        else:
            print(f"  WARNING: Missing predictions for L={L}")

    return predictions_data


def plot_mae_vs_lookback(results, output_dir):
    """Create MAE vs Lookback Window L plot."""
    print(f"\nPlotting MAE vs Lookback Window...")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(results['lookback_windows'], results['lstm_mae'],
            'b-o', linewidth=2.5, markersize=6, label='LSTM', alpha=0.8)
    ax.plot(results['lookback_windows'], results['cnn_mae'],
            'r-s', linewidth=2.5, markersize=6, label='CNN', alpha=0.8)
    ax.plot(results['lookback_windows'], results['gru_mae'],
            'g-^', linewidth=2.5, markersize=6, label='GRU', alpha=0.8)
    ax.plot(results['lookback_windows'], results['informer_mae'],
            'm-D', linewidth=2.5, markersize=6, label='Informer', alpha=0.8)

    # Find and mark best L values
    best_lstm_idx = np.nanargmin(results['lstm_mae'])
    best_cnn_idx = np.nanargmin(results['cnn_mae'])
    best_gru_idx = np.nanargmin(results['gru_mae'])
    best_informer_idx = np.nanargmin(results['informer_mae'])

    ax.plot(results['lookback_windows'][best_lstm_idx], results['lstm_mae'][best_lstm_idx],
            'bo', markersize=12, markeredgewidth=2, markerfacecolor='lightblue',
            markeredgecolor='darkblue', zorder=5)
    ax.plot(results['lookback_windows'][best_cnn_idx], results['cnn_mae'][best_cnn_idx],
            'rs', markersize=12, markeredgewidth=2, markerfacecolor='lightcoral',
            markeredgecolor='darkred', zorder=5)
    ax.plot(results['lookback_windows'][best_gru_idx], results['gru_mae'][best_gru_idx],
            'g^', markersize=12, markeredgewidth=2, markerfacecolor='lightgreen',
            markeredgecolor='darkgreen', zorder=5)
    ax.plot(results['lookback_windows'][best_informer_idx], results['informer_mae'][best_informer_idx],
            'mD', markersize=12, markeredgewidth=2, markerfacecolor='plum',
            markeredgecolor='purple', zorder=5)

    ax.set_xlabel('Lookback Window (L)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test MAE', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance vs Lookback Window Size (L=1..60)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(0, 65, 5))

    # Add annotations for best L
    best_lstm_l = results['lookback_windows'][best_lstm_idx]
    best_lstm_mae = results['lstm_mae'][best_lstm_idx]
    best_cnn_l = results['lookback_windows'][best_cnn_idx]
    best_cnn_mae = results['cnn_mae'][best_cnn_idx]

    ax.annotate(f'Best LSTM\nL={best_lstm_l}, MAE={best_lstm_mae:.6f}',
                xy=(best_lstm_l, best_lstm_mae), xytext=(-40, 30),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkblue', lw=1.5))

    ax.annotate(f'Best CNN\nL={best_cnn_l}, MAE={best_cnn_mae:.6f}',
                xy=(best_cnn_l, best_cnn_mae), xytext=(40, -40),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred', lw=1.5))

    best_gru_l = results['lookback_windows'][best_gru_idx]
    best_gru_mae = results['gru_mae'][best_gru_idx]

    ax.annotate(f'Best GRU\nL={best_gru_l}, MAE={best_gru_mae:.6f}',
                xy=(best_gru_l, best_gru_mae), xytext=(40, 30),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkgreen', lw=1.5))

    best_informer_l = results['lookback_windows'][best_informer_idx]
    best_informer_mae = results['informer_mae'][best_informer_idx]

    ax.annotate(f'Best Informer\nL={best_informer_l}, MAE={best_informer_mae:.6f}',
                xy=(best_informer_l, best_informer_mae), xytext=(-40, -40),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='purple', lw=1.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, '01_mae_vs_lookback_window.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    return (best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
            best_gru_l, best_gru_mae, best_informer_l, best_informer_mae)


def plot_predictions_vs_actual(predictions_data, output_dir, model_type='LSTM'):
    """Create prediction vs actual plots for different L values."""
    best_L_values = sorted(predictions_data.keys())

    # Plot time series (normalized scale)
    fig, axes = plt.subplots(len(best_L_values), 1, figsize=(15, 5 * len(best_L_values)))

    if len(best_L_values) == 1:
        axes = [axes]

    for idx, L in enumerate(best_L_values):
        data = predictions_data[L]
        y_actual = data['y_actual_normalized']
        pred_key = {'LSTM': 'lstm_pred_normalized', 'CNN': 'cnn_pred_normalized', 'GRU': 'gru_pred_normalized', 'INFORMER': 'informer_pred_normalized'}
        pred = data[pred_key[model_type]]
        color = {'LSTM': 'b', 'CNN': 'r', 'GRU': 'g', 'INFORMER': 'm'}[model_type]

        axes[idx].plot(y_actual, 'k-', alpha=0.7, linewidth=1.5, label='Actual', zorder=2)
        axes[idx].plot(pred, f'{color}-', alpha=0.6,
                      linewidth=1.2, label=f'{model_type} Predicted', zorder=1)

        axes[idx].set_xlabel('Time Step (Days)', fontsize=11)
        axes[idx].set_ylabel('Daily Return', fontsize=11)
        axes[idx].set_title(f'{model_type} Predictions vs Actual (L={L}, Returns)',
                           fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'02_{model_type}_predictions_vs_actual_timeseries.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot time series (original scale)
    fig, axes = plt.subplots(len(best_L_values), 1, figsize=(15, 5 * len(best_L_values)))

    if len(best_L_values) == 1:
        axes = [axes]

    for idx, L in enumerate(best_L_values):
        data = predictions_data[L]
        y_actual = data['y_actual_original']
        pred_key = {'LSTM': 'lstm_pred_original', 'CNN': 'cnn_pred_original', 'GRU': 'gru_pred_original', 'INFORMER': 'informer_pred_original'}
        pred = data[pred_key[model_type]]
        color = {'LSTM': 'b', 'CNN': 'r', 'GRU': 'g', 'INFORMER': 'm'}[model_type]

        axes[idx].plot(y_actual, 'k-', alpha=0.7, linewidth=1.5, label='Actual', zorder=2)
        axes[idx].plot(pred, f'{color}-', alpha=0.6,
                      linewidth=1.2, label=f'{model_type} Predicted', zorder=1)

        axes[idx].set_xlabel('Time Step (Days)', fontsize=11)
        axes[idx].set_ylabel('Close Price (Original Scale)', fontsize=11)
        axes[idx].set_title(f'{model_type} Predictions vs Actual (L={L}, Original Scale)', 
                           fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10, loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'03_{model_type}_predictions_vs_actual_original_scale.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot scatter plots (original scale)
    fig, axes = plt.subplots(1, len(best_L_values), figsize=(7 * len(best_L_values), 6))

    if len(best_L_values) == 1:
        axes = [axes]

    for idx, L in enumerate(best_L_values):
        data = predictions_data[L]
        y_actual = data['y_actual_original']
        pred_key = {'LSTM': 'lstm_pred_original', 'CNN': 'cnn_pred_original', 'GRU': 'gru_pred_original', 'INFORMER': 'informer_pred_original'}
        pred = data[pred_key[model_type]]
        color = {'LSTM': 'blue', 'CNN': 'red', 'GRU': 'green', 'INFORMER': 'magenta'}[model_type]

        axes[idx].scatter(y_actual, pred, alpha=0.5, s=20, c=color)

        min_val = min(y_actual.min(), pred.min())
        max_val = max(y_actual.max(), pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

        axes[idx].set_xlabel('Actual Close Price', fontsize=11)
        axes[idx].set_ylabel('Predicted Close Price', fontsize=11)
        axes[idx].set_title(f'{model_type}: Scatter (L={L})', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'04_{model_type}_scatter_predictions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def save_results(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                 best_gru_l, best_gru_mae, best_informer_l, best_informer_mae, output_dir):
    """Save comprehensive results to JSON file."""
    def _safe_list(key):
        return [float(x) if not np.isnan(x) else None for x in results.get(key, [])]

    # Compute RMSE from loss if not already in results
    def _rmse_list(loss_key, rmse_key):
        if rmse_key in results and results[rmse_key]:
            return _safe_list(rmse_key)
        return [float(np.sqrt(x)) if not np.isnan(x) else None for x in results.get(loss_key, [])]

    results_json = {
        'timestamp': datetime.now().isoformat(),
        'best_lstm_config': BEST_LSTM_CONFIG,
        'best_cnn_config': BEST_CNN_CONFIG,
        'best_gru_config': BEST_GRU_CONFIG,
        'best_informer_config': BEST_INFORMER_CONFIG,
        'lookback_windows': results['lookback_windows'],
        'lstm_mae': _safe_list('lstm_mae'),
        'cnn_mae': _safe_list('cnn_mae'),
        'gru_mae': _safe_list('gru_mae'),
        'informer_mae': _safe_list('informer_mae'),
        'lstm_loss': _safe_list('lstm_loss'),
        'cnn_loss': _safe_list('cnn_loss'),
        'gru_loss': _safe_list('gru_loss'),
        'informer_loss': _safe_list('informer_loss'),
        'lstm_rmse': _rmse_list('lstm_loss', 'lstm_rmse'),
        'cnn_rmse': _rmse_list('cnn_loss', 'cnn_rmse'),
        'gru_rmse': _rmse_list('gru_loss', 'gru_rmse'),
        'informer_rmse': _rmse_list('informer_loss', 'informer_rmse'),
        'best_l_lstm': {
            'l_value': int(best_lstm_l),
            'mae': float(best_lstm_mae),
            'rmse': float(np.sqrt(results['lstm_loss'][np.nanargmin(results['lstm_mae'])]))
        },
        'best_l_cnn': {
            'l_value': int(best_cnn_l),
            'mae': float(best_cnn_mae),
            'rmse': float(np.sqrt(results['cnn_loss'][np.nanargmin(results['cnn_mae'])]))
        },
        'best_l_gru': {
            'l_value': int(best_gru_l),
            'mae': float(best_gru_mae),
            'rmse': float(np.sqrt(results['gru_loss'][np.nanargmin(results['gru_mae'])]))
        },
        'best_l_informer': {
            'l_value': int(best_informer_l),
            'mae': float(best_informer_mae),
            'rmse': float(np.sqrt(results['informer_loss'][np.nanargmin(results['informer_mae'])]))
        }
    }

    results_path = os.path.join(output_dir, 'lookback_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {results_path}")


def print_summary(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                  best_gru_l, best_gru_mae, best_informer_l, best_informer_mae):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    best_lstm_rmse = np.sqrt(results['lstm_loss'][np.nanargmin(results['lstm_mae'])])
    best_cnn_rmse = np.sqrt(results['cnn_loss'][np.nanargmin(results['cnn_mae'])])
    best_gru_rmse = np.sqrt(results['gru_loss'][np.nanargmin(results['gru_mae'])])
    best_informer_rmse = np.sqrt(results['informer_loss'][np.nanargmin(results['informer_mae'])])

    print(f"\nBest LSTM L: {best_lstm_l}")
    print(f"  Test MAE: {best_lstm_mae:.6f}, RMSE: {best_lstm_rmse:.6f}")
    print(f"  Config: {BEST_LSTM_CONFIG['name']}")

    print(f"\nBest CNN L: {best_cnn_l}")
    print(f"  Test MAE: {best_cnn_mae:.6f}, RMSE: {best_cnn_rmse:.6f}")
    print(f"  Config: {BEST_CNN_CONFIG['name']}")

    print(f"\nBest GRU L: {best_gru_l}")
    print(f"  Test MAE: {best_gru_mae:.6f}, RMSE: {best_gru_rmse:.6f}")
    print(f"  Config: {BEST_GRU_CONFIG['name']}")

    print(f"\nBest Informer L: {best_informer_l}")
    print(f"  Test MAE: {best_informer_mae:.6f}, RMSE: {best_informer_rmse:.6f}")
    print(f"  Config: {BEST_INFORMER_CONFIG['name']}")

    print(f"\nTop 5 LSTM Results:")
    print("-" * 60)
    lstm_mae_sorted = sorted(enumerate(results['lstm_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(lstm_mae_sorted[:5], 1):
        rmse = np.sqrt(results['lstm_loss'][idx])
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    print(f"\nTop 5 CNN Results:")
    print("-" * 60)
    cnn_mae_sorted = sorted(enumerate(results['cnn_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(cnn_mae_sorted[:5], 1):
        rmse = np.sqrt(results['cnn_loss'][idx])
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    print(f"\nTop 5 GRU Results:")
    print("-" * 60)
    gru_mae_sorted = sorted(enumerate(results['gru_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(gru_mae_sorted[:5], 1):
        rmse = np.sqrt(results['gru_loss'][idx])
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    print(f"\nTop 5 Informer Results:")
    print("-" * 60)
    informer_mae_sorted = sorted(enumerate(results['informer_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(informer_mae_sorted[:5], 1):
        rmse = np.sqrt(results['informer_loss'][idx])
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}, RMSE={rmse:.6f}")


def run_single_lookback(data_path, output_dir, L):
    """Train LSTM+CNN for a single lookback window L. Save result to L_{L:02d}.json."""
    import tensorflow as tf
    n_threads = int(os.environ.get('TF_WORKER_THREADS', '4'))
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    np.random.seed(42)
    tf.random.set_seed(42)

    preparator = DataPreparator(data_path)
    train_data, _, test_data = preparator.load_and_prepare()

    result = {'L': L, 'lstm_mae': None, 'lstm_loss': None, 'lstm_rmse': None,
              'cnn_mae': None, 'cnn_loss': None, 'cnn_rmse': None,
              'gru_mae': None, 'gru_loss': None, 'gru_rmse': None,
              'informer_mae': None, 'informer_loss': None, 'informer_rmse': None}

    try:
        X_train, y_train = create_sequences(train_data, L)
        X_test, y_test = create_sequences(test_data, L)
        n_features = X_train.shape[2]

        np.random.seed(42)
        tf.random.set_seed(42)

        # LSTM
        lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
        lstm_mae, lstm_loss = train_and_evaluate(
            lstm_model, X_train, y_train, X_test, y_test, BEST_LSTM_CONFIG
        )
        result['lstm_mae'] = float(lstm_mae)
        result['lstm_loss'] = float(lstm_loss)
        result['lstm_rmse'] = float(np.sqrt(lstm_loss))
        del lstm_model
        tf.keras.backend.clear_session()

        # CNN
        cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
        cnn_mae, cnn_loss = train_and_evaluate(
            cnn_model, X_train, y_train, X_test, y_test, BEST_CNN_CONFIG
        )
        result['cnn_mae'] = float(cnn_mae)
        result['cnn_loss'] = float(cnn_loss)
        result['cnn_rmse'] = float(np.sqrt(cnn_loss))
        del cnn_model
        tf.keras.backend.clear_session()

        # GRU
        gru_model = build_gru_model(L, n_features, BEST_GRU_CONFIG)
        gru_mae, gru_loss = train_and_evaluate(
            gru_model, X_train, y_train, X_test, y_test, BEST_GRU_CONFIG
        )
        result['gru_mae'] = float(gru_mae)
        result['gru_loss'] = float(gru_loss)
        result['gru_rmse'] = float(np.sqrt(gru_loss))
        del gru_model
        tf.keras.backend.clear_session()

        # Informer (PyTorch)
        import torch
        torch.manual_seed(42)
        informer_model = build_informer_sweep_model(L, n_features, BEST_INFORMER_CONFIG)
        informer_mae, informer_loss = train_and_evaluate_informer(
            informer_model, X_train, y_train, X_test, y_test,
            BEST_INFORMER_CONFIG, L
        )
        result['informer_mae'] = float(informer_mae)
        result['informer_loss'] = float(informer_loss)
        result['informer_rmse'] = float(np.sqrt(informer_loss))
        del informer_model

    except Exception as e:
        print(f"ERROR L={L}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f'L_{L:02d}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  L={L:2d}: LSTM MAE={result['lstm_mae']}, RMSE={result['lstm_rmse']}, "
          f"CNN MAE={result['cnn_mae']}, RMSE={result['cnn_rmse']}, "
          f"GRU MAE={result['gru_mae']}, RMSE={result['gru_rmse']}, "
          f"Inf MAE={result['informer_mae']}, RMSE={result['informer_rmse']}")


def collect_and_plot(data_path, output_dir):
    """Collect all L_*.json results, generate plots and predictions."""
    import tensorflow as tf

    print(f"Collecting results from {output_dir}...")

    # Read all individual L results
    results = {
        'lookback_windows': LOOKBACK_WINDOWS,
        'lstm_mae': [],
        'cnn_mae': [],
        'gru_mae': [],
        'lstm_loss': [],
        'cnn_loss': [],
        'gru_loss': [],
        'lstm_rmse': [],
        'cnn_rmse': [],
        'gru_rmse': [],
        'informer_mae': [],
        'informer_loss': [],
        'informer_rmse': []
    }

    for L in LOOKBACK_WINDOWS:
        result_path = os.path.join(output_dir, f'L_{L:02d}.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                r = json.load(f)
            results['lstm_mae'].append(r['lstm_mae'] if r['lstm_mae'] is not None else np.nan)
            results['cnn_mae'].append(r['cnn_mae'] if r['cnn_mae'] is not None else np.nan)
            results['gru_mae'].append(r.get('gru_mae') if r.get('gru_mae') is not None else np.nan)
            results['informer_mae'].append(r.get('informer_mae') if r.get('informer_mae') is not None else np.nan)
            results['lstm_loss'].append(r['lstm_loss'] if r['lstm_loss'] is not None else np.nan)
            results['cnn_loss'].append(r['cnn_loss'] if r['cnn_loss'] is not None else np.nan)
            results['gru_loss'].append(r.get('gru_loss') if r.get('gru_loss') is not None else np.nan)
            results['informer_loss'].append(r.get('informer_loss') if r.get('informer_loss') is not None else np.nan)
            # RMSE: read from JSON or compute from loss
            for model in ['lstm', 'cnn', 'gru', 'informer']:
                rmse_val = r.get(f'{model}_rmse')
                if rmse_val is not None:
                    results[f'{model}_rmse'].append(rmse_val)
                else:
                    loss_val = r.get(f'{model}_loss')
                    results[f'{model}_rmse'].append(np.sqrt(loss_val) if loss_val is not None else np.nan)
        else:
            print(f"  WARNING: Missing L={L}")
            results['lstm_mae'].append(np.nan)
            results['cnn_mae'].append(np.nan)
            results['gru_mae'].append(np.nan)
            results['informer_mae'].append(np.nan)
            results['lstm_loss'].append(np.nan)
            results['cnn_loss'].append(np.nan)
            results['gru_loss'].append(np.nan)
            results['informer_loss'].append(np.nan)
            results['lstm_rmse'].append(np.nan)
            results['cnn_rmse'].append(np.nan)
            results['gru_rmse'].append(np.nan)
            results['informer_rmse'].append(np.nan)

    valid_count = sum(1 for x in results['lstm_mae'] if not np.isnan(x))
    print(f"  Collected {valid_count}/{len(LOOKBACK_WINDOWS)} L-values")

    # Plot MAE vs Lookback Window
    (best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
     best_gru_l, best_gru_mae, best_informer_l, best_informer_mae) = plot_mae_vs_lookback(results, output_dir)

    # Determine best L values (top 3)
    lstm_mae_sorted = sorted(
        [(i, m) for i, m in enumerate(results['lstm_mae']) if not np.isnan(m)],
        key=lambda x: x[1]
    )
    cnn_mae_sorted = sorted(
        [(i, m) for i, m in enumerate(results['cnn_mae']) if not np.isnan(m)],
        key=lambda x: x[1]
    )
    gru_mae_sorted = sorted(
        [(i, m) for i, m in enumerate(results['gru_mae']) if not np.isnan(m)],
        key=lambda x: x[1]
    )
    informer_mae_sorted = sorted(
        [(i, m) for i, m in enumerate(results['informer_mae']) if not np.isnan(m)],
        key=lambda x: x[1]
    )

    best_lstm_L_values = [results['lookback_windows'][idx] for idx, _ in lstm_mae_sorted[:3]]
    best_cnn_L_values  = [results['lookback_windows'][idx] for idx, _ in cnn_mae_sorted[:3]]
    best_gru_L_values  = [results['lookback_windows'][idx] for idx, _ in gru_mae_sorted[:3]]
    best_informer_L_values = [results['lookback_windows'][idx] for idx, _ in informer_mae_sorted[:3]]

    print(f"\nBest 3 L values for LSTM:     {best_lstm_L_values}")
    print(f"Best 3 L values for CNN:      {best_cnn_L_values}")
    print(f"Best 3 L values for GRU:      {best_gru_L_values}")
    print(f"Best 3 L values for Informer: {best_informer_L_values}")

    # Load data for predictions
    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    # Generate predictions for best L values (union of all best L values)
    all_best_L = sorted(set(best_lstm_L_values + best_cnn_L_values +
                            best_gru_L_values + best_informer_L_values))
    all_predictions = generate_predictions_multiple_L(preparator, train_data, val_data, test_data,
                                                      all_best_L, output_dir,
                                                      data_path=data_path)

    # Create prediction plots
    print(f"\nGenerating plots...")
    plot_predictions_vs_actual(all_predictions, output_dir, 'LSTM')
    plot_predictions_vs_actual(all_predictions, output_dir, 'CNN')
    plot_predictions_vs_actual(all_predictions, output_dir, 'GRU')
    plot_predictions_vs_actual(all_predictions, output_dir, 'INFORMER')

    # Save results
    save_results(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                 best_gru_l, best_gru_mae, best_informer_l, best_informer_mae, output_dir)

    # Print summary
    print_summary(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                  best_gru_l, best_gru_mae, best_informer_l, best_informer_mae)

    # Clean up individual L files
    for L in LOOKBACK_WINDOWS:
        lf = os.path.join(output_dir, f'L_{L:02d}.json')
        if os.path.exists(lf):
            os.remove(lf)
    print("  Cleaned up individual L_*.json files.")


# All indices available
INDICES = {
    'SP500':    'SP500_historical_data.csv',
    'DAX':      'DAX_historical_data.csv',
    'NASDAQ':   'NASDAQ_historical_data.csv',
    'FTSE100':  'FTSE100_historical_data.csv',
    'HANG_SENG':'HANG_SENG_historical_data.csv',
    'NIKKEI':   'NIKKEI_historical_data.csv',
    '10Y_Bond': '10-Year Bond_historical_data.csv',
    '30Y_Bond': '30 Year Bond_historical_data.csv',
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Lookback Window Sweep')
    parser.add_argument('--index', type=str, default=None,
                        help='Single index to evaluate (e.g. SP500, DAX). Default: all')
    parser.add_argument('--lookback', type=int, default=None,
                        help='Single lookback window L to train (for parallel dispatch)')
    parser.add_argument('--collect', action='store_true',
                        help='Collect individual L results and generate plots')
    parser.add_argument('--predict-one', type=int, default=None,
                        help='Train all models for a single L and save predictions as .npz')
    parser.add_argument('--best-configs', type=str, default=None,
                        help='Path to best_configurations.json with per-index configs')
    args = parser.parse_args()

    # Override global configs from per-index best_configurations.json
    if args.best_configs and args.index:
        try:
            with open(args.best_configs, 'r') as f:
                all_best = json.load(f)
            if args.index in all_best:
                idx_best = all_best[args.index]
                if 'best_lstm' in idx_best:
                    cfg = idx_best['best_lstm']['config']
                    BEST_LSTM_CONFIG.clear()
                    BEST_LSTM_CONFIG['name'] = idx_best['best_lstm']['config_name']
                    BEST_LSTM_CONFIG.update({k: v for k, v in cfg.items() if k != 'name'})
                if 'best_cnn' in idx_best:
                    cfg = idx_best['best_cnn']['config']
                    BEST_CNN_CONFIG.clear()
                    BEST_CNN_CONFIG['name'] = idx_best['best_cnn']['config_name']
                    BEST_CNN_CONFIG.update({k: v for k, v in cfg.items() if k != 'name'})
                if 'best_gru' in idx_best:
                    cfg = idx_best['best_gru']['config']
                    BEST_GRU_CONFIG.clear()
                    BEST_GRU_CONFIG['name'] = idx_best['best_gru']['config_name']
                    BEST_GRU_CONFIG.update({k: v for k, v in cfg.items() if k != 'name'})
                if 'best_informer' in idx_best:
                    cfg = idx_best['best_informer']['config']
                    BEST_INFORMER_CONFIG.clear()
                    BEST_INFORMER_CONFIG['name'] = idx_best['best_informer']['config_name']
                    BEST_INFORMER_CONFIG.update({k: v for k, v in cfg.items() if k != 'name'})
                print(f"Loaded per-index configs for {args.index}: "
                      f"LSTM={BEST_LSTM_CONFIG['name']}, CNN={BEST_CNN_CONFIG['name']}, "
                      f"GRU={BEST_GRU_CONFIG['name']}, Informer={BEST_INFORMER_CONFIG['name']}")
        except Exception as e:
            print(f"Warning: Could not load per-index configs: {e}. Using defaults.")

    data_dir = os.path.join(PROJECT_ROOT, 'stockData', 'preprocessedData')
    results_base = os.path.join(STEP_DIR, 'results')

    # Single lookback mode: train one L for one index
    if args.lookback is not None:
        if not args.index:
            print("Error: --lookback requires --index")
            sys.exit(1)
        if args.index not in INDICES:
            print(f"Error: Unknown index '{args.index}'. Available: {list(INDICES.keys())}")
            sys.exit(1)
        data_path = os.path.join(data_dir, INDICES[args.index])
        output_dir = os.path.join(results_base, args.index)
        run_single_lookback(data_path, output_dir, args.lookback)
        sys.exit(0)

    # Predict-one mode: train all models for one L and save .npz
    if args.predict_one is not None:
        if not args.index:
            print("Error: --predict-one requires --index")
            sys.exit(1)
        if args.index not in INDICES:
            print(f"Error: Unknown index '{args.index}'. Available: {list(INDICES.keys())}")
            sys.exit(1)
        data_path = os.path.join(data_dir, INDICES[args.index])
        output_dir = os.path.join(results_base, args.index)
        os.makedirs(output_dir, exist_ok=True)
        predict_single_L(data_path, output_dir, args.predict_one)
        sys.exit(0)

    # Collect mode: assemble results and generate plots
    if args.collect:
        if not args.index:
            print("Error: --collect requires --index")
            sys.exit(1)
        if args.index not in INDICES:
            print(f"Error: Unknown index '{args.index}'. Available: {list(INDICES.keys())}")
            sys.exit(1)
        data_path = os.path.join(data_dir, INDICES[args.index])
        output_dir = os.path.join(results_base, args.index)
        collect_and_plot(data_path, output_dir)
        sys.exit(0)

    # Default: full sequential mode (legacy)
    if args.index:
        if args.index not in INDICES:
            print(f"Error: Unknown index '{args.index}'. Available: {list(INDICES.keys())}")
            sys.exit(1)
        indices_to_run = {args.index: INDICES[args.index]}
    else:
        indices_to_run = INDICES

    for index_name, filename in indices_to_run.items():
        print(f"\n{'#'*70}")
        print(f"# INDEX: {index_name}")
        print(f"{'#'*70}")

        data_path = os.path.join(data_dir, filename)
        output_dir = os.path.join(results_base, index_name)
        os.makedirs(output_dir, exist_ok=True)

        # Run evaluation
        results, preparator, train_data, val_data, test_data = evaluate_lookback_windows(data_path, output_dir)

        # Plot MAE vs Lookback Window
        (best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
     best_gru_l, best_gru_mae, best_informer_l, best_informer_mae) = plot_mae_vs_lookback(results, output_dir)

        # Determine best L values (top 3)
        lstm_mae_sorted = sorted(enumerate(results['lstm_mae']), key=lambda x: x[1])
        cnn_mae_sorted  = sorted(enumerate(results['cnn_mae']),  key=lambda x: x[1])
        gru_mae_sorted  = sorted(enumerate(results['gru_mae']),  key=lambda x: x[1])
        informer_mae_sorted = sorted(enumerate(results['informer_mae']), key=lambda x: x[1])

        best_lstm_L_values = [results['lookback_windows'][idx] for idx, _ in lstm_mae_sorted[:3]]
        best_cnn_L_values  = [results['lookback_windows'][idx] for idx, _ in cnn_mae_sorted[:3]]
        best_gru_L_values  = [results['lookback_windows'][idx] for idx, _ in gru_mae_sorted[:3]]
        best_informer_L_values = [results['lookback_windows'][idx] for idx, _ in informer_mae_sorted[:3]]

        print(f"\nBest 3 L values for LSTM:     {best_lstm_L_values}")
        print(f"Best 3 L values for CNN:      {best_cnn_L_values}")
        print(f"Best 3 L values for GRU:      {best_gru_L_values}")
        print(f"Best 3 L values for Informer: {best_informer_L_values}")

        # Generate predictions for best L values (union of all)
        all_best_L = sorted(set(best_lstm_L_values + best_cnn_L_values +
                                best_gru_L_values + best_informer_L_values))
        all_predictions = generate_predictions_multiple_L(preparator, train_data, val_data, test_data,
                                                          all_best_L, output_dir)

        # Create prediction plots
        print(f"\nGenerating plots...")
        plot_predictions_vs_actual(all_predictions, output_dir, 'LSTM')
        plot_predictions_vs_actual(all_predictions, output_dir, 'CNN')
        plot_predictions_vs_actual(all_predictions, output_dir, 'GRU')

        # Save results
        save_results(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                     best_gru_l, best_gru_mae, best_informer_l, best_informer_mae, output_dir)

        # Print summary
        print_summary(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae,
                      best_gru_l, best_gru_mae, best_informer_l, best_informer_mae)

        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE: {index_name}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}")
