"""
Lookback Window Sweep Analysis (L=1..60)
==========================================
Parameter sweep analysis of LSTM and CNN models across lookback window sizes L = 1..60.
Uses BEST configurations from 24h hyperparameter tuning.

Generates:
1. MAE vs L plot (Parameter sweep visualization)
2. Prediction plots for best L value(s)

LSTM Best: d0.2_u64_lr0.005_b8_e100 (L=46, Test MAE: 0.011312)
CNN Best:  k5_p2_d0.2_b16_e100     (L=7, Test MAE: 0.010875)
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import DataPreparator, create_sequences


# =============================================================================
# BEST CONFIGURATIONS (from hyperparameter tuning)
# =============================================================================

BEST_LSTM_CONFIG = {
    'name': 'd0.2_u64_lr0.005_b8_e100',
    'dropout': 0.2,
    'dense_units': 64,
    'lr': 0.005,
    'batch': 8,
    'epochs': 100
}

BEST_CNN_CONFIG = {
    'name': 'k5_p2_d0.2_b16_e100',
    'kernel': 5,
    'pool': 2,
    'dropout': 0.2,
    'batch': 16,
    'epochs': 100
}

# Lookback windows to test: L = 1..60
LOOKBACK_WINDOWS = list(range(1, 61))

# Fixed LSTM/CNN architecture parameters
LSTM_UNITS = [128, 64]
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


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, config, model_type):
    """Train model and return test MAE."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch'],
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    return test_mae, test_loss


def evaluate_lookback_windows(data_path, output_dir):
    """Evaluate both models for different lookback windows L=1..60."""
    print("=" * 70)
    print("COMPREHENSIVE LOOKBACK WINDOW (L) EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing L = {LOOKBACK_WINDOWS[0]} to {LOOKBACK_WINDOWS[-1]}")
    print(f"\nBest LSTM Config:  {BEST_LSTM_CONFIG['name']}")
    print(f"Best CNN Config:   {BEST_CNN_CONFIG['name']}")
    print()

    # Load and prepare data once
    print("Loading and preparing data...")
    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    results = {
        'lookback_windows': LOOKBACK_WINDOWS,
        'lstm_mae': [],
        'cnn_mae': [],
        'lstm_loss': [],
        'cnn_loss': []
    }

    # Evaluate each lookback window
    for idx, L in enumerate(LOOKBACK_WINDOWS, 1):
        print(f"\r[{idx}/{len(LOOKBACK_WINDOWS)}] Testing L = {L:3d}...", end=" ", flush=True)

        try:
            # Create sequences
            X_train, y_train = create_sequences(train_data, L)
            X_val, y_val = create_sequences(val_data, L)
            X_test, y_test = create_sequences(test_data, L)

            n_features = X_train.shape[2]

            # LSTM
            lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
            lstm_mae, lstm_loss = train_and_evaluate(
                lstm_model, X_train, y_train, X_val, y_val, X_test, y_test,
                BEST_LSTM_CONFIG, 'LSTM'
            )
            results['lstm_mae'].append(lstm_mae)
            results['lstm_loss'].append(lstm_loss)

            # CNN
            cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
            cnn_mae, cnn_loss = train_and_evaluate(
                cnn_model, X_train, y_train, X_val, y_val, X_test, y_test,
                BEST_CNN_CONFIG, 'CNN'
            )
            results['cnn_mae'].append(cnn_mae)
            results['cnn_loss'].append(cnn_loss)

            # Clear models to free memory
            del lstm_model, cnn_model
            import tensorflow as tf
            tf.keras.backend.clear_session()

            print(f"LSTM MAE: {lstm_mae:.6f}, CNN MAE: {cnn_mae:.6f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results['lstm_mae'].append(np.nan)
            results['cnn_mae'].append(np.nan)
            results['lstm_loss'].append(np.nan)
            results['cnn_loss'].append(np.nan)

    print("\n")
    return results, preparator, train_data, val_data, test_data


def generate_predictions_multiple_L(preparator, train_data, val_data, test_data, 
                                     best_L_values, output_dir):
    """Generate predictions for multiple best L values."""
    print(f"\n{'='*70}")
    print(f"GENERATING PREDICTIONS FOR BEST L VALUES: {best_L_values}")
    print(f"{'='*70}")

    predictions_data = {}

    for L in best_L_values:
        print(f"\nProcessing L = {L}...", end=" ", flush=True)

        # Create sequences
        X_train, y_train = create_sequences(train_data, L)
        X_val, y_val = create_sequences(val_data, L)
        X_test, y_test = create_sequences(test_data, L)

        n_features = X_train.shape[2]

        # Train LSTM
        lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]
        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=BEST_LSTM_CONFIG['epochs'],
            batch_size=BEST_LSTM_CONFIG['batch'],
            callbacks=callbacks,
            verbose=0
        )
        lstm_predictions = lstm_model.predict(X_test, verbose=0).flatten()

        # Train CNN
        cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
        cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=BEST_CNN_CONFIG['epochs'],
            batch_size=BEST_CNN_CONFIG['batch'],
            callbacks=callbacks,
            verbose=0
        )
        cnn_predictions = cnn_model.predict(X_test, verbose=0).flatten()

        # Inverse transform to original scale
        y_test_original = preparator.inverse_transform(y_test.reshape(-1, 1)).flatten()
        lstm_pred_original = preparator.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
        cnn_pred_original = preparator.inverse_transform(cnn_predictions.reshape(-1, 1)).flatten()

        predictions_data[L] = {
            'y_actual_normalized': y_test,
            'lstm_pred_normalized': lstm_predictions,
            'cnn_pred_normalized': cnn_predictions,
            'y_actual_original': y_test_original,
            'lstm_pred_original': lstm_pred_original,
            'cnn_pred_original': cnn_pred_original,
            'test_indices': np.arange(L + 1, len(test_data) + 1)
        }

        # Clear models
        del lstm_model, cnn_model
        import tensorflow as tf
        tf.keras.backend.clear_session()

        print(f"Done!")

    return predictions_data


def plot_mae_vs_lookback(results, output_dir):
    """Create MAE vs Lookback Window L plot."""
    print(f"\nPlotting MAE vs Lookback Window...")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(results['lookback_windows'], results['lstm_mae'],
            'b-o', linewidth=2.5, markersize=6, label='LSTM', alpha=0.8)
    ax.plot(results['lookback_windows'], results['cnn_mae'],
            'r-s', linewidth=2.5, markersize=6, label='CNN', alpha=0.8)

    # Find and mark best L values
    best_lstm_idx = np.nanargmin(results['lstm_mae'])
    best_cnn_idx = np.nanargmin(results['cnn_mae'])

    ax.plot(results['lookback_windows'][best_lstm_idx], results['lstm_mae'][best_lstm_idx],
            'bo', markersize=12, markeredgewidth=2, markerfacecolor='lightblue',
            markeredgecolor='darkblue', zorder=5)
    ax.plot(results['lookback_windows'][best_cnn_idx], results['cnn_mae'][best_cnn_idx],
            'rs', markersize=12, markeredgewidth=2, markerfacecolor='lightcoral',
            markeredgecolor='darkred', zorder=5)

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

    plt.tight_layout()
    plot_path = os.path.join(output_dir, '01_mae_vs_lookback_window.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    return best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae


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
        pred = data['lstm_pred_normalized'] if model_type == 'LSTM' else data['cnn_pred_normalized']

        axes[idx].plot(y_actual, 'k-', alpha=0.7, linewidth=1.5, label='Actual', zorder=2)
        axes[idx].plot(pred, f'{"b" if model_type == "LSTM" else "r"}-', alpha=0.6, 
                      linewidth=1.2, label=f'{model_type} Predicted', zorder=1)

        axes[idx].set_xlabel('Time Step (Days)', fontsize=11)
        axes[idx].set_ylabel('Normalized Close Price', fontsize=11)
        axes[idx].set_title(f'{model_type} Predictions vs Actual (L={L}, Normalized Scale)', 
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
        pred = data['lstm_pred_original'] if model_type == 'LSTM' else data['cnn_pred_original']

        axes[idx].plot(y_actual, 'k-', alpha=0.7, linewidth=1.5, label='Actual', zorder=2)
        axes[idx].plot(pred, f'{"b" if model_type == "LSTM" else "r"}-', alpha=0.6, 
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
        pred = data['lstm_pred_original'] if model_type == 'LSTM' else data['cnn_pred_original']

        axes[idx].scatter(y_actual, pred, alpha=0.5, s=20, 
                         c='blue' if model_type == 'LSTM' else 'red')

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


def save_results(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae, output_dir):
    """Save comprehensive results to JSON file."""
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'best_lstm_config': BEST_LSTM_CONFIG,
        'best_cnn_config': BEST_CNN_CONFIG,
        'lookback_windows': results['lookback_windows'],
        'lstm_mae': [float(x) if not np.isnan(x) else None for x in results['lstm_mae']],
        'cnn_mae': [float(x) if not np.isnan(x) else None for x in results['cnn_mae']],
        'lstm_loss': [float(x) if not np.isnan(x) else None for x in results['lstm_loss']],
        'cnn_loss': [float(x) if not np.isnan(x) else None for x in results['cnn_loss']],
        'best_l_lstm': {
            'l_value': int(best_lstm_l),
            'mae': float(best_lstm_mae)
        },
        'best_l_cnn': {
            'l_value': int(best_cnn_l),
            'mae': float(best_cnn_mae)
        }
    }

    results_path = os.path.join(output_dir, 'lookback_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {results_path}")


def print_summary(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nBest LSTM L: {best_lstm_l}")
    print(f"  Test MAE: {best_lstm_mae:.6f}")
    print(f"  Config: {BEST_LSTM_CONFIG['name']}")

    print(f"\nBest CNN L: {best_cnn_l}")
    print(f"  Test MAE: {best_cnn_mae:.6f}")
    print(f"  Config: {BEST_CNN_CONFIG['name']}")

    print(f"\nTop 5 LSTM Results:")
    print("-" * 50)
    lstm_mae_sorted = sorted(enumerate(results['lstm_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(lstm_mae_sorted[:5], 1):
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}")

    print(f"\nTop 5 CNN Results:")
    print("-" * 50)
    cnn_mae_sorted = sorted(enumerate(results['cnn_mae']), key=lambda x: x[1])
    for i, (idx, mae) in enumerate(cnn_mae_sorted[:5], 1):
        print(f"  {i}. L={results['lookback_windows'][idx]:2d}, MAE={mae:.6f}")


if __name__ == "__main__":
    # Data path
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'stockData',
        'preprocessedData',
        'SP500_historical_data.csv'
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results', 'lookback_sweep'), exist_ok=True)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'lookback_sweep', 'lookback_evaluation_sweep')
    os.makedirs(output_dir, exist_ok=True)

    # Run evaluation
    results, preparator, train_data, val_data, test_data = evaluate_lookback_windows(data_path, output_dir)

    # Plot MAE vs Lookback Window
    best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae = plot_mae_vs_lookback(results, output_dir)

    # Determine best L values (up to 3)
    lstm_mae_sorted = sorted(enumerate(results['lstm_mae']), key=lambda x: x[1])
    cnn_mae_sorted = sorted(enumerate(results['cnn_mae']), key=lambda x: x[1])

    # Get best 1-3 L values for both models
    best_lstm_L_values = [results['lookback_windows'][idx] for idx, _ in lstm_mae_sorted[:3]]
    best_cnn_L_values = [results['lookback_windows'][idx] for idx, _ in cnn_mae_sorted[:3]]

    print(f"\nBest 3 L values for LSTM: {best_lstm_L_values}")
    print(f"Best 3 L values for CNN:  {best_cnn_L_values}")

    # Generate predictions for best L values
    lstm_predictions = generate_predictions_multiple_L(preparator, train_data, val_data, test_data,
                                                        best_lstm_L_values, output_dir)
    cnn_predictions = generate_predictions_multiple_L(preparator, train_data, val_data, test_data,
                                                       best_cnn_L_values, output_dir)

    # Create prediction plots
    print(f"\nGenerating plots...")
    plot_predictions_vs_actual(lstm_predictions, output_dir, 'LSTM')
    plot_predictions_vs_actual(cnn_predictions, output_dir, 'CNN')

    # Save results
    save_results(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae, output_dir)

    # Print summary
    print_summary(results, best_lstm_l, best_lstm_mae, best_cnn_l, best_cnn_mae)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")
