"""
Lookback Window (L) Evaluation Script
======================================
Evaluates LSTM and CNN models with different lookback window sizes L.
Creates visualizations:
1. MAE vs L for both models
2. Predicted vs Actual values for fixed L

Uses best configurations from hyperparameter tuning:
- LSTM: dropout=0.2, dense_units=32, lr=0.005, batch=8, epochs=50
- CNN: kernel=5, pool=2, dropout=0.2, batch=8, epochs=100
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import DataPreparator, create_sequences


# Best configurations from hyperparameter tuning
BEST_LSTM_CONFIG = {
    'dropout': 0.2,
    'dense_units': 32,
    'lr': 0.005,
    'batch': 8,
    'epochs': 50
}

BEST_CNN_CONFIG = {
    'kernel': 5,
    'pool': 2,
    'dropout': 0.2,
    'batch': 8,
    'epochs': 100
}

# Lookback windows to test
LOOKBACK_WINDOWS = [10, 20, 30, 40, 50, 60, 80, 100, 120]

# Fixed L for prediction comparison
FIXED_L = 60


def build_lstm_model(lookback_window, n_features, config):
    """Build LSTM model with best configuration."""
    model = Sequential([
        Input(shape=(lookback_window, n_features)),
        LSTM(128, return_sequences=True, name='LSTM_1'),
        Dropout(config['dropout']),
        LSTM(64, return_sequences=False, name='LSTM_2'),
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
    # Check if lookback window is large enough for pooling
    # After 2 pooling layers with pool_size=2, we need at least 4 time steps
    min_lookback = config['pool'] * config['pool'] * 2

    if lookback_window < min_lookback:
        # Use smaller pool size for small lookback windows
        pool_size = 1
    else:
        pool_size = config['pool']

    model = Sequential([
        Input(shape=(lookback_window, n_features)),
        Conv1D(64, kernel_size=config['kernel'], activation='relu', padding='same', name='Conv1D_1'),
        MaxPooling1D(pool_size=pool_size, name='MaxPool_1'),
        Dropout(config['dropout']),
        Conv1D(128, kernel_size=config['kernel'], activation='relu', padding='same', name='Conv1D_2'),
        MaxPooling1D(pool_size=pool_size, name='MaxPool_2'),
        Dropout(config['dropout']),
        Conv1D(256, kernel_size=config['kernel'], activation='relu', padding='same', name='Conv1D_3'),
        Dropout(config['dropout']),
        Flatten(name='Flatten'),
        Dense(64, activation='relu', name='Dense_1'),
        Dropout(config['dropout']),
        Dense(32, activation='relu', name='Dense_2'),
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
    """Evaluate both models for different lookback windows."""
    print("=" * 70)
    print("LOOKBACK WINDOW (L) EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback windows to test: {LOOKBACK_WINDOWS}")
    print()

    # Load and prepare data once
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
    for L in LOOKBACK_WINDOWS:
        print(f"\n{'='*50}")
        print(f"Testing L = {L}")
        print(f"{'='*50}")

        # Create sequences
        X_train, y_train = create_sequences(train_data, L)
        X_val, y_val = create_sequences(val_data, L)
        X_test, y_test = create_sequences(test_data, L)

        n_features = X_train.shape[2]

        print(f"  Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # LSTM
        print(f"  Training LSTM...", end=" ", flush=True)
        lstm_model = build_lstm_model(L, n_features, BEST_LSTM_CONFIG)
        lstm_mae, lstm_loss = train_and_evaluate(
            lstm_model, X_train, y_train, X_val, y_val, X_test, y_test,
            BEST_LSTM_CONFIG, 'LSTM'
        )
        results['lstm_mae'].append(lstm_mae)
        results['lstm_loss'].append(lstm_loss)
        print(f"MAE: {lstm_mae:.6f}")

        # CNN
        print(f"  Training CNN...", end=" ", flush=True)
        cnn_model = build_cnn_model(L, n_features, BEST_CNN_CONFIG)
        cnn_mae, cnn_loss = train_and_evaluate(
            cnn_model, X_train, y_train, X_val, y_val, X_test, y_test,
            BEST_CNN_CONFIG, 'CNN'
        )
        results['cnn_mae'].append(cnn_mae)
        results['cnn_loss'].append(cnn_loss)
        print(f"MAE: {cnn_mae:.6f}")

        # Clear models to free memory
        del lstm_model, cnn_model
        import tensorflow as tf
        tf.keras.backend.clear_session()

    return results, preparator, train_data, val_data, test_data


def generate_predictions_fixed_L(preparator, train_data, val_data, test_data, output_dir):
    """Generate predictions for fixed L and compare with actual values."""
    print(f"\n{'='*70}")
    print(f"GENERATING PREDICTIONS FOR FIXED L = {FIXED_L}")
    print(f"{'='*70}")

    # Create sequences
    X_train, y_train = create_sequences(train_data, FIXED_L)
    X_val, y_val = create_sequences(val_data, FIXED_L)
    X_test, y_test = create_sequences(test_data, FIXED_L)

    n_features = X_train.shape[2]

    # Train LSTM
    print("Training LSTM for predictions...")
    lstm_model = build_lstm_model(FIXED_L, n_features, BEST_LSTM_CONFIG)
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
    print("Training CNN for predictions...")
    cnn_model = build_cnn_model(FIXED_L, n_features, BEST_CNN_CONFIG)
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

    return y_test, lstm_predictions, cnn_predictions, y_test_original, lstm_pred_original, cnn_pred_original


def plot_results(results, y_actual, lstm_pred, cnn_pred,
                 y_actual_orig, lstm_pred_orig, cnn_pred_orig, output_dir):
    """Create visualization plots."""
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: MAE vs Lookback Window L
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results['lookback_windows'], results['lstm_mae'],
            'b-o', linewidth=2, markersize=8, label='LSTM')
    ax.plot(results['lookback_windows'], results['cnn_mae'],
            'r-s', linewidth=2, markersize=8, label='CNN')

    ax.set_xlabel('Lookback Window (L)', fontsize=12)
    ax.set_ylabel('Test MAE', fontsize=12)
    ax.set_title('Model Performance vs Lookback Window Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(results['lookback_windows'])

    # Add value labels
    for i, (lstm_mae, cnn_mae) in enumerate(zip(results['lstm_mae'], results['cnn_mae'])):
        L = results['lookback_windows'][i]
        ax.annotate(f'{lstm_mae:.4f}', (L, lstm_mae), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, color='blue')
        ax.annotate(f'{cnn_mae:.4f}', (L, cnn_mae), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=8, color='red')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'mae_vs_lookback_window.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 2: Predicted vs Actual (Normalized Scale) - Time Series
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # LSTM predictions
    axes[0].plot(y_actual, 'k-', alpha=0.7, linewidth=1, label='Actual')
    axes[0].plot(lstm_pred, 'b-', alpha=0.7, linewidth=1, label='LSTM Predicted')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Normalized Close Price')
    axes[0].set_title(f'LSTM: Predicted vs Actual (L={FIXED_L})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CNN predictions
    axes[1].plot(y_actual, 'k-', alpha=0.7, linewidth=1, label='Actual')
    axes[1].plot(cnn_pred, 'r-', alpha=0.7, linewidth=1, label='CNN Predicted')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Normalized Close Price')
    axes[1].set_title(f'CNN: Predicted vs Actual (L={FIXED_L})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'predictions_vs_actual_L{FIXED_L}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 3: Predicted vs Actual (Original Scale) - Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LSTM scatter
    axes[0].scatter(y_actual_orig, lstm_pred_orig, alpha=0.5, s=10, c='blue')
    min_val = min(y_actual_orig.min(), lstm_pred_orig.min())
    max_val = max(y_actual_orig.max(), lstm_pred_orig.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Close Price')
    axes[0].set_ylabel('Predicted Close Price')
    axes[0].set_title(f'LSTM: Scatter Plot (L={FIXED_L})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CNN scatter
    axes[1].scatter(y_actual_orig, cnn_pred_orig, alpha=0.5, s=10, c='red')
    min_val = min(y_actual_orig.min(), cnn_pred_orig.min())
    max_val = max(y_actual_orig.max(), cnn_pred_orig.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Close Price')
    axes[1].set_ylabel('Predicted Close Price')
    axes[1].set_title(f'CNN: Scatter Plot (L={FIXED_L})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'scatter_predictions_L{FIXED_L}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 4: Zoomed view of last 100 predictions
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    last_n = 100

    axes[0].plot(range(last_n), y_actual[-last_n:], 'k-', linewidth=2, label='Actual')
    axes[0].plot(range(last_n), lstm_pred[-last_n:], 'b--', linewidth=2, label='LSTM Predicted')
    axes[0].set_xlabel('Time Step (Last 100)')
    axes[0].set_ylabel('Normalized Close Price')
    axes[0].set_title(f'LSTM: Last {last_n} Predictions (L={FIXED_L})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(last_n), y_actual[-last_n:], 'k-', linewidth=2, label='Actual')
    axes[1].plot(range(last_n), cnn_pred[-last_n:], 'r--', linewidth=2, label='CNN Predicted')
    axes[1].set_xlabel('Time Step (Last 100)')
    axes[1].set_ylabel('Normalized Close Price')
    axes[1].set_title(f'CNN: Last {last_n} Predictions (L={FIXED_L})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'predictions_zoomed_L{FIXED_L}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Combined comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(y_actual, 'k-', alpha=0.8, linewidth=1.5, label='Actual')
    ax.plot(lstm_pred, 'b-', alpha=0.6, linewidth=1, label='LSTM')
    ax.plot(cnn_pred, 'r-', alpha=0.6, linewidth=1, label='CNN')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Close Price')
    ax.set_title(f'LSTM vs CNN vs Actual (L={FIXED_L})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'combined_comparison_L{FIXED_L}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def save_results(results, output_dir):
    """Save results to JSON file."""
    results_path = os.path.join(output_dir, 'lookback_evaluation_results.json')

    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'lookback_windows': results['lookback_windows'],
        'lstm_mae': [float(x) for x in results['lstm_mae']],
        'cnn_mae': [float(x) for x in results['cnn_mae']],
        'lstm_loss': [float(x) for x in results['lstm_loss']],
        'cnn_loss': [float(x) for x in results['cnn_loss']],
        'best_lstm_config': BEST_LSTM_CONFIG,
        'best_cnn_config': BEST_CNN_CONFIG,
        'timestamp': datetime.now().isoformat()
    }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {results_path}")


def print_summary(results):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print("\nMAE by Lookback Window:")
    print("-" * 50)
    print(f"{'L':<10} {'LSTM MAE':<15} {'CNN MAE':<15}")
    print("-" * 50)

    for i, L in enumerate(results['lookback_windows']):
        print(f"{L:<10} {results['lstm_mae'][i]:<15.6f} {results['cnn_mae'][i]:<15.6f}")

    print("-" * 50)

    # Find best L for each model
    best_lstm_idx = np.argmin(results['lstm_mae'])
    best_cnn_idx = np.argmin(results['cnn_mae'])

    print(f"\nBest LSTM: L={results['lookback_windows'][best_lstm_idx]}, MAE={results['lstm_mae'][best_lstm_idx]:.6f}")
    print(f"Best CNN:  L={results['lookback_windows'][best_cnn_idx]}, MAE={results['cnn_mae'][best_cnn_idx]:.6f}")


if __name__ == "__main__":
    # Data path
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'stockData',
        'preprocessedData',
        'SP500_historical_data.csv'
    )

    output_dir = os.path.join(os.path.dirname(__file__), 'lookback_evaluation')

    # Run evaluation
    results, preparator, train_data, val_data, test_data = evaluate_lookback_windows(data_path, output_dir)

    # Generate predictions for fixed L
    (y_actual, lstm_pred, cnn_pred,
     y_actual_orig, lstm_pred_orig, cnn_pred_orig) = generate_predictions_fixed_L(
        preparator, train_data, val_data, test_data, output_dir
    )

    # Create plots
    plot_results(results, y_actual, lstm_pred, cnn_pred,
                y_actual_orig, lstm_pred_orig, cnn_pred_orig, output_dir)

    # Save results
    save_results(results, output_dir)

    # Print summary
    print_summary(results)

    print(f"\n{'='*70}")
    print("LOOKBACK WINDOW EVALUATION COMPLETE")
    print(f"{'='*70}")
