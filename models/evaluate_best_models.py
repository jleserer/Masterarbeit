"""
Evaluation der besten LSTM und CNN Modelle
==========================================
Vergleicht die Vorhersagen der beiden besten Modelle mit den realen Werten.
Verwendet die optimalen Konfigurationen aus dem Hyperparameter-Tuning.

Best LSTM: dropout=0.2, dense_units=32, lr=0.005, batch=8, epochs=50
Best CNN: kernel=5, pool=2, dropout=0.2, batch=8, epochs=100
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_preparation import DataPreparator, create_sequences
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Fixed lookback window
L = 60

# Best LSTM configuration (from hyperparameter tuning)
LSTM_CONFIG = {
    'dropout': 0.2,
    'dense_units': 32,
    'learning_rate': 0.005,
    'batch_size': 8,
    'epochs': 50,
    'lstm_units': [64, 32]
}

# Best CNN configuration (from hyperparameter tuning)
CNN_CONFIG = {
    'kernel_size': 5,
    'pool_size': 2,
    'dropout': 0.2,
    'batch_size': 8,
    'epochs': 100,
    'conv_filters': [64, 128, 256],
    'dense_units': 64,
    'learning_rate': 0.001
}


def build_lstm_model(input_shape, config):
    """Build LSTM model with best configuration."""
    model = Sequential([
        LSTM(config['lstm_units'][0], return_sequences=True, input_shape=input_shape),
        Dropout(config['dropout']),
        LSTM(config['lstm_units'][1], return_sequences=False),
        Dropout(config['dropout']),
        Dense(config['dense_units'], activation='relu'),
        Dropout(config['dropout']),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_cnn_model(input_shape, config):
    """Build CNN model with best configuration."""
    model = Sequential([
        Conv1D(config['conv_filters'][0], kernel_size=config['kernel_size'],
               activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=config['pool_size']),
        Dropout(config['dropout']),

        Conv1D(config['conv_filters'][1], kernel_size=config['kernel_size'],
               activation='relu', padding='same'),
        MaxPooling1D(pool_size=config['pool_size']),
        Dropout(config['dropout']),

        Conv1D(config['conv_filters'][2], kernel_size=config['kernel_size'],
               activation='relu', padding='same'),
        Dropout(config['dropout']),

        Flatten(),
        Dense(config['dense_units'], activation='relu'),
        Dropout(config['dropout']),
        Dense(config['dense_units'] // 2, activation='relu'),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def main():
    print("=" * 70)
    print("EVALUATION DER BESTEN MODELLE")
    print("=" * 70)
    print(f"Lookback Window L = {L}")
    print()

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData',
                             'preprocessedData', 'SP500_historical_data.csv')

    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    # Create sequences
    X_train, y_train = create_sequences(train_data, L)
    X_val, y_val = create_sequences(val_data, L)
    X_test, y_test = create_sequences(test_data, L)

    print(f"\nSequence Shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train LSTM
    print("\n" + "=" * 70)
    print("Training Best LSTM Model")
    print("=" * 70)
    print(f"Config: {LSTM_CONFIG}")

    lstm_model = build_lstm_model(input_shape, LSTM_CONFIG)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_CONFIG['epochs'],
        batch_size=LSTM_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_mae = mean_absolute_error(y_test, lstm_pred)
    lstm_mse = mean_squared_error(y_test, lstm_pred)
    print(f"\nLSTM Test MAE: {lstm_mae:.6f}")
    print(f"LSTM Test MSE: {lstm_mse:.6f}")

    # Train CNN
    print("\n" + "=" * 70)
    print("Training Best CNN Model")
    print("=" * 70)
    print(f"Config: {CNN_CONFIG}")

    # Clear session for clean CNN training
    tf.keras.backend.clear_session()

    cnn_model = build_cnn_model(input_shape, CNN_CONFIG)
    cnn_history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_CONFIG['epochs'],
        batch_size=CNN_CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    cnn_pred = cnn_model.predict(X_test, verbose=0).flatten()
    cnn_mae = mean_absolute_error(y_test, cnn_pred)
    cnn_mse = mean_squared_error(y_test, cnn_pred)
    print(f"\nCNN Test MAE: {cnn_mae:.6f}")
    print(f"CNN Test MSE: {cnn_mse:.6f}")

    # Inverse transform to original scale
    y_test_original = preparator.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_pred_original = preparator.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
    cnn_pred_original = preparator.inverse_transform(cnn_pred.reshape(-1, 1)).flatten()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluation', 'SP500')
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Predictions vs Actual Values (Test Set)
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # LSTM predictions vs actual
    axes[0].plot(y_test_original, label='Actual', color='black', linewidth=1)
    axes[0].plot(lstm_pred_original, label=f'LSTM Prediction (MAE={lstm_mae:.4f})',
                 color='blue', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Close Price')
    axes[0].set_title(f'LSTM: Predicted vs Actual (L={L})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CNN predictions vs actual
    axes[1].plot(y_test_original, label='Actual', color='black', linewidth=1)
    axes[1].plot(cnn_pred_original, label=f'CNN Prediction (MAE={cnn_mae:.4f})',
                 color='red', alpha=0.7, linewidth=1)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Close Price')
    axes[1].set_title(f'CNN: Predicted vs Actual (L={L})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 2: Both models comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test_original, label='Actual', color='black', linewidth=1.5)
    ax.plot(lstm_pred_original, label=f'LSTM (MAE={lstm_mae:.4f})',
            color='blue', alpha=0.7, linewidth=1)
    ax.plot(cnn_pred_original, label=f'CNN (MAE={cnn_mae:.4f})',
            color='red', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.set_title(f'Model Comparison: Predicted vs Actual (L={L})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 3: Zoomed view (first 200 samples)
    fig, ax = plt.subplots(figsize=(14, 6))
    zoom_range = min(200, len(y_test_original))
    ax.plot(y_test_original[:zoom_range], label='Actual', color='black', linewidth=1.5)
    ax.plot(lstm_pred_original[:zoom_range], label=f'LSTM', color='blue', alpha=0.7, linewidth=1)
    ax.plot(cnn_pred_original[:zoom_range], label=f'CNN', color='red', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.set_title(f'Model Comparison (Zoomed): First {zoom_range} Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison_zoomed.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 4: Training History
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # LSTM Loss
    axes[0, 0].plot(lstm_history.history['loss'], label='Train')
    axes[0, 0].plot(lstm_history.history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('LSTM Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # LSTM MAE
    axes[0, 1].plot(lstm_history.history['mae'], label='Train')
    axes[0, 1].plot(lstm_history.history['val_mae'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('LSTM Training MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # CNN Loss
    axes[1, 0].plot(cnn_history.history['loss'], label='Train')
    axes[1, 0].plot(cnn_history.history['val_loss'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (MSE)')
    axes[1, 0].set_title('CNN Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # CNN MAE
    axes[1, 1].plot(cnn_history.history['mae'], label='Train')
    axes[1, 1].plot(cnn_history.history['val_mae'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('CNN Training MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Lookback Window: L = {L}")
    print(f"\nLSTM (Best Config):")
    print(f"  Test MAE: {lstm_mae:.6f}")
    print(f"  Test MSE: {lstm_mse:.6f}")
    print(f"\nCNN (Best Config):")
    print(f"  Test MAE: {cnn_mae:.6f}")
    print(f"  Test MSE: {cnn_mse:.6f}")
    print(f"\nBetter Model: {'LSTM' if lstm_mae < cnn_mae else 'CNN'}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
