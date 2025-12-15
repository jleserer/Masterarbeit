"""
LSTM Model for Stock Price Prediction
======================================
Architecture & Hyperparameters (based on paper comparison):
  - Layers: 2-3 stacked LSTM layers (64-128 units each)
  - Dropout: 0.2-0.3 for regularization
  - Dense layers: 32 units for output projection
  - Optimizer: Adam (learning rate 0.001)
  - Loss: MSE (for single-output regression)
  
Loopback Window: L=30 (30 trading days history)
Target: Single-output (Close price only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import DataPreparator, create_sequences


class LSTMModel:
    """LSTM-based stock price predictor."""
    
    # Architecture Configuration
    LOOKBACK_WINDOW = 30  # L: number of previous time steps
    LSTM_UNITS = [128, 64]  # Two LSTM layers
    DROPOUT_RATE = 0.2
    DENSE_UNITS = 32
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    
    def __init__(self, data_path, output_dir='lstm_results'):
        """
        Args:
            data_path: Path to CSV file
            output_dir: Directory to save results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.preparator = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.history = None
        self.predictions = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_data(self):
        """Load and prepare data with sequences."""
        print("=" * 70)
        print("STEP 1: Data Preparation (65%-15%-20% split)")
        print("=" * 70)
        
        self.preparator = DataPreparator(self.data_path)
        train_data, val_data, test_data = self.preparator.load_and_prepare()
        
        # Create sequences with loopback window
        print(f"\nCreating sequences with Loopback Window L={self.LOOKBACK_WINDOW}...")
        self.X_train, self.y_train = create_sequences(train_data, self.LOOKBACK_WINDOW)
        self.X_val, self.y_val = create_sequences(val_data, self.LOOKBACK_WINDOW)
        self.X_test, self.y_test = create_sequences(test_data, self.LOOKBACK_WINDOW)
        
        print(f"Training sequences: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Validation sequences: X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"Test sequences: X={self.X_test.shape}, y={self.y_test.shape}")
    
    def build_model(self):
        """Build LSTM model."""
        print("\n" + "=" * 70)
        print("STEP 2: Build LSTM Model")
        print("=" * 70)
        
        n_features = self.X_train.shape[2]
        
        self.model = Sequential([
            # First LSTM layer with return sequences
            LSTM(self.LSTM_UNITS[0], 
                 input_shape=(self.LOOKBACK_WINDOW, n_features),
                 return_sequences=True, 
                 name='LSTM_1'),
            Dropout(self.DROPOUT_RATE),
            
            # Second LSTM layer
            LSTM(self.LSTM_UNITS[1], 
                 return_sequences=False, 
                 name='LSTM_2'),
            Dropout(self.DROPOUT_RATE),
            
            # Dense layers
            Dense(self.DENSE_UNITS, activation='relu', name='Dense_1'),
            Dropout(self.DROPOUT_RATE),
            
            # Output layer (single output: Close)
            Dense(1, activation='linear', name='Output')
        ])
        
        optimizer = Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nHyperparameters:")
        print(f"  - LSTM Units: {self.LSTM_UNITS}")
        print(f"  - Dropout Rate: {self.DROPOUT_RATE}")
        print(f"  - Dense Units: {self.DENSE_UNITS}")
        print(f"  - Learning Rate: {self.LEARNING_RATE}")
        print(f"  - Batch Size: {self.BATCH_SIZE}")
        print(f"  - Epochs: {self.EPOCHS}")
    
    def train(self):
        """Train the model."""
        print("\n" + "=" * 70)
        print("STEP 3: Train LSTM Model")
        print("=" * 70)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n" + "=" * 70)
        print("STEP 4: Evaluate Model")
        print("=" * 70)
        
        train_loss, train_mae = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss, val_mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"\nTraining   Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")
        print(f"Test       Loss: {test_loss:.6f}, MAE: {test_mae:.6f}")
        
        return test_loss, test_mae
    
    def predict(self):
        """Generate predictions."""
        print("\n" + "=" * 70)
        print("STEP 5: Generate Predictions")
        print("=" * 70)
        
        self.predictions['train'] = self.model.predict(self.X_train, verbose=0)
        self.predictions['val'] = self.model.predict(self.X_val, verbose=0)
        self.predictions['test'] = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform to original scale
        self.predictions['train_original'] = self.preparator.inverse_transform(self.predictions['train'])
        self.predictions['val_original'] = self.preparator.inverse_transform(self.predictions['val'])
        self.predictions['test_original'] = self.preparator.inverse_transform(self.predictions['test'])
        
        print("Predictions generated and inverse-transformed!")
    
    def plot_results(self):
        """Plot training history and predictions."""
        print("\n" + "=" * 70)
        print("STEP 6: Plot Results")
        print("=" * 70)
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('LSTM Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('LSTM Training History - MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'lstm_training_history.png')
        plt.savefig(plot_path)
        print(f"Saved training history plot: {plot_path}")
        plt.close()
    
    def save_model(self):
        """Save model to disk."""
        model_path = os.path.join(self.output_dir, 'lstm_model.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def run_full_pipeline(self):
        """Execute complete training pipeline."""
        self.prepare_data()
        self.build_model()
        self.train()
        self.evaluate()
        self.predict()
        self.plot_results()
        self.save_model()
        
        print("\n" + "=" * 70)
        print("LSTM Pipeline completed!")
        print("=" * 70)


if __name__ == "__main__":
    # Use one stock dataset
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'stockData', 
        'normalizedData', 
        'SP500_historical_data.csv'
    )
    
    lstm = LSTMModel(data_path, output_dir='lstm_results')
    lstm.run_full_pipeline()
