"""
CNN Model for Stock Price Prediction
====================================
Architecture & Hyperparameters (based on paper comparison):
  - Convolution Layers: 2-3 with filters [64, 128, 256]
  - Kernel Size: 3 or 5
  - Max Pooling: 2 stride for dimensionality reduction
  - Flattening + Dense layers for regression
  - Optimizer: Adam (learning rate 0.001)
  - Loss: MSE (for single-output regression)

Loopback Window: L=60 (60 trading days floating window, same as LSTM for fair comparison)
Target: Single-output (Close price only)

CNN treats the time-series data as a 2D feature map where:
- First dimension = time steps (1 to L)
- Second dimension = single feature (Close)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import DataPreparator, create_sequences


class CNNModel:
    """CNN-based stock price predictor."""
    
    # Architecture Configuration
    LOOKBACK_WINDOW = 60  # L: 60 days floating window (same as LSTM for comparison)
    CONV_FILTERS = [64, 128, 256]  # Progressive increase
    KERNEL_SIZE = 5  # 5-point convolution filter - #3 ausprobieren
    POOL_SIZE = 2 # 4
    DROPOUT_RATE = 0.2 #0.4
    DENSE_UNITS = 64
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32 #8 16 32 <- mit einer Konfig von den anderen Configs prüfen
    EPOCHS = 100 #40-50 <- mit einer Konfig von den anderen Configs prüfen
    
    def __init__(self, data_path, output_dir=os.path.join('..', 'results', 'tuning', 'CNN', 'SP500')):
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
        """Build CNN model."""
        print("\n" + "=" * 70)
        print("STEP 2: Build CNN Model")
        print("=" * 70)
        
        n_features = self.X_train.shape[2]
        
        self.model = Sequential([
            # First Conv1D layer
            Conv1D(self.CONV_FILTERS[0], 
                   kernel_size=self.KERNEL_SIZE, 
                   activation='relu',
                   input_shape=(self.LOOKBACK_WINDOW, n_features),
                   padding='same',
                   name='Conv1D_1'),
            MaxPooling1D(pool_size=self.POOL_SIZE, name='MaxPool_1'),
            Dropout(self.DROPOUT_RATE),
            
            # Second Conv1D layer
            Conv1D(self.CONV_FILTERS[1], 
                   kernel_size=self.KERNEL_SIZE, 
                   activation='relu',
                   padding='same',
                   name='Conv1D_2'),
            MaxPooling1D(pool_size=self.POOL_SIZE, name='MaxPool_2'),
            Dropout(self.DROPOUT_RATE),
            
            # Third Conv1D layer
            Conv1D(self.CONV_FILTERS[2], 
                   kernel_size=self.KERNEL_SIZE, 
                   activation='relu',
                   padding='same',
                   name='Conv1D_3'),
            Dropout(self.DROPOUT_RATE),
            
            # Flatten and Dense layers
            Flatten(name='Flatten'),
            Dense(self.DENSE_UNITS, activation='relu', name='Dense_1'),
            Dropout(self.DROPOUT_RATE),
            Dense(self.DENSE_UNITS // 2, activation='relu', name='Dense_2'),
            
            # Output layer (single output: Close)
            Dense(1, activation='linear', name='Output')
        ])
        
        optimizer = Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nHyperparameters:")
        print(f"  - Conv Filters: {self.CONV_FILTERS}")
        print(f"  - Kernel Size: {self.KERNEL_SIZE}")
        print(f"  - Pool Size: {self.POOL_SIZE}")
        print(f"  - Dropout Rate: {self.DROPOUT_RATE}")
        print(f"  - Dense Units: {self.DENSE_UNITS}")
        print(f"  - Learning Rate: {self.LEARNING_RATE}")
        print(f"  - Batch Size: {self.BATCH_SIZE}")
        print(f"  - Epochs: {self.EPOCHS}")
    
    def train(self):
        """Train the model."""
        print("\n" + "=" * 70)
        print("STEP 3: Train CNN Model")
        print("=" * 70)
        
        # With percentage returns, val data has a similar distribution to training data
        # (no domain shift). EarlyStopping could be used but is kept disabled for
        # consistency with the hyperparameter tuning results.
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
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

        train_rmse, val_rmse, test_rmse = np.sqrt(train_loss), np.sqrt(val_loss), np.sqrt(test_loss)

        print(f"\nTraining   Loss: {train_loss:.6f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        print(f"Test       Loss: {test_loss:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        return test_loss, test_mae, test_rmse
    
    def predict(self):
        """Generate predictions."""
        print("\n" + "=" * 70)
        print("STEP 5: Generate Predictions")
        print("=" * 70)
        
        self.predictions['train'] = self.model.predict(self.X_train, verbose=0)
        self.predictions['val'] = self.model.predict(self.X_val, verbose=0)
        self.predictions['test'] = self.model.predict(self.X_test, verbose=0)

        # Inverse transform to original price scale
        self.predictions['train_original'] = self.preparator.inverse_transform(
            self.predictions['train'], split='train', lookback=self.LOOKBACK_WINDOW)
        self.predictions['val_original'] = self.preparator.inverse_transform(
            self.predictions['val'], split='val', lookback=self.LOOKBACK_WINDOW)
        self.predictions['test_original'] = self.preparator.inverse_transform(
            self.predictions['test'], split='test', lookback=self.LOOKBACK_WINDOW)
        
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
        axes[0].set_title('CNN Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('CNN Training History - MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'cnn_training_history.png')
        plt.savefig(plot_path)
        print(f"Saved training history plot: {plot_path}")
        plt.close()
    
    def save_model(self):
        """Save model to disk."""
        model_path = os.path.join(self.output_dir, 'cnn_model.h5')
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
        print("CNN Pipeline completed!")
        print("=" * 70)


if __name__ == "__main__":
    # Use one stock dataset
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'stockData', 
        'preprocessedData',
        'SP500_historical_data.csv'
    )
    
    cnn = CNNModel(data_path, output_dir=os.path.join('..', 'results', 'tuning', 'CNN', 'SP500'))
    cnn.run_full_pipeline()
