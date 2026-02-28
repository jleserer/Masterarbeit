"""
Data Preparation Module
========================
Loads normalized stock data and splits it according to Goodfellow, Bengio, Courville (2016):
- Training: 65%
- Validation: 15%
- Test: 20%

Handles scaling and preprocessing for LSTM and CNN models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreparator:
    """
    Prepares and splits stock market data for ML/DL models.
    Uses Goodfellow 65%-15%-20% split for Train-Val-Test.
    
    Predicts only Close price (single output).
    """

    def __init__(self, data_path, target_columns=None, start_date=None):
        """
        Args:
            data_path: Path to CSV file
            target_columns: List of columns to use (default: ['Close'])
            start_date: Optional filter to only use data from this date onwards (e.g. '2015-01-01')
        """
        self.data_path = data_path
        self.target_columns = target_columns or ['Close']
        self.start_date = start_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_scaler = None  # For inverse transform

    def load_and_prepare(self):
        """Load CSV and prepare data with 65%-15%-20% split."""
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data shape: {self.data.shape}")

        # Filter by start date if specified (date is in first column)
        if self.start_date is not None:
            dates = pd.to_datetime(self.data.iloc[:, 0], errors='coerce')
            self.data = self.data[dates >= pd.Timestamp(self.start_date)].reset_index(drop=True)
            print(f"Filtered to data from {self.start_date}: {self.data.shape}")

        # Select target columns
        selected_data = self.data[self.target_columns].apply(pd.to_numeric, errors='coerce')
        selected_data = selected_data.dropna()
        print(f"Selected data shape after cleaning: {selected_data.shape}")

        # Sequential split for time series (no shuffling)
        n = len(selected_data)
        train_end = int(0.65 * n)
        val_end = train_end + int(0.15 * n)

        train_raw = selected_data.values[:train_end]
        val_raw   = selected_data.values[train_end:val_end]
        test_raw  = selected_data.values[val_end:]

        # Fit scaler ONLY on training data to avoid data leakage.
        # Val/Test may be slightly outside [0,1] if prices changed significantly.
        self.scaler.fit(train_raw)
        self.train_data = self.scaler.transform(train_raw)
        self.val_data   = self.scaler.transform(val_raw)
        self.test_data  = self.scaler.transform(test_raw)

        print(f"\nSplit Summary (Goodfellow 65%-15%-20%):")
        print(f"  Train: {len(self.train_data)} samples ({100*len(self.train_data)/n:.1f}%)")
        print(f"  Val:   {len(self.val_data)} samples ({100*len(self.val_data)/n:.1f}%)")
        print(f"  Test:  {len(self.test_data)} samples ({100*len(self.test_data)/n:.1f}%)")

        return self.train_data, self.val_data, self.test_data

    def get_feature_names(self):
        """Return target column names."""
        return self.target_columns

    def inverse_transform(self, scaled_data):
        """Transform scaled predictions back to original scale."""
        return self.scaler.inverse_transform(scaled_data)


def create_sequences(data, lookback_window, target_column=0):
    """
    Create sequences using Loopback-Window approach with floating window.
    Uses backward-looking window (same as sliceWindow.py implementation).

    Args:
        data: 2D array of shape (n_samples, n_features)
        lookback_window: L (window length for sequences)
        target_column: Index of target column (default: 0 for 'Close')

    Returns:
        X: Sequences of shape (n_sequences, L, n_features)
        y: Target values of shape (n_sequences,) - single column
    """

    """
    Forward looking window for sequence creation.

    X, y = [], []

    for i in range(len(data) - lookback_window):
        # Window: [i, i+1, ..., i+L-1] -> Target: [i+L]
        X.append(data[i:i + lookback_window])
        y.append(data[i + lookback_window])

    return np.array(X), np.array(y)
    """

    """
     Backward looking window for sequence creation.
     
     """
    X, y = [], []

    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    for i in range(lookback_window, len(data)):
        # Use previous L days' data: [i-L, i-L+1, ..., i-1]
        X.append(data[i - lookback_window:i, :])
        # Target: Next day's value at target_column
        y.append(data[i, target_column])

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData', 'preprocessedData', 'SP500_historical_data.csv')
    
    preparator = DataPreparator(data_path)
    train, val, test = preparator.load_and_prepare()
    
    # Example: Create sequences with lookback window L=20
    L = 20
    X_train, y_train = create_sequences(train, L)
    X_val, y_val = create_sequences(val, L)
    X_test, y_test = create_sequences(test, L)
    
    print(f"\nSequence Shapes (Loopback Window L={L}):")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
