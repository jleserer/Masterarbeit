"""
Data Preparation Module
========================
Loads stock data, computes percentage returns, and splits according to
Goodfellow, Bengio, Courville (2016):
- Training: 65%
- Validation: 15%
- Test: 20%

Uses percentage returns r(t) = (price(t) - price(t-1)) / price(t-1) instead of
absolute prices. Returns are approximately stationary, eliminating the domain shift
that occurs when MinMaxScaler is fitted on training data only (e.g. SP500 training
prices 242-1565 but test prices 2234-6905).
"""

import os
import numpy as np
import pandas as pd


class DataPreparator:
    """
    Prepares and splits stock market data for ML/DL models.
    Uses Goodfellow 65%-15%-20% split for Train-Val-Test.

    Computes percentage returns from Close prices. Returns are stationary
    and have a similar distribution across all splits, avoiding domain shift.
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
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_base_prices = None
        self.val_base_prices = None
        self.test_base_prices = None

    def load_and_prepare(self):
        """Load CSV, compute returns, and split 65%-15%-20%."""
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

        # Compute percentage returns: r(t) = (price(t) - price(t-1)) / price(t-1)
        prices = selected_data.values  # (N, n_features)
        returns = (prices[1:] - prices[:-1]) / prices[:-1]  # (N-1, n_features)
        base_prices = prices[:-1]  # price at t-1 for each return, aligned by index

        # Sequential split on returns (no shuffling)
        n = len(returns)
        train_end = int(0.65 * n)
        val_end = train_end + int(0.15 * n)

        self.train_data = returns[:train_end]
        self.val_data   = returns[train_end:val_end]
        self.test_data  = returns[val_end:]

        # Store base prices for inverse_transform (price at t-1 for each return)
        self.train_base_prices = base_prices[:train_end]
        self.val_base_prices   = base_prices[train_end:val_end]
        self.test_base_prices  = base_prices[val_end:]

        n_total = n + 1  # original price count
        print(f"\nSplit Summary (Goodfellow 65%-15%-20%):")
        print(f"  Prices: {n_total}, Returns: {n}")
        print(f"  Train: {len(self.train_data)} samples ({100*len(self.train_data)/n:.1f}%)")
        print(f"  Val:   {len(self.val_data)} samples ({100*len(self.val_data)/n:.1f}%)")
        print(f"  Test:  {len(self.test_data)} samples ({100*len(self.test_data)/n:.1f}%)")
        print(f"  Train returns range: [{self.train_data.min():.4f}, {self.train_data.max():.4f}]")
        print(f"  Test  returns range: [{self.test_data.min():.4f}, {self.test_data.max():.4f}]")

        return self.train_data, self.val_data, self.test_data

    def get_feature_names(self):
        """Return target column names."""
        return self.target_columns

    def inverse_transform(self, predicted_returns, split='test', lookback=0):
        """Convert predicted returns back to original price scale.

        After create_sequences(data, L), the k-th target is data[L+k].
        The corresponding base price (price at t-1) is base_prices[L+k].

        Args:
            predicted_returns: Array of shape (n,) or (n, 1)
            split: Which data split ('train', 'val', 'test')
            lookback: Lookback window L used in create_sequences

        Returns:
            Predicted prices as array matching input shape.
            price(t) = base_price(t-1) * (1 + predicted_return(t))
        """
        base_map = {
            'train': self.train_base_prices,
            'val': self.val_base_prices,
            'test': self.test_base_prices,
        }
        base = base_map[split]

        was_2d = (predicted_returns.ndim == 2)
        flat = predicted_returns.flatten()
        n = len(flat)

        relevant_base = base[lookback:lookback + n, 0]  # Close price column
        predicted_prices = relevant_base * (1 + flat)

        if was_2d:
            return predicted_prices.reshape(-1, 1)
        return predicted_prices


def create_sequences(data, lookback_window, target_column=0):
    """
    Create sequences using Loopback-Window approach with floating window.
    Uses backward-looking window approach.

    Args:
        data: 2D array of shape (n_samples, n_features)
        lookback_window: L (window length for sequences)
        target_column: Index of target column (default: 0 for 'Close')

    Returns:
        X: Sequences of shape (n_sequences, L, n_features)
        y: Target values of shape (n_sequences,) - single column
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
