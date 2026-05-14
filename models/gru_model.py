"""
GRU Model for Stock Price Prediction
======================================
Architecture & Hyperparameters (parallel to LSTM model):
  - Layers: 2 stacked GRU layers (config.GRU_UNITS, default [128, 64])
  - Dropout: 0.2 for regularization
  - Dense layers: 32 units for output projection
  - Optimizer: Adam (learning rate 0.001)
  - Loss: MSE (for single-output regression)

Lookback Window: aus config.LOOKBACK_WINDOW (default 60)
Target: Single-output (Close price only)

Hinweis: Diese Klasse ist der Standalone-Pfad. Die Pipeline (parameter_tuning.py)
baut Modelle direkt aus den config.py-Konstanten und nutzt diese Klasse nicht.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config import GRU_UNITS as _CONFIG_GRU_UNITS, LOOKBACK_WINDOW as _CONFIG_LOOKBACK_WINDOW
from models.base_keras_model import BaseKerasTimeSeriesModel


class GRUModel(BaseKerasTimeSeriesModel):
    """GRU-based stock price predictor."""

    MODEL_LABEL = 'GRU'
    LOOKBACK_WINDOW = _CONFIG_LOOKBACK_WINDOW
    GRU_UNITS = list(_CONFIG_GRU_UNITS)

    def __init__(self, data_path,
                 output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'parameter_tuning', 'results',
                                         'GRU', 'SP500')):
        super().__init__(data_path, output_dir)

    def build_model(self):
        print("\n" + "=" * 70)
        print(f"STEP 2: Build {self.MODEL_LABEL} Model")
        print("=" * 70)

        n_features = self.X_train.shape[2]

        self.model = Sequential([
            GRU(self.GRU_UNITS[0],
                input_shape=(self.LOOKBACK_WINDOW, n_features),
                return_sequences=True, name='GRU_1'),
            Dropout(self.DROPOUT_RATE),

            GRU(self.GRU_UNITS[1],
                return_sequences=False, name='GRU_2'),
            Dropout(self.DROPOUT_RATE),

            Dense(self.DENSE_UNITS, activation='relu', name='Dense_1'),
            Dropout(self.DROPOUT_RATE),

            Dense(1, activation='linear', name='Output'),
        ])

        self.model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE),
                           loss='mse', metrics=['mae'])

        print("\nModel Architecture:")
        self.model.summary()
        print(f"\nHyperparameters: GRU Units={self.GRU_UNITS}, "
              f"Dropout={self.DROPOUT_RATE}, Dense={self.DENSE_UNITS}, "
              f"LR={self.LEARNING_RATE}, Batch={self.BATCH_SIZE}, Epochs={self.EPOCHS}")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData',
                             'preprocessedData', 'SP500_historical_data.csv')
    GRUModel(data_path).run_full_pipeline()
