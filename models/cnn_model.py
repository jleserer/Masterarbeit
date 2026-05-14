"""
CNN Model for Stock Price Prediction
====================================
Architecture & Hyperparameters (based on paper comparison):
  - Convolution Layers: 3 mit Filtern aus config.CONV_FILTERS (default [64,128,256])
  - Kernel Size: 5
  - Max Pooling: 2 stride for dimensionality reduction
  - Flattening + Dense layers for regression
  - Optimizer: Adam (learning rate aus config.LEARNING_RATE_CNN)
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from config import (
    CONV_FILTERS as _CONFIG_CONV_FILTERS,
    DENSE_UNITS_CNN as _CONFIG_DENSE_UNITS_CNN,
    LEARNING_RATE_CNN as _CONFIG_LEARNING_RATE_CNN,
    LOOKBACK_WINDOW as _CONFIG_LOOKBACK_WINDOW,
)
from models.base_keras_model import BaseKerasTimeSeriesModel


class CNNModel(BaseKerasTimeSeriesModel):
    """CNN-based stock price predictor."""

    MODEL_LABEL = 'CNN'
    LOOKBACK_WINDOW = _CONFIG_LOOKBACK_WINDOW
    CONV_FILTERS = list(_CONFIG_CONV_FILTERS)
    KERNEL_SIZE = 5
    POOL_SIZE = 2
    DENSE_UNITS = _CONFIG_DENSE_UNITS_CNN
    LEARNING_RATE = _CONFIG_LEARNING_RATE_CNN

    def __init__(self, data_path,
                 output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'parameter_tuning', 'results',
                                         'CNN', 'SP500')):
        super().__init__(data_path, output_dir)

    def build_model(self):
        print("\n" + "=" * 70)
        print(f"STEP 2: Build {self.MODEL_LABEL} Model")
        print("=" * 70)

        n_features = self.X_train.shape[2]

        self.model = Sequential([
            Conv1D(self.CONV_FILTERS[0],
                   kernel_size=self.KERNEL_SIZE, activation='relu',
                   input_shape=(self.LOOKBACK_WINDOW, n_features),
                   padding='same', name='Conv1D_1'),
            MaxPooling1D(pool_size=self.POOL_SIZE, name='MaxPool_1'),
            Dropout(self.DROPOUT_RATE),

            Conv1D(self.CONV_FILTERS[1],
                   kernel_size=self.KERNEL_SIZE, activation='relu',
                   padding='same', name='Conv1D_2'),
            MaxPooling1D(pool_size=self.POOL_SIZE, name='MaxPool_2'),
            Dropout(self.DROPOUT_RATE),

            Conv1D(self.CONV_FILTERS[2],
                   kernel_size=self.KERNEL_SIZE, activation='relu',
                   padding='same', name='Conv1D_3'),
            Dropout(self.DROPOUT_RATE),

            Flatten(name='Flatten'),
            Dense(self.DENSE_UNITS, activation='relu', name='Dense_1'),
            Dropout(self.DROPOUT_RATE),
            Dense(self.DENSE_UNITS // 2, activation='relu', name='Dense_2'),
            Dense(1, activation='linear', name='Output'),
        ])

        self.model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE),
                           loss='mse', metrics=['mae'])

        print("\nModel Architecture:")
        self.model.summary()
        print(f"\nHyperparameters: Conv Filters={self.CONV_FILTERS}, "
              f"Kernel={self.KERNEL_SIZE}, Pool={self.POOL_SIZE}, "
              f"Dropout={self.DROPOUT_RATE}, Dense={self.DENSE_UNITS}, "
              f"LR={self.LEARNING_RATE}, Batch={self.BATCH_SIZE}, Epochs={self.EPOCHS}")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData',
                             'preprocessedData', 'SP500_historical_data.csv')
    CNNModel(data_path).run_full_pipeline()
