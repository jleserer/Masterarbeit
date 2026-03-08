"""
Model Comparison & Testing Script
=================================
Runs LSTM, CNN, GRU and Informer models and compares their performance.
"""

import os
import sys
import json
import time
import numpy as np

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.gru_model import GRUModel
from models.informer_model import InformerModel


def run_comparison(data_path, stock_name='SP500'):
    """
    Run all four models and compare results.

    Args:
        data_path: Path to stock data CSV
        stock_name: Name of the stock for reporting
    """

    print("\n" + "=" * 80)
    print(f"STOCK PRICE PREDICTION - Model Comparison for {stock_name}")
    print("=" * 80)

    results = {
        'stock': stock_name,
        'models': {}
    }

    # ========== LSTM Model ==========
    print("\n\n" + "#" * 80)
    print("# LSTM MODEL TRAINING")
    print("#" * 80)

    lstm_output_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results', 'LSTM', stock_name)
    lstm_model = LSTMModel(data_path, output_dir=lstm_output_dir)

    lstm_start = time.time()
    lstm_model.run_full_pipeline()
    lstm_time = time.time() - lstm_start

    results['models']['LSTM'] = {
        'output_dir': lstm_output_dir,
        'training_time_seconds': lstm_time,
        'lookback_window': LSTMModel.LOOKBACK_WINDOW,
        'architecture': {
            'layers': 'LSTM(128) + LSTM(64) + Dense(32)',
            'dropout': LSTMModel.DROPOUT_RATE,
            'learning_rate': LSTMModel.LEARNING_RATE
        }
    }

    # ========== CNN Model ==========
    print("\n\n" + "#" * 80)
    print("# CNN MODEL TRAINING")
    print("#" * 80)

    cnn_output_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results', 'CNN', stock_name)
    cnn_model = CNNModel(data_path, output_dir=cnn_output_dir)

    cnn_start = time.time()
    cnn_model.run_full_pipeline()
    cnn_time = time.time() - cnn_start

    results['models']['CNN'] = {
        'output_dir': cnn_output_dir,
        'training_time_seconds': cnn_time,
        'lookback_window': CNNModel.LOOKBACK_WINDOW,
        'architecture': {
            'layers': 'Conv1D(64,128,256) + Dense(64,32)',
            'kernel_size': CNNModel.KERNEL_SIZE,
            'dropout': CNNModel.DROPOUT_RATE,
            'learning_rate': CNNModel.LEARNING_RATE
        }
    }

    # ========== GRU Model ==========
    print("\n\n" + "#" * 80)
    print("# GRU MODEL TRAINING")
    print("#" * 80)

    gru_output_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results', 'GRU', stock_name)
    gru_model = GRUModel(data_path, output_dir=gru_output_dir)

    gru_start = time.time()
    gru_model.run_full_pipeline()
    gru_time = time.time() - gru_start

    results['models']['GRU'] = {
        'output_dir': gru_output_dir,
        'training_time_seconds': gru_time,
        'lookback_window': GRUModel.LOOKBACK_WINDOW,
        'architecture': {
            'layers': 'GRU(128) + GRU(64) + Dense(32)',
            'dropout': GRUModel.DROPOUT_RATE,
            'learning_rate': GRUModel.LEARNING_RATE
        }
    }

    # ========== Informer Model ==========
    print("\n\n" + "#" * 80)
    print("# INFORMER MODEL TRAINING")
    print("#" * 80)

    informer_output_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results', 'INFORMER', stock_name)
    informer_model = InformerModel(data_path, output_dir=informer_output_dir)

    informer_start = time.time()
    informer_model.run_full_pipeline()
    informer_time = time.time() - informer_start

    results['models']['Informer'] = {
        'output_dir': informer_output_dir,
        'training_time_seconds': informer_time,
        'lookback_window': InformerModel.LOOKBACK_WINDOW,
        'architecture': {
            'layers': f'Encoder({InformerModel.E_LAYERS}L) + Decoder({InformerModel.D_LAYERS}L)',
            'd_model': InformerModel.D_MODEL,
            'n_heads': InformerModel.N_HEADS,
            'dropout': InformerModel.DROPOUT,
            'learning_rate': InformerModel.LEARNING_RATE
        }
    }

    # ========== Comparison Summary ==========
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nLSTM     Training Time: {lstm_time:.2f} seconds")
    print(f"CNN      Training Time: {cnn_time:.2f} seconds")
    print(f"GRU      Training Time: {gru_time:.2f} seconds")
    print(f"Informer Training Time: {informer_time:.2f} seconds")

    print("\nResults saved to:")
    print(f"  - LSTM:     {lstm_output_dir}/")
    print(f"  - CNN:      {cnn_output_dir}/")
    print(f"  - GRU:      {gru_output_dir}/")
    print(f"  - Informer: {informer_output_dir}/")

    # Save comparison results
    results_file = os.path.join(STEP_DIR, 'results', stock_name, 'model_comparison_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_ROOT,
        'stockData',
        'preprocessedData',
        'SP500_historical_data.csv'
    )

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    run_comparison(data_path, stock_name='SP500')
