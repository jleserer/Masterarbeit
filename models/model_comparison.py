"""
Model Comparison & Testing Script
=================================
Runs LSTM, CNN and GRU models and compares their performance.
"""

import os
import sys
import json
import time
import numpy as np
from lstm_model import LSTMModel
from cnn_model import CNNModel
from gru_model import GRUModel


def run_comparison(data_path, stock_name='SP500'):
    """
    Run both LSTM and CNN models and compare results.
    
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
    
    lstm_output_dir = os.path.join('..', 'results', 'tuning', 'LSTM', stock_name)
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
    
    cnn_output_dir = os.path.join('..', 'results', 'tuning', 'CNN', stock_name)
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

    gru_output_dir = os.path.join('..', 'results', 'tuning', 'GRU', stock_name)
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

    # ========== Comparison Summary ==========
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nLSTM Training Time: {lstm_time:.2f} seconds")
    print(f"CNN Training Time:  {cnn_time:.2f} seconds")
    print(f"GRU Training Time:  {gru_time:.2f} seconds")
    
    print("\nResults saved to:")
    print(f"  - LSTM: {lstm_output_dir}/")
    print(f"  - CNN:  {cnn_output_dir}/")
    print(f"  - GRU:  {gru_output_dir}/")
    
    # Save comparison results
    results_file = os.path.join('results', 'evaluation', stock_name, 'model_comparison_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    # Select data path
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'stockData',
        'preprocessedData',
        'SP500_historical_data.csv'
    )
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Run comparison
    run_comparison(data_path, stock_name='SP500')
