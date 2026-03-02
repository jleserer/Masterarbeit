"""Quick verification: All 8 indices with 10 epochs, parallelized."""
import os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCHS = 10


def verify_index(index_name, filename):
    """Train LSTM+CNN+GRU+Informer on one index with minimal config, verify inverse_transform."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    np.random.seed(42)
    tf.random.set_seed(42)

    # Limit threads per worker (8 indices / ~16 cores = 2 threads each)
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from data_preparation import DataPreparator, create_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
    from tensorflow.keras.optimizers import Adam

    data_path = os.path.join(os.path.dirname(__file__), '..', 'stockData', 'preprocessedData', filename)
    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    L = 20  # small lookback
    X_train, y_train = create_sequences(train_data, L)
    X_test, y_test = create_sequences(test_data, L)
    n_feat = X_train.shape[2]

    results = {'index': index_name, 'L': L}

    # --- LSTM (5 epochs) ---
    t0 = time.time()
    lstm = Sequential([
        Input(shape=(L, n_feat)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32, return_sequences=False), Dropout(0.2),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(1, activation='linear')
    ])
    lstm.compile(optimizer=Adam(0.005), loss='mse', metrics=['mae'])
    lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
    lstm_loss, lstm_mae = lstm.evaluate(X_test, y_test, verbose=0)
    lstm_pred = lstm.predict(X_test, verbose=0)
    lstm_pred_prices = preparator.inverse_transform(lstm_pred, split='test', lookback=L)
    y_test_prices = preparator.inverse_transform(y_test.reshape(-1, 1), split='test', lookback=L)
    lstm_time = time.time() - t0
    del lstm
    tf.keras.backend.clear_session()

    results['lstm_mae'] = float(lstm_mae)
    results['lstm_rmse'] = float(np.sqrt(lstm_loss))
    results['lstm_time'] = lstm_time
    results['lstm_pred_price_range'] = (float(lstm_pred_prices.min()), float(lstm_pred_prices.max()))

    # --- CNN (5 epochs) ---
    t0 = time.time()
    cnn = Sequential([
        Input(shape=(L, n_feat)),
        Conv1D(64, 3, activation='relu', padding='same'), MaxPooling1D(2), Dropout(0.2),
        Conv1D(128, 3, activation='relu', padding='same'), MaxPooling1D(2), Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(1, activation='linear')
    ])
    cnn.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    cnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
    cnn_loss, cnn_mae = cnn.evaluate(X_test, y_test, verbose=0)
    cnn_pred = cnn.predict(X_test, verbose=0)
    cnn_pred_prices = preparator.inverse_transform(cnn_pred, split='test', lookback=L)
    cnn_time = time.time() - t0
    del cnn
    tf.keras.backend.clear_session()

    results['cnn_mae'] = float(cnn_mae)
    results['cnn_rmse'] = float(np.sqrt(cnn_loss))
    results['cnn_time'] = cnn_time
    results['cnn_pred_price_range'] = (float(cnn_pred_prices.min()), float(cnn_pred_prices.max()))

    # --- GRU ---
    t0 = time.time()
    gru = Sequential([
        Input(shape=(L, n_feat)),
        GRU(64, return_sequences=True), Dropout(0.2),
        GRU(32, return_sequences=False), Dropout(0.2),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(1, activation='linear')
    ])
    gru.compile(optimizer=Adam(0.005), loss='mse', metrics=['mae'])
    gru.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
    gru_loss, gru_mae = gru.evaluate(X_test, y_test, verbose=0)
    gru_pred = gru.predict(X_test, verbose=0)
    gru_pred_prices = preparator.inverse_transform(gru_pred, split='test', lookback=L)
    gru_time = time.time() - t0
    del gru
    tf.keras.backend.clear_session()

    results['gru_mae'] = float(gru_mae)
    results['gru_rmse'] = float(np.sqrt(gru_loss))
    results['gru_time'] = gru_time
    results['gru_pred_price_range'] = (float(gru_pred_prices.min()), float(gru_pred_prices.max()))

    # --- Informer (PyTorch) ---
    t0 = time.time()
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(3)
    from informer_model import build_informer, train_informer, evaluate_informer, predict_informer

    informer_config = {
        'd_model': 32, 'n_heads': 4, 'dropout': 0.05,
        'lr': 0.001, 'batch': 32, 'epochs': EPOCHS,
        'e_layers': 2, 'd_layers': 1, 'factor': 5
    }
    informer = build_informer(L, n_feat, informer_config)
    train_informer(informer, X_train, y_train, informer_config, L)
    informer_loss, informer_mae = evaluate_informer(informer, X_test, y_test, L)
    informer_pred = predict_informer(informer, X_test, L)
    informer_pred_prices = preparator.inverse_transform(informer_pred.reshape(-1, 1), split='test', lookback=L)
    informer_time = time.time() - t0
    del informer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results['informer_mae'] = float(informer_mae)
    results['informer_rmse'] = float(np.sqrt(informer_loss))
    results['informer_time'] = informer_time
    results['informer_pred_price_range'] = (float(informer_pred_prices.min()), float(informer_pred_prices.max()))

    # Sanity checks
    results['actual_price_range'] = (float(y_test_prices.min()), float(y_test_prices.max()))
    results['returns_train_range'] = (float(train_data.min()), float(train_data.max()))
    results['returns_test_range'] = (float(test_data.min()), float(test_data.max()))

    return results


def main():
    indices = {
        'SP500':     'SP500_historical_data.csv',
        'DAX':       'DAX_historical_data.csv',
        'NASDAQ':    'NASDAQ_historical_data.csv',
        'FTSE100':   'FTSE100_historical_data.csv',
        'HANG_SENG': 'HANG_SENG_historical_data.csv',
        'NIKKEI':    'NIKKEI_historical_data.csv',
        '10Y_Bond':  '10-Year Bond_historical_data.csv',
        '30Y_Bond':  '30 Year Bond_historical_data.csv',
    }

    print("=" * 70)
    print("QUICK VERIFICATION: Returns Transformation (All 8 Indices)")
    print(f"Indices: {len(indices)}, Models: LSTM+CNN+GRU+Informer, Epochs: {EPOCHS}, L: 20, Workers: 8")
    print("=" * 70)

    t_start = time.time()

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(verify_index, name, fn): name for name, fn in indices.items()}
        all_results = {}
        for future in futures:
            res = future.result()
            all_results[res['index']] = res

    total_time = time.time() - t_start

    # Print results
    for name, r in all_results.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Returns Train range: [{r['returns_train_range'][0]:.4f}, {r['returns_train_range'][1]:.4f}]")
        print(f"  Returns Test  range: [{r['returns_test_range'][0]:.4f}, {r['returns_test_range'][1]:.4f}]")
        print(f"  -> Domain shift?     {'NO' if abs(r['returns_train_range'][0] - r['returns_test_range'][0]) < 0.5 else 'YES!'}")
        print(f"")
        print(f"  LSTM     MAE: {r['lstm_mae']:.6f}, RMSE: {r['lstm_rmse']:.6f}  ({r['lstm_time']:.1f}s)")
        print(f"  CNN      MAE: {r['cnn_mae']:.6f}, RMSE: {r['cnn_rmse']:.6f}  ({r['cnn_time']:.1f}s)")
        print(f"  GRU      MAE: {r['gru_mae']:.6f}, RMSE: {r['gru_rmse']:.6f}  ({r['gru_time']:.1f}s)")
        print(f"  Informer MAE: {r['informer_mae']:.6f}, RMSE: {r['informer_rmse']:.6f}  ({r['informer_time']:.1f}s)")
        print(f"")
        print(f"  Actual prices:       [{r['actual_price_range'][0]:.2f}, {r['actual_price_range'][1]:.2f}]")
        print(f"  LSTM pred prices:    [{r['lstm_pred_price_range'][0]:.2f}, {r['lstm_pred_price_range'][1]:.2f}]")
        print(f"  CNN  pred prices:    [{r['cnn_pred_price_range'][0]:.2f}, {r['cnn_pred_price_range'][1]:.2f}]")
        print(f"  GRU  pred prices:    [{r['gru_pred_price_range'][0]:.2f}, {r['gru_pred_price_range'][1]:.2f}]")
        print(f"  Inf  pred prices:    [{r['informer_pred_price_range'][0]:.2f}, {r['informer_pred_price_range'][1]:.2f}]")

    print(f"\n{'='*70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
