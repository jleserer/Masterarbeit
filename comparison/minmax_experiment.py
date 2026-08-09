"""
MinMax-Kontrollexperiment: MinMax-Scaling vs. Log-Return vs. Naive-Baseline
============================================================================
Prüft den Übertrag der Log-Return-Ergebnisse auf MinMax-Skalierung unter
möglichst günstigen Bedingungen für MinMax:

  - Nur S&P 500, nur GRU (Best-Konfiguration aus dem Tuning der Arbeit:
    L*=1, dropout=0.2, dense_units=16, lr=0.001, batch=8, epochs=100).
  - 3-Jahres-Fenster vor Covid, so gewählt, dass das letzte Jahr keine
    wesentliche Niveauänderung zeigt (Fenster-Scan siehe results.json):
    2016-03-01 bis 2019-02-28 — letztes Jahr +2,6 %, Test-Segment
    überschreitet das Train-Maximum nur um ~2 % (minimaler Domain-Shift).
  - Chronologischer 65/15/20-Split wie in der Arbeit (Abschnitt 4.3).

Drei Ansätze, ausgewertet auf identischen Test-Tagen auf der Preis-Ebene
(MAPE primär, dazu MAE, RMSE, R², Directional Accuracy, ΔMAPE vs. Naive):

  A) MinMax:    Preise mit auf dem Train-Split gefittetem MinMax-Scaler
                skaliert; GRU prognostiziert den skalierten Folgetagespreis.
  B) LogReturn: wie in der Arbeit; GRU prognostiziert r̂(t),
                P̂(t) = P(t−1)·exp(r̂(t)).
  C) Naive:     Random Walk P̂(t) = P(t−1).

Output: comparison/results/minmax_test/{results.json, predictions.npz,
        01_test_prices.png, 02_mape_comparison.png}

Usage:
    python comparison/minmax_experiment.py
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import INDICES, GRU_UNITS, set_all_seeds, apply_determinism
from data_preparation.data_preparation import create_sequences

# --- Experiment-Spezifikation -----------------------------------------------
WINDOW_START = '2016-02-28'   # exklusiv (dates > START)
WINDOW_END = '2019-02-28'     # inklusiv (dates <= END)
LOOKBACK = 1                  # L* des GRU für SP500 (best_configurations.json)
GRU_CONFIG = {'dropout': 0.2, 'dense_units': 16, 'lr': 0.001,
              'batch': 8, 'epochs': 100}
OUT_DIR = os.path.join(STEP_DIR, 'results', 'minmax_test')

COLORS = {'minmax': '#FF9800', 'logreturn': '#4CAF50', 'naive': 'black'}
LABELS = {'minmax': 'GRU (MinMax)', 'logreturn': 'GRU (Log-Return)',
          'naive': 'Naive P̂(t)=P(t−1)'}


def load_window():
    """Lädt S&P-500-Schlusskurse + Daten und schneidet das Experimentfenster."""
    csv_path = os.path.join(PROJECT_ROOT, 'stockData', 'preprocessedData',
                            INDICES['SP500'])
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df['Price'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    close = pd.to_numeric(df['Close'], errors='coerce')
    ok = dates.notna() & close.notna()
    dates, close = dates[ok].reset_index(drop=True), close[ok].reset_index(drop=True)

    m = (dates > pd.Timestamp(WINDOW_START)) & (dates <= pd.Timestamp(WINDOW_END))
    return dates[m].reset_index(drop=True), close[m].values.astype(float)


def build_gru(lookback, config):
    """GRU exakt wie parameter_tuning.build_gru_model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        GRU(GRU_UNITS[0], input_shape=(lookback, 1),
            return_sequences=True, name='GRU_1'),
        Dropout(config['dropout']),
        GRU(GRU_UNITS[1], return_sequences=False, name='GRU_2'),
        Dropout(config['dropout']),
        Dense(config['dense_units'], activation='relu', name='Dense_1'),
        Dropout(config['dropout']),
        Dense(1, activation='linear', name='Output'),
    ])
    model.compile(optimizer=Adam(learning_rate=config['lr']), loss='mse')
    return model


def train_and_predict(train, val, test, lookback, config):
    """Sequenzbildung pro Split (wie Pipeline), Training mit festen Epochen,
    Rückgabe der Test-Prognosen (im jeweiligen Zielraum)."""
    X_train, y_train = create_sequences(train.reshape(-1, 1), lookback)
    X_val, y_val = create_sequences(val.reshape(-1, 1), lookback)
    X_test, y_test = create_sequences(test.reshape(-1, 1), lookback)

    set_all_seeds()
    model = build_gru(lookback, config)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=config['epochs'], batch_size=config['batch'], verbose=0)
    return model.predict(X_test, verbose=0).flatten(), y_test


def price_metrics(actual_prices, pred_prices, prev_prices):
    """Reporting-Metriken auf Preis-Ebene, konsistent zu post_processing."""
    err = actual_prices - pred_prices
    mape = float(np.mean(np.abs(err / actual_prices)) * 100)
    naive_mape = float(np.mean(np.abs((actual_prices - prev_prices)
                                      / actual_prices)) * 100)
    ss_tot = np.sum((actual_prices - actual_prices.mean()) ** 2)

    actual_dir = np.sign(actual_prices - prev_prices)
    pred_dir = np.sign(pred_prices - prev_prices)
    nz = actual_dir != 0

    # M-Metrik: vorzeichenbehaftetes Verhältnis der Prognoseabweichung zur
    # tatsächlichen Tagesänderung, M = mean[(x(t) − x̂(t)) / (x(t) − x(t−1))].
    # Per Definition 1 für die Naive-Baseline. Term > 0: Abweichung von x(t)
    # fällt auf die Seite von x(t−1); Term > 1: Prognose liegt jenseits von
    # x(t−1); Term < 0: Prognose überschießt x(t). Der Nenner wird an Tagen
    # mit minimaler Kursänderung beliebig klein (hier bis 0,02 Punkte), der
    # Mittelwert ist daher ausreißerdominiert — Median und Anteilswerte
    # werden als robuste Begleitgrößen mitberichtet.
    day_change = actual_prices - prev_prices
    ratio = err / day_change

    return {
        'mape_pct': mape,
        'naive_baseline_mape_pct': naive_mape,
        'mape_improvement_pct': float((naive_mape - mape) / naive_mape * 100),
        'mae_price': float(np.mean(np.abs(err))),
        'rmse_price': float(np.sqrt(np.mean(err ** 2))),
        'r_squared_price': float(1 - np.sum(err ** 2) / ss_tot),
        'directional_accuracy_pct': float(np.mean(actual_dir[nz] == pred_dir[nz]) * 100),
        'm_ratio_mean': float(np.mean(ratio)),
        'm_ratio_median': float(np.median(ratio)),
        'm_ratio_share_positive_pct': float(np.mean(ratio > 0) * 100),
        'm_ratio_share_between_0_1_pct': float(np.mean((ratio > 0) & (ratio <= 1)) * 100),
        'm_ratio_share_gt_1_pct': float(np.mean(ratio > 1) * 100),
        'm_ratio_share_le_0_pct': float(np.mean(ratio <= 0) * 100),
    }


def main():
    print('=' * 70)
    print('MINMAX-KONTROLLEXPERIMENT  (SP500, GRU, 3 Jahre vor Covid)')
    print('=' * 70)
    apply_determinism(set_tf=True, set_torch=False)

    dates, prices = load_window()
    returns = np.log(prices[1:] / prices[:-1])
    n = len(returns)
    train_end = int(0.65 * n)
    val_end = train_end + int(0.15 * n)
    L = LOOKBACK

    # Preis-Splits spiegelbildlich zu den Return-Splits: Return r[k] gehört zu
    # Preis prices[k+1]; der Preis-Split enthält je Split die Zielpreise.
    train_p = prices[:train_end + 1]
    val_p = prices[train_end + 1:val_end + 1]
    test_p = prices[val_end + 1:]

    # Identische Test-Tage aller Ansätze: Ziele = prices[val_end+1+L:]
    actual = prices[val_end + 1 + L:]
    prev = prices[val_end + L:-1]
    test_dates = dates[val_end + 1 + L:].reset_index(drop=True)

    print(f'\nFenster: {dates.iloc[0].date()} bis {dates.iloc[-1].date()} '
          f'({len(prices)} Handelstage)')
    print(f'Split 65/15/20 auf {n} Returns: Train {train_end}, '
          f'Val {val_end - train_end}, Test {n - val_end} (− L={L} '
          f'→ {len(actual)} Test-Tage)')
    print(f'Train-Preisspanne [{train_p.min():.0f}, {train_p.max():.0f}], '
          f'Test [{test_p.min():.0f}, {test_p.max():.0f}] '
          f'(max. Überschreitung {max(test_p.max() / train_p.max() - 1, 0) * 100:.2f} %)')

    # --- A) MinMax auf Preisen (Scaler nur auf Train gefittet) ---------------
    p_min, p_max = train_p.min(), train_p.max()
    scale = lambda p: (p - p_min) / (p_max - p_min)
    print('\n[1/2] Training GRU auf MinMax-skalierten Preisen...')
    pred_scaled, _ = train_and_predict(scale(train_p), scale(val_p), scale(test_p),
                                       L, GRU_CONFIG)
    pred_minmax = pred_scaled * (p_max - p_min) + p_min

    # --- B) Log-Returns (wie in der Arbeit) ----------------------------------
    print('[2/2] Training GRU auf Log-Returns...')
    pred_ret, _ = train_and_predict(returns[:train_end],
                                    returns[train_end:val_end],
                                    returns[val_end:], L, GRU_CONFIG)
    pred_logreturn = prev * np.exp(pred_ret)

    assert len(pred_minmax) == len(pred_logreturn) == len(actual)

    # --- Metriken ------------------------------------------------------------
    results = {
        'minmax': price_metrics(actual, pred_minmax, prev),
        'logreturn': price_metrics(actual, pred_logreturn, prev),
        'naive': price_metrics(actual, prev, prev),
    }
    results['naive']['directional_accuracy_pct'] = None  # Naive: keine Richtung
    results['naive']['mape_improvement_pct'] = 0.0

    scaled_test = scale(test_p)
    meta = {
        'timestamp': datetime.now().isoformat(),
        'index': 'SP500', 'model': 'GRU',
        'window': {'start': str(dates.iloc[0].date()),
                   'end': str(dates.iloc[-1].date()),
                   'n_days': int(len(prices)),
                   'selection_criterion':
                       '3-Jahres-Fenster vor Covid mit minimaler '
                       'Niveauänderung im letzten Jahr (+2,6 %) und minimalem '
                       'Test-Überschuss über die Train-Preisspanne (~2 %)'},
        'split': {'principle': '65/15/20 (chronologisch, wie Abschnitt 4.3)',
                  'n_returns': int(n), 'train': int(train_end),
                  'val': int(val_end - train_end), 'test': int(n - val_end),
                  'n_test_days': int(len(actual)),
                  'test_period': f'{test_dates.iloc[0].date()} bis '
                                 f'{test_dates.iloc[-1].date()}'},
        'gru_config': {**GRU_CONFIG, 'L': L, 'gru_units': list(GRU_UNITS),
                       'source': 'best_configurations.json SP500/best_gru'},
        'minmax_diagnostics': {
            'train_price_min': float(p_min), 'train_price_max': float(p_max),
            'scaled_val_range': [float(scale(val_p).min()), float(scale(val_p).max())],
            'scaled_test_range': [float(scaled_test.min()), float(scaled_test.max())],
            'test_days_above_train_max_pct':
                float(np.mean(scaled_test[L:] > 1.0) * 100),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT_DIR, 'predictions.npz'),
                        test_dates=test_dates.astype(str).values,
                        actual_prices=actual, prev_prices=prev,
                        pred_minmax=pred_minmax, pred_logreturn=pred_logreturn)

    # --- Plots ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates, actual, 'k-', linewidth=1.8, label='Actual', alpha=0.9)
    ax.plot(test_dates, pred_minmax, color=COLORS['minmax'], linewidth=1.2,
            label=LABELS['minmax'], alpha=0.85)
    ax.plot(test_dates, pred_logreturn, color=COLORS['logreturn'], linewidth=1.2,
            label=LABELS['logreturn'], alpha=0.85)
    ax.set_ylabel('S&P 500 (Indexpunkte)', fontsize=12)
    ax.set_title('MinMax-Kontrollexperiment — Test-Split: Actual vs. Prognosen',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '01_test_prices.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    keys = ['minmax', 'logreturn', 'naive']
    mapes = [results[k]['mape_pct'] for k in keys]
    bars = ax.bar([LABELS[k] for k in keys], mapes,
                  color=[COLORS[k] for k in keys], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f} %', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_ylabel('MAPE (%, Preis-Ebene)', fontsize=12)
    ax.set_title('MAPE auf dem Test-Split (identische Test-Tage)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '02_mape_comparison.png'), dpi=150)
    plt.close()

    # --- Summary -------------------------------------------------------------
    print(f'\n{"Ansatz":22s} {"MAPE%":>9s} {"ΔMAPE":>9s} {"MAE":>8s} '
          f'{"RMSE":>8s} {"R²":>8s} {"DA%":>7s}')
    print('-' * 78)
    for k in keys:
        r = results[k]
        da = f"{r['directional_accuracy_pct']:6.2f}%" \
            if r['directional_accuracy_pct'] is not None else '     —'
        print(f'{LABELS[k]:22s} {r["mape_pct"]:8.4f}% '
              f'{r["mape_improvement_pct"]:+8.3f}% {r["mae_price"]:8.2f} '
              f'{r["rmse_price"]:8.2f} {r["r_squared_price"]:8.4f} {da}')

    print(f'\nM-Metrik  M = mean[(x(t)−x̂(t)) / (x(t)−x(t−1))]  (Naive per Definition 1):')
    print(f'{"Ansatz":22s} {"M(mean)":>9s} {"Median":>8s} {">0":>7s} '
          f'{"(0,1]":>7s} {">1":>7s} {"<=0":>7s}')
    print('-' * 78)
    for k in keys:
        r = results[k]
        print(f'{LABELS[k]:22s} {r["m_ratio_mean"]:9.4f} {r["m_ratio_median"]:8.4f} '
              f'{r["m_ratio_share_positive_pct"]:6.1f}% '
              f'{r["m_ratio_share_between_0_1_pct"]:6.1f}% '
              f'{r["m_ratio_share_gt_1_pct"]:6.1f}% '
              f'{r["m_ratio_share_le_0_pct"]:6.1f}%')
    print(f'\nErgebnisse gespeichert in: {OUT_DIR}')


if __name__ == '__main__':
    main()
