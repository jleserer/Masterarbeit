"""
SMA-20-Baseline-Auswertung auf dem Test-Split.
================================================
Berechnet die Prognosegüte eines 20-Tage gleitenden Durchschnitts (SMA-20)
als Vergleichs-Baseline zu den Deep-Learning-Modellen — auf der **Preis-Ebene**
(zurücktransformierte echte Kurswerte), konsistent mit den übrigen
Reporting-Metriken der Arbeit.

SMA-20-Prognose:  P̂(t) = (1/20) · Σ P(t-i),  i = 1..20

Die Split-Logik (65/15/20 auf Log-Returns) wird exakt aus DataPreparator
übernommen, damit der SMA-20 auf demselben Test-Segment ausgewertet wird wie
die Modelle. Für die ersten Test-Tage greift das SMA-Fenster auf Preise des
unmittelbar vorhergehenden Validierungssegments zurück — das ist kein
Look-Ahead-Bias, da nur vergangene Preise verwendet werden.

Output: comparison/results/sma_baseline.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import INDICES  # noqa: E402

CSV_DIR = PROJECT_ROOT / 'stockData' / 'preprocessedData'
OUT_PATH = PROJECT_ROOT / 'comparison' / 'results' / 'sma_baseline.json'

SMA_WINDOW = 20


def load_prices(index_name):
    """Lädt die bereinigte Close-Preisreihe — exakt wie DataPreparator."""
    df = pd.read_csv(CSV_DIR / INDICES[index_name])
    close = pd.to_numeric(df['Close'], errors='coerce')
    prices = close.dropna().values.astype(float)
    return prices


def compute_sma_metrics(index_name):
    """Berechnet SMA-20-Metriken auf dem Test-Split (Preis-Ebene).

    Returns dict mit MAE, RMSE, MAPE, R² (Preise) sowie MAE/RMSE und
    MAE-Verbesserung auf Log-Return-Ebene und Directional Accuracy.
    """
    prices = load_prices(index_name)

    # Log-Returns + Split exakt wie DataPreparator.load_and_prepare
    returns = np.log(prices[1:] / prices[:-1])
    n = len(returns)
    train_end = int(0.65 * n)
    val_end = train_end + int(0.15 * n)

    # Test-Returns: returns[val_end:].  Return r[k] gehört zu Preis prices[k+1].
    # Das y des Test-Splits sind die Preise prices[val_end+1 : n+1].
    test_price_start = val_end + 1            # Index des ersten Test-Preises
    test_actual_prices = prices[test_price_start:]   # tatsächliche Test-Preise

    # SMA-20-Prognose für jeden Test-Preis P(t):  Mittel der 20 Vortagespreise.
    # P(t) hat Preis-Index pi; SMA nutzt prices[pi-20 : pi].
    sma_pred_prices = np.empty_like(test_actual_prices)
    for j, pi in enumerate(range(test_price_start, len(prices))):
        window = prices[pi - SMA_WINDOW:pi]
        sma_pred_prices[j] = window.mean()

    # vorheriger tatsächlicher Preis P(t-1) — für Return-Ebene + Naive-Vergleich
    prev_prices = prices[test_price_start - 1:len(prices) - 1]

    # --- Preis-Ebene ---
    err = test_actual_prices - sma_pred_prices
    mae_price = float(np.mean(np.abs(err)))
    rmse_price = float(np.sqrt(np.mean(err ** 2)))
    mape_price = float(np.mean(np.abs(err / test_actual_prices)) * 100)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((test_actual_prices - test_actual_prices.mean()) ** 2)
    r2_price = float(1 - ss_res / ss_tot)

    # --- Return-Ebene ---
    # SMA-20 als Return-Prognose: r̂(t) = ln(SMA(t) / P(t-1))
    sma_pred_returns = np.log(sma_pred_prices / prev_prices)
    actual_returns = np.log(test_actual_prices / prev_prices)
    err_r = actual_returns - sma_pred_returns
    mae_ret = float(np.mean(np.abs(err_r)))
    rmse_ret = float(np.sqrt(np.mean(err_r ** 2)))
    naive_mae_ret = float(np.mean(np.abs(actual_returns)))   # Naive-Zero-Baseline
    mae_improvement = float((naive_mae_ret - mae_ret) / naive_mae_ret * 100)

    # Directional Accuracy (Return-Ebene)
    nz = actual_returns != 0
    dir_acc = float(np.mean(np.sign(actual_returns[nz]) ==
                            np.sign(sma_pred_returns[nz])) * 100)

    return {
        'n_test': int(len(test_actual_prices)),
        'mae_price': mae_price,
        'rmse_price': rmse_price,
        'mape_price_pct': mape_price,
        'r_squared_price': r2_price,
        'mae_return': mae_ret,
        'rmse_return': rmse_ret,
        'naive_baseline_mae_return': naive_mae_ret,
        'mae_improvement_pct': mae_improvement,
        'directional_accuracy_pct': dir_acc,
    }


def main():
    results = {}
    print("SMA-20-Baseline — Auswertung auf dem Test-Split (Preis-Ebene)")
    print("=" * 78)
    print(f"{'Index':11s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>9s} "
          f"{'R2':>9s} {'DirAcc%':>9s} {'MAE-Impr%':>10s}")
    print("-" * 78)
    for index_name in INDICES:
        m = compute_sma_metrics(index_name)
        results[index_name] = m
        print(f"{index_name:11s} {m['mae_price']:10.4f} {m['rmse_price']:10.4f} "
              f"{m['mape_price_pct']:8.4f}% {m['r_squared_price']:9.5f} "
              f"{m['directional_accuracy_pct']:8.2f}% {m['mae_improvement_pct']:+10.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == '__main__':
    main()
