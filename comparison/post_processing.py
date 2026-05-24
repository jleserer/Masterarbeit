"""
Post-Processing: Extended Metrics & Visualizations
====================================================
Phase 6 der Pipeline. Läuft NACH Phase 4 (best_configurations.json geschrieben).

WICHTIG: Diese Phase trainiert nichts mehr neu. Sie liest die in Phase 3 erzeugten
pred_test.npz aus parameter_tuning/results/<MODEL>/<INDEX>/L<best_L>/pred_test.npz
und berechnet darauf:

  - Directional Accuracy (% korrekte Up/Down-Predictions)
  - Naive Baseline-Vergleich (always-zero predictor)
  - MAE Improvement vs Baseline
  - R-squared
  - Cumulative Returns plots
  - Cross-Index Heatmaps

Usage:
    python post_processing.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import INDICES

MODELS = ['lstm', 'cnn', 'gru', 'informer']
MODEL_COLORS = {'lstm': '#2196F3', 'cnn': '#F44336', 'gru': '#4CAF50', 'informer': '#9C27B0'}
MODEL_LABELS = {'lstm': 'LSTM', 'cnn': 'CNN', 'gru': 'GRU', 'informer': 'Informer'}


# =============================================================================
# 1. LOAD BEST CONFIGURATIONS
# =============================================================================

def load_best_configs(results_dir):
    """Lade best_<model> Einträge aus best_configurations.json (Phase 4).

    Returns:
        dict: {index: {model: {'best_L': int, 'config_name': str}}}
    """
    best_configs_path = os.path.join(results_dir, 'best_configurations.json')
    if not os.path.exists(best_configs_path):
        print(f"ERROR: {best_configs_path} not found. Run the pipeline first.")
        sys.exit(1)

    with open(best_configs_path, 'r') as f:
        all_best = json.load(f)

    configs = {}
    for index in INDICES:
        if index not in all_best:
            print(f"WARNING: {index} nicht in best_configurations.json — skip")
            continue
        idx_data = all_best[index]
        configs[index] = {}
        for model, key in [('lstm', 'best_lstm'), ('cnn', 'best_cnn'),
                           ('gru', 'best_gru'), ('informer', 'best_informer')]:
            if key in idx_data:
                configs[index][model] = {
                    'best_L': int(idx_data[key]['L']),
                    'config_name': idx_data[key]['config_name'],
                }
    return configs


# =============================================================================
# 2. LOAD PREDICTIONS (statt re-training)
# =============================================================================

def load_predictions(index_name, model_name, best_L, results_dir):
    """Liest die in Phase 3 gespeicherten Test-Predictions.

    pred_test.npz enthält:
      - y_actual_returns, y_pred_returns      (Log-Returns)
      - y_actual_prices,  y_pred_prices       (Preise nach Inverse-Transform)

    Returns:
        (y_actual_returns, y_pred_returns, y_actual_prices, y_pred_prices)
    """
    npz_path = os.path.join(results_dir, model_name.upper(), index_name,
                            f'L{best_L:02d}', 'pred_test.npz')
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as d:
        return (d['y_actual_returns'].copy(),
                d['y_pred_returns'].copy(),
                d['y_actual_prices'].copy(),
                d['y_pred_prices'].copy())


# =============================================================================
# 3. COMPUTE EXTENDED METRICS
# =============================================================================

def compute_metrics(y_actual, y_predicted, y_actual_prices=None, y_pred_prices=None):
    """Berechnet erweiterte Metriken aus tatsächlichen und vorhergesagten Werten.

    MAE, MSE, RMSE, Naive-Baseline und Directional Accuracy werden auf der
    Log-Return-Ebene berechnet (y_actual / y_predicted) — nur dort sind diese
    Größen sinnvoll: die Naive-Baseline (Vorhersage = 0) und der Richtungsanteil
    haben auf Preis-Ebene keine sinnvolle Interpretation.

    Das R² wird hingegen auf der zurücktransformierten **Preis-Ebene** berechnet
    (y_actual_prices / y_pred_prices), wenn diese übergeben werden. Damit ist der
    Wert vergleichbar mit der Literatur, die R² typischerweise auf Preis-Level
    angibt (z.B. Selvin et al. 2017, Nelson et al. 2017). Hinweis: Auf Preis-Ebene
    ist R² durch den starken Trend/Level der Kursreihe nach oben verzerrt — der
    Wert misst primär die Trendfolge, nicht die Prognosekraft der Tagesbewegung.
    Fallback auf Return-Ebene, falls keine Preis-Arrays vorliegen.
    """
    mae = np.mean(np.abs(y_actual - y_predicted))
    mse = np.mean((y_actual - y_predicted) ** 2)
    rmse = np.sqrt(mse)

    # Naive baseline: always predict 0 (= keine Bewegung) — Return-Ebene
    naive_mae = np.mean(np.abs(y_actual))
    mae_improvement = (naive_mae - mae) / naive_mae * 100 if naive_mae > 0 else 0.0

    # Directional accuracy: korrektes Vorzeichen? — Return-Ebene
    nonzero_mask = y_actual != 0
    if nonzero_mask.sum() > 0:
        correct_direction = np.sign(y_actual[nonzero_mask]) == np.sign(y_predicted[nonzero_mask])
        directional_accuracy = float(np.mean(correct_direction) * 100)
    else:
        directional_accuracy = 50.0

    # R-squared auf Preis-Ebene (zurücktransformiert) — vergleichbar mit Literatur.
    # Fallback auf Return-Ebene, falls Preis-Arrays fehlen.
    if y_actual_prices is not None and y_pred_prices is not None:
        r2_actual, r2_pred = y_actual_prices, y_pred_prices
    else:
        r2_actual, r2_pred = y_actual, y_predicted
    ss_res = np.sum((r2_actual - r2_pred) ** 2)
    ss_tot = np.sum((r2_actual - np.mean(r2_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'naive_baseline_mae': float(naive_mae),
        'mae_improvement_pct': float(mae_improvement),
        'directional_accuracy_pct': float(directional_accuracy),
        'r_squared': float(r_squared),
    }


# =============================================================================
# 4. PLOTS - PER INDEX
# =============================================================================

def plot_directional_accuracy(index_metrics, output_dir, index_name):
    """Bar plot: Directional Accuracy pro Modell + 50%-Baseline."""
    models = [m for m in MODELS if m in index_metrics]
    if not models:
        return
    accuracies = [index_metrics[m]['directional_accuracy_pct'] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, label='Random Baseline (50%)')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title(f'{index_name} - Directional Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, max(max(accuracies) + 5, 55))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_directional_accuracy.png'), dpi=150)
    plt.close()


def plot_mae_vs_baseline(index_metrics, output_dir, index_name):
    """Bar plot: Model MAE vs Naive Baseline MAE."""
    models = [m for m in MODELS if m in index_metrics]
    if not models:
        return
    maes = [index_metrics[m]['mae'] for m in models]
    baseline_mae = index_metrics[models[0]]['naive_baseline_mae']
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=baseline_mae, color='black', linestyle='--', linewidth=1.5,
               label=f'Naive Baseline (always 0): {baseline_mae:.6f}')

    for bar, mae_val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f'{mae_val:.6f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('MAE (Log Returns)', fontsize=12)
    ax.set_title(f'{index_name} - MAE vs Naive Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_mae_vs_baseline.png'), dpi=150)
    plt.close()


def plot_cumulative_returns(predictions, output_dir, index_name):
    """Cumulative returns: Actual vs alle Modell-Predictions."""
    models = [m for m in MODELS if m in predictions]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    y_actual = predictions[models[0]]['y_actual']
    cum_actual = np.cumsum(y_actual)
    ax.plot(cum_actual, 'k-', linewidth=2, label='Actual', alpha=0.9)

    for model in models:
        y_pred = predictions[model]['y_predicted']
        cum_pred = np.cumsum(y_pred)
        ax.plot(cum_pred, color=MODEL_COLORS[model], linewidth=1.5,
                label=MODEL_LABELS[model], alpha=0.7)

    ax.set_xlabel('Time Step (Days)', fontsize=12)
    ax.set_ylabel('Cumulative Log Return', fontsize=12)
    ax.set_title(f'{index_name} - Cumulative Returns: Actual vs Predicted',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_cumulative_returns.png'), dpi=150)
    plt.close()


def plot_returns_overlay(predictions, output_dir, index_name):
    """Returns time series mit Modell-Predictions überlagert (zoomed)."""
    models = [m for m in MODELS if m in predictions]
    if not models:
        return
    y_actual = predictions[models[0]]['y_actual']

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        y_pred = predictions[model]['y_predicted']
        ax.plot(y_actual, 'k-', alpha=0.5, linewidth=0.8, label='Actual')
        ax.plot(y_pred, color=MODEL_COLORS[model], alpha=0.8, linewidth=0.8,
                label=f'{MODEL_LABELS[model]} Predicted')
        ax.set_ylabel('Log Return', fontsize=10)
        ax.set_title(f'{MODEL_LABELS[model]} (L={predictions[model]["best_L"]})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step (Days)', fontsize=11)
    plt.suptitle(f'{index_name} - Returns: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_returns_overlay.png'), dpi=150)
    plt.close()


# =============================================================================
# 5. PLOTS - CROSS-INDEX
# =============================================================================

def plot_cross_index_heatmap(all_metrics, metric_key, title, filename, output_dir,
                              fmt='.1f', cmap='RdYlGn', vmin=None, vmax=None):
    """Heatmap: Indizes x Modelle für eine Metrik."""
    indices = sorted(all_metrics.keys())
    if not indices:
        return
    models = [m for m in MODELS if m in all_metrics[indices[0]]]

    data = np.zeros((len(indices), len(models)))
    for i, idx in enumerate(indices):
        for j, model in enumerate(models):
            data[i, j] = all_metrics[idx].get(model, {}).get(metric_key, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=[MODEL_LABELS[m] for m in models],
                yticklabels=indices,
                ax=ax, linewidths=0.5, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("POST-PROCESSING: Extended Metrics & Visualizations")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results')
    output_base = os.path.join(STEP_DIR, 'results')

    print("\n[1/4] Loading best configurations...")
    configs = load_best_configs(results_dir)
    print(f"  Loaded configs for {len(configs)} indices")

    for index in sorted(configs.keys()):
        for model in MODELS:
            if model in configs[index]:
                cfg = configs[index][model]
                print(f"  {index:12s} {model:10s}: L={cfg['best_L']:2d}, config={cfg['config_name']}")

    print("\n[2/4] Loading test predictions from Phase-3 npz files...")
    all_predictions = {}  # {index: {model: {y_actual, y_predicted, ...}}}
    all_metrics = {}

    for index in sorted(configs.keys()):
        all_predictions[index] = {}
        all_metrics[index] = {}
        for model in MODELS:
            if model not in configs[index]:
                continue
            best_L = configs[index][model]['best_L']
            loaded = load_predictions(index, model, best_L, results_dir)
            if loaded is None:
                print(f"  WARN: {index}/{model}/L{best_L:02d}/pred_test.npz fehlt — skip")
                continue
            y_actual_returns, y_pred_returns, y_actual_prices, y_pred_prices = loaded
            all_predictions[index][model] = {
                'y_actual':            y_actual_returns,
                'y_predicted':         y_pred_returns,
                'y_actual_prices':     y_actual_prices,
                'y_predicted_prices':  y_pred_prices,
                'best_L':              best_L,
            }
            metrics = compute_metrics(y_actual_returns, y_pred_returns,
                                      y_actual_prices, y_pred_prices)
            metrics['best_L'] = best_L
            all_metrics[index][model] = metrics
            print(f"  {index:12s} {MODEL_LABELS[model]:10s} L={best_L:2d}: "
                  f"DA={metrics['directional_accuracy_pct']:.1f}%, "
                  f"MAE={metrics['mae']:.6f}, "
                  f"Impr={metrics['mae_improvement_pct']:+.2f}%, "
                  f"R²={metrics['r_squared']:.4f}")

    print(f"\n[3/4] Generating plots...")
    for index in sorted(all_predictions.keys()):
        if not all_predictions[index]:
            continue
        idx_output = os.path.join(output_base, index)
        os.makedirs(idx_output, exist_ok=True)
        plot_directional_accuracy(all_metrics[index], idx_output, index)
        plot_mae_vs_baseline(all_metrics[index], idx_output, index)
        plot_cumulative_returns(all_predictions[index], idx_output, index)
        plot_returns_overlay(all_predictions[index], idx_output, index)
        print(f"  {index}: 4 plots saved")

    os.makedirs(output_base, exist_ok=True)
    plot_cross_index_heatmap(
        all_metrics, 'directional_accuracy_pct',
        'Directional Accuracy (%) by Index and Model',
        'cross_index_directional_accuracy.png', output_base,
        fmt='.1f', cmap='RdYlGn', vmin=45, vmax=55
    )
    plot_cross_index_heatmap(
        all_metrics, 'mae_improvement_pct',
        'MAE Improvement vs Naive Baseline (%) by Index and Model',
        'cross_index_mae_improvement.png', output_base,
        fmt='.2f', cmap='RdYlGn'
    )
    plot_cross_index_heatmap(
        all_metrics, 'r_squared',
        'R² (Preis-Ebene) by Index and Model',
        'cross_index_r_squared.png', output_base,
        fmt='.4f', cmap='RdYlGn', vmin=0.0, vmax=1.0
    )
    print(f"  Cross-index: 3 heatmaps saved")

    print(f"\n[4/4] Saving results...")
    for index in sorted(all_predictions.keys()):
        if not all_predictions[index]:
            continue
        pred_dir = os.path.join(output_base, 'predictions', index)
        os.makedirs(pred_dir, exist_ok=True)
        save_dict = {}
        for model in all_predictions[index]:
            preds = all_predictions[index][model]
            save_dict[f'{model}_y_actual']           = preds['y_actual']
            save_dict[f'{model}_y_predicted']        = preds['y_predicted']
            save_dict[f'{model}_y_actual_prices']    = preds['y_actual_prices']
            save_dict[f'{model}_y_predicted_prices'] = preds['y_predicted_prices']
            save_dict[f'{model}_best_L']             = np.array([preds['best_L']])
        np.savez_compressed(os.path.join(pred_dir, 'predictions.npz'), **save_dict)

    for index in sorted(all_metrics.keys()):
        if not all_metrics[index]:
            continue
        idx_output = os.path.join(output_base, index)
        os.makedirs(idx_output, exist_ok=True)
        with open(os.path.join(idx_output, 'extended_metrics.json'), 'w') as f:
            json.dump(all_metrics[index], f, indent=2)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'indices': sorted(all_metrics.keys()),
        'models': MODELS,
        'metrics': all_metrics,
    }
    with open(os.path.join(output_base, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_base}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Index':12s} {'Model':10s} {'DA%':>6s} {'MAE':>10s} "
          f"{'Baseline':>10s} {'Impr%':>7s} {'R²':>8s}")
    print("-" * 65)
    for index in sorted(all_metrics.keys()):
        for model in MODELS:
            if model in all_metrics[index]:
                m = all_metrics[index][model]
                print(f"{index:12s} {MODEL_LABELS[model]:10s} "
                      f"{m['directional_accuracy_pct']:5.1f}% "
                      f"{m['mae']:10.6f} "
                      f"{m['naive_baseline_mae']:10.6f} "
                      f"{m['mae_improvement_pct']:+6.2f}% "
                      f"{m['r_squared']:8.4f}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
