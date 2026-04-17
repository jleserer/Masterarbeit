"""
Post-Processing: Extended Metrics & Visualizations
====================================================
Runs AFTER the pipeline completes. Re-trains only the best config per model
per index (32 runs total) and computes extended metrics:

- Directional Accuracy (% correct up/down predictions)
- Naive Baseline comparison (always-zero predictor)
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
import time
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lookback_window_sweep'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

from data_preparation import DataPreparator, create_sequences
from lookback_window_sweep import (
    build_lstm_model, build_cnn_model, build_gru_model,
    build_informer_sweep_model, train_and_evaluate_informer
)
from models.informer_model import predict_informer
from config import INDICES

MODELS = ['lstm', 'cnn', 'gru', 'informer']
MODEL_COLORS = {'lstm': '#2196F3', 'cnn': '#F44336', 'gru': '#4CAF50', 'informer': '#9C27B0'}
MODEL_LABELS = {'lstm': 'LSTM', 'cnn': 'CNN', 'gru': 'GRU', 'informer': 'Informer'}


# =============================================================================
# 1. LOAD BEST CONFIGURATIONS
# =============================================================================

def load_best_configs(results_dir):
    """Load best config per model per index from pipeline results.

    Returns:
        dict: {index: {model: {'config': {...}, 'best_L': int}}}
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
            print(f"WARNING: {index} not found in best_configurations.json, skipping")
            continue

        idx_data = all_best[index]
        configs[index] = {}

        # Load best configs from tuning
        for model, key in [('lstm', 'best_lstm'), ('cnn', 'best_cnn'),
                           ('gru', 'best_gru'), ('informer', 'best_informer')]:
            if key in idx_data:
                configs[index][model] = {
                    'config': idx_data[key]['config'],
                    'config_name': idx_data[key]['config_name'],
                }

        # Load best L from sweep results
        sweep_path = os.path.join(PROJECT_ROOT, 'lookback_window_sweep', 'results', index,
                                   'lookback_evaluation_results.json')
        if os.path.exists(sweep_path):
            with open(sweep_path, 'r') as f:
                sweep_data = json.load(f)

            for model, key in [('lstm', 'best_l_lstm'), ('cnn', 'best_l_cnn'),
                               ('gru', 'best_l_gru'), ('informer', 'best_l_informer')]:
                if model in configs[index] and key in sweep_data:
                    configs[index][model]['best_L'] = sweep_data[key]['l_value']
                elif model in configs[index]:
                    configs[index][model]['best_L'] = 60  # fallback
        else:
            print(f"WARNING: Sweep results not found for {index}, using L=60")
            for model in configs[index]:
                configs[index][model]['best_L'] = 60

    return configs


# =============================================================================
# 2. TRAIN & PREDICT
# =============================================================================

def train_and_predict(index_name, model_name, config, best_L, data_path):
    """Train one model with best config at best L. Returns (y_actual, y_predicted) as returns."""
    preparator = DataPreparator(data_path)
    train_data, val_data, test_data = preparator.load_and_prepare()

    X_train, y_train = create_sequences(train_data, best_L)
    X_test, y_test = create_sequences(test_data, best_L)
    n_features = X_train.shape[2]

    np.random.seed(42)
    tf.random.set_seed(42)

    if model_name == 'lstm':
        model = build_lstm_model(best_L, n_features, config)
        model.fit(X_train, y_train, epochs=config['epochs'],
                  batch_size=config['batch'], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        del model
        tf.keras.backend.clear_session()

    elif model_name == 'cnn':
        model = build_cnn_model(best_L, n_features, config)
        model.fit(X_train, y_train, epochs=config['epochs'],
                  batch_size=config['batch'], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        del model
        tf.keras.backend.clear_session()

    elif model_name == 'gru':
        model = build_gru_model(best_L, n_features, config)
        model.fit(X_train, y_train, epochs=config['epochs'],
                  batch_size=config['batch'], verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        del model
        tf.keras.backend.clear_session()

    elif model_name == 'informer':
        import torch
        torch.manual_seed(42)
        model = build_informer_sweep_model(best_L, n_features, config)
        train_and_evaluate_informer(model, X_train, y_train, X_test, y_test,
                                     config, best_L)
        y_pred = predict_informer(model, X_test, best_L)
        del model

    # Inverse transform to prices
    y_actual_prices = preparator.inverse_transform(
        y_test.reshape(-1, 1), split='test', lookback=best_L).flatten()
    y_pred_prices = preparator.inverse_transform(
        y_pred.reshape(-1, 1), split='test', lookback=best_L).flatten()

    return y_test, y_pred, y_actual_prices, y_pred_prices


# =============================================================================
# 3. COMPUTE EXTENDED METRICS
# =============================================================================

def compute_metrics(y_actual, y_predicted):
    """Compute extended metrics from actual and predicted returns.

    Returns:
        dict with all metrics
    """
    mae = np.mean(np.abs(y_actual - y_predicted))
    mse = np.mean((y_actual - y_predicted) ** 2)
    rmse = np.sqrt(mse)

    # Naive baseline: always predict 0
    naive_mae = np.mean(np.abs(y_actual))
    mae_improvement = (naive_mae - mae) / naive_mae * 100  # positive = better than baseline

    # Directional accuracy: did the model predict the correct sign?
    # Exclude days where actual return is exactly 0
    nonzero_mask = y_actual != 0
    if nonzero_mask.sum() > 0:
        correct_direction = np.sign(y_actual[nonzero_mask]) == np.sign(y_predicted[nonzero_mask])
        directional_accuracy = np.mean(correct_direction) * 100
    else:
        directional_accuracy = 50.0

    # R-squared
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
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
    """Bar plot: Directional Accuracy per model + 50% baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = [m for m in MODELS if m in index_metrics]
    accuracies = [index_metrics[m]['directional_accuracy_pct'] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    bars = ax.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, label='Random Baseline (50%)')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title(f'{index_name} - Directional Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(accuracies) + 5, 55))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_directional_accuracy.png'), dpi=150)
    plt.close()


def plot_mae_vs_baseline(index_metrics, output_dir, index_name):
    """Bar plot: Model MAE vs Naive Baseline MAE."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = [m for m in MODELS if m in index_metrics]
    maes = [index_metrics[m]['mae'] for m in models]
    baseline_mae = index_metrics[models[0]]['naive_baseline_mae']
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    bars = ax.bar(labels, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=baseline_mae, color='black', linestyle='--', linewidth=1.5,
               label=f'Naive Baseline (always 0): {baseline_mae:.6f}')

    for bar, mae_val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{mae_val:.6f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('MAE (Log Returns)', fontsize=12)
    ax.set_title(f'{index_name} - MAE vs Naive Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_mae_vs_baseline.png'), dpi=150)
    plt.close()


def plot_cumulative_returns(predictions, output_dir, index_name):
    """Cumulative returns: Actual vs all model predictions."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Actual cumulative returns
    models = [m for m in MODELS if m in predictions]
    y_actual = predictions[models[0]]['y_actual']
    cum_actual = np.cumsum(y_actual)
    ax.plot(cum_actual, 'k-', linewidth=2, label='Actual', alpha=0.9)

    # Model cumulative returns
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
    """Returns time series with model predictions overlaid (zoomed)."""
    models = [m for m in MODELS if m in predictions]
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
    """Heatmap: indices x models for a given metric."""
    indices = sorted(all_metrics.keys())
    models = [m for m in MODELS if m in all_metrics[indices[0]]]

    data = np.zeros((len(indices), len(models)))
    for i, idx in enumerate(indices):
        for j, model in enumerate(models):
            data[i, j] = all_metrics[idx][model][metric_key]

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

    data_dir = os.path.join(PROJECT_ROOT, 'stockData', 'preprocessedData')
    results_dir = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results')
    output_base = os.path.join(STEP_DIR, 'results')

    # Step 1: Load best configurations
    print("\n[1/5] Loading best configurations...")
    configs = load_best_configs(results_dir)
    print(f"  Loaded configs for {len(configs)} indices")

    for index in sorted(configs.keys()):
        for model in MODELS:
            if model in configs[index]:
                cfg = configs[index][model]
                print(f"  {index:12s} {model:10s}: L={cfg['best_L']:2d}, config={cfg['config_name']}")

    # Step 2: Train & Predict
    print(f"\n[2/5] Training {sum(len(v) for v in configs.values())} models...")
    all_predictions = {}  # {index: {model: {y_actual, y_predicted, ...}}}
    all_metrics = {}      # {index: {model: {metric: value}}}

    for index in sorted(configs.keys()):
        print(f"\n  === {index} ===")
        data_path = os.path.join(data_dir, INDICES[index])
        all_predictions[index] = {}
        all_metrics[index] = {}

        for model in MODELS:
            if model not in configs[index]:
                continue

            cfg = configs[index][model]
            best_L = cfg['best_L']
            config = cfg['config']
            t0 = time.time()

            print(f"    {MODEL_LABELS[model]:10s} (L={best_L})...", end=" ", flush=True)

            y_actual, y_pred, y_actual_prices, y_pred_prices = train_and_predict(
                index, model, config, best_L, data_path
            )

            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s")

            all_predictions[index][model] = {
                'y_actual': y_actual,
                'y_predicted': y_pred,
                'y_actual_prices': y_actual_prices,
                'y_predicted_prices': y_pred_prices,
                'best_L': best_L,
            }

    # Step 3: Compute metrics
    print(f"\n[3/5] Computing extended metrics...")
    for index in sorted(all_predictions.keys()):
        all_metrics[index] = {}
        for model in all_predictions[index]:
            preds = all_predictions[index][model]
            metrics = compute_metrics(preds['y_actual'], preds['y_predicted'])
            metrics['best_L'] = preds['best_L']
            all_metrics[index][model] = metrics

            print(f"  {index:12s} {MODEL_LABELS[model]:10s}: "
                  f"DA={metrics['directional_accuracy_pct']:.1f}%, "
                  f"MAE={metrics['mae']:.6f}, "
                  f"Baseline={metrics['naive_baseline_mae']:.6f}, "
                  f"Impr={metrics['mae_improvement_pct']:+.2f}%, "
                  f"R²={metrics['r_squared']:.4f}")

    # Step 4: Generate plots
    print(f"\n[4/5] Generating plots...")

    # Per-index plots
    for index in sorted(all_predictions.keys()):
        idx_output = os.path.join(output_base, index)
        os.makedirs(idx_output, exist_ok=True)

        plot_directional_accuracy(all_metrics[index], idx_output, index)
        plot_mae_vs_baseline(all_metrics[index], idx_output, index)
        plot_cumulative_returns(all_predictions[index], idx_output, index)
        plot_returns_overlay(all_predictions[index], idx_output, index)
        print(f"  {index}: 4 plots saved")

    # Cross-index plots
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
        'R² by Index and Model',
        'cross_index_r_squared.png', output_base,
        fmt='.4f', cmap='RdYlGn'
    )
    print(f"  Cross-index: 3 heatmaps saved")

    # Step 5: Save results
    print(f"\n[5/5] Saving results...")

    # Save predictions as npz
    for index in sorted(all_predictions.keys()):
        pred_dir = os.path.join(output_base, 'predictions', index)
        os.makedirs(pred_dir, exist_ok=True)

        save_dict = {}
        for model in all_predictions[index]:
            preds = all_predictions[index][model]
            save_dict[f'{model}_y_actual'] = preds['y_actual']
            save_dict[f'{model}_y_predicted'] = preds['y_predicted']
            save_dict[f'{model}_y_actual_prices'] = preds['y_actual_prices']
            save_dict[f'{model}_y_predicted_prices'] = preds['y_predicted_prices']
            save_dict[f'{model}_best_L'] = np.array([preds['best_L']])
        np.savez_compressed(os.path.join(pred_dir, 'predictions.npz'), **save_dict)

    # Save metrics JSON per index
    for index in sorted(all_metrics.keys()):
        idx_output = os.path.join(output_base, index)
        os.makedirs(idx_output, exist_ok=True)
        with open(os.path.join(idx_output, 'extended_metrics.json'), 'w') as f:
            json.dump(all_metrics[index], f, indent=2)

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'indices': sorted(all_metrics.keys()),
        'models': MODELS,
        'metrics': all_metrics,
    }
    with open(os.path.join(output_base, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_base}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Index':12s} {'Model':10s} {'DA%':>6s} {'MAE':>10s} {'Baseline':>10s} {'Impr%':>7s} {'R²':>8s}")
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
