"""
MAPE-Variante von Abbildung 10 der Thesis (S. 62).
===================================================
Gruppierter Balkenplot: MAPE der vier DL-Modelle und der SMA-20-Baseline pro
Index — auf der Preis-Ebene, konsistent zu Tabelle 4 und der Baseline-Methodik.

Datenquellen (müssen zuvor erzeugt sein):
  - comparison/results/summary.json       (post_processing.py: mape_pct)
  - comparison/results/sma_baseline.json  (sma_baseline.py: mape_price_pct)

Output: comparison/results/mape_sma_vs_dl.png

Usage:
    python comparison/plot_sma_mape.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(STEP_DIR, 'results')

MODELS = ['lstm', 'gru', 'cnn', 'informer']
MODEL_COLORS = {'lstm': '#2196F3', 'cnn': '#F44336', 'gru': '#4CAF50', 'informer': '#9C27B0'}
MODEL_LABELS = {'lstm': 'LSTM', 'cnn': 'CNN', 'gru': 'GRU', 'informer': 'Informer'}
SMA_COLOR = '#757575'

INDEX_ORDER = ['SP500', 'DAX', 'NASDAQ', 'HANG_SENG', 'NIKKEI', '10Y_Bond']
INDEX_LABELS = {'SP500': 'S&P 500', 'DAX': 'DAX', 'NASDAQ': 'NASDAQ',
                'HANG_SENG': 'Hang Seng', 'NIKKEI': 'Nikkei 225',
                '10Y_Bond': '10Y Bond'}


def main():
    with open(os.path.join(RESULTS_DIR, 'summary.json'), encoding='utf-8') as f:
        metrics = json.load(f)['metrics']
    with open(os.path.join(RESULTS_DIR, 'sma_baseline.json'), encoding='utf-8') as f:
        sma = json.load(f)

    x = np.arange(len(INDEX_ORDER))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(MODELS):
        vals = [metrics[ix][model]['mape_pct'] for ix in INDEX_ORDER]
        ax.bar(x + (i - 2) * width, vals, width, color=MODEL_COLORS[model],
               alpha=0.85, edgecolor='black', linewidth=0.4,
               label=MODEL_LABELS[model])
    sma_vals = [sma[ix]['mape_price_pct'] for ix in INDEX_ORDER]
    bars_sma = ax.bar(x + 2 * width, sma_vals, width, color=SMA_COLOR,
                      alpha=0.85, edgecolor='black', linewidth=0.4,
                      hatch='//', label='SMA-20')

    for bar, val in zip(bars_sma, sma_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([INDEX_LABELS[ix] for ix in INDEX_ORDER], fontsize=11)
    ax.set_ylabel('MAPE in % (Preis-Ebene, Test-Split)', fontsize=12)
    ax.set_title('MAPE der vier DL-Modelle und der SMA-20-Baseline pro Index',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=5, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'mape_sma_vs_dl.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Gespeichert: {out}')


if __name__ == '__main__':
    main()
