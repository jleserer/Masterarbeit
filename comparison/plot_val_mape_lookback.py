"""
Val-only-Variante von Abbildung 2 der Thesis (S. 49).
======================================================
Zeigt die Abhängigkeit des MAPE vom Lookback-Fenster L ausschließlich auf dem
Validierungs-Split — also auf denjenigen Daten, auf denen L* tatsächlich ausgewählt
wurde. Sterne markieren das je Modell gewählte L*.

Datenquelle: parameter_tuning/results/best_configurations.json
(val_curve_coarse je Modell: Val-MARE an den sieben Stützstellen
L ∈ {1, 10, 20, 30, 40, 50, 60}; MAPE = MARE · 100).

Output: comparison/results/val_mape_vs_lookback_<INDEX>.png

Usage:
    python comparison/plot_val_mape_lookback.py            # Default: NIKKEI
    python comparison/plot_val_mape_lookback.py SP500
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(STEP_DIR)
RESULTS_DIR = os.path.join(STEP_DIR, 'results')

MODELS = ['lstm', 'cnn', 'gru', 'informer']
MODEL_COLORS = {'lstm': '#2196F3', 'cnn': '#F44336', 'gru': '#4CAF50', 'informer': '#9C27B0'}
MODEL_LABELS = {'lstm': 'LSTM', 'cnn': 'CNN', 'gru': 'GRU', 'informer': 'Informer'}


def main(index_name='NIKKEI'):
    best_path = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results',
                             'best_configurations.json')
    with open(best_path, encoding='utf-8') as f:
        best = json.load(f)[index_name]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in MODELS:
        e = best[f'best_{m}']
        curve = e['val_curve_coarse']
        Ls = [c['L'] for c in curve]
        vals = [c['val_mare'] * 100 for c in curve]
        ax.plot(Ls, vals, 'o-', color=MODEL_COLORS[m], linewidth=1.6,
                markersize=5, label=MODEL_LABELS[m], alpha=0.85)
        ax.scatter([e['L']], [e['val_mare'] * 100], marker='*', s=220,
                   color=MODEL_COLORS[m], edgecolor='black', linewidth=0.6,
                   zorder=5)

    ax.set_xlabel('Lookback-Fenster L', fontsize=11)
    ax.set_ylabel('Val-MAPE in % (Preis-Ebene)', fontsize=11)
    ax.set_title(f'{index_name} — Validierungs-MAPE in Abhängigkeit des '
                 f'Lookback-Fensters L\n(Sterne: auf dem Validierungs-Split '
                 f'gewähltes L*)', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'val_mape_vs_lookback_{index_name}.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Gespeichert: {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'NIKKEI')
