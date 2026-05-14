"""
Aggregiert pro (Index, Modell) alle Hyperparameter-Tuning-Runs in EINE Tabelle.

Output pro Kombination (4 Modelle x 6 Indizes = 24 Tabellen):
  - <results>/LSTM/SP500/tuning_table.csv    (alle 48 Configs, sortiert nach Val Loss)
  - <results>/LSTM/SP500/tuning_table.md     (Markdown, direkt in Thesis einbindbar)
  - <results>/LSTM/SP500/tuning_table.png    (Rank-Plot: Val-Loss + Val-MARE pro Config)

Plus eine Zusammenfassung:
  - <results>/tuning_summary.md               (beste Config pro Index x Modell)

Nutzt ausschließlich schon vorhandene results.json — kein Re-Training.

Usage:
  python parameter_tuning/generate_tuning_tables.py
  python parameter_tuning/generate_tuning_tables.py --results-dir parameter_tuning/results
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_INDICES


MODEL_DIRS = ['LSTM', 'CNN', 'GRU', 'INFORMER']
# Spalten-Reihenfolge pro Modell (erste: Identifier; dann Hyperparameter; dann Metriken)
MODEL_HP_COLUMNS = {
    'LSTM':     ['dropout', 'dense_units', 'lr', 'batch', 'epochs'],
    'CNN':      ['kernel', 'pool', 'dropout', 'batch', 'epochs'],
    'GRU':      ['dropout', 'dense_units', 'lr', 'batch', 'epochs'],
    'INFORMER': ['d_model', 'n_heads', 'dropout', 'lr', 'batch', 'epochs'],
}
METRIC_COLUMNS = ['train_loss', 'train_mare', 'val_loss', 'val_mare']


def collect_configs(model_dir: Path, model_name: str):
    """Sammelt alle results.json unter <model_dir> in eine Liste von flachen Dicts."""
    rows = []
    if not model_dir.exists():
        return rows
    for config_dir in sorted(model_dir.iterdir()):
        rj = config_dir / 'results.json'
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception as e:
            print(f"  WARN: cannot parse {rj}: {e}")
            continue
        if 'metrics' not in data:
            continue

        row = {'config_name': data.get('config_name', config_dir.name)}
        for hp in MODEL_HP_COLUMNS[model_name]:
            row[hp] = data.get('config', {}).get(hp)
        for m in METRIC_COLUMNS:
            v = data.get('metrics', {}).get(m)
            row[m] = float(v) if v is not None else np.nan
        row['training_time_s'] = data.get('training_time')
        row['epochs_trained'] = data.get('epochs_trained')
        rows.append(row)
    return rows


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, float_format='%.6f')


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Einfache Markdown-Tabelle ohne externe Dependency."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |"]
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def write_markdown(df: pd.DataFrame, path: Path, title: str):
    """Markdown-Tabelle, sortiert nach Val Loss. 6 Nachkommastellen."""
    lines = [f"# {title}", "", f"_{len(df)} Konfigurationen. Sortiert nach `val_loss` (Selektionsmetrik)._", ""]
    display = df.copy()
    for col in METRIC_COLUMNS:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    if 'training_time_s' in display:
        display['training_time_s'] = display['training_time_s'].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "")
    lines.append(_df_to_markdown(display))
    path.write_text("\n".join(lines), encoding='utf-8')


def plot_ranking(df: pd.DataFrame, path: Path, title: str):
    """Scatter: val_loss und val_mare pro Config (sortiert). Eine Zeile pro Config."""
    n = len(df)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.18 * n + 1)),
                             gridspec_kw={'wspace': 0.25})

    # Val Loss
    axes[0].barh(range(n), df['val_loss'].values, color='#1E88E5', edgecolor='black', linewidth=0.3)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(df['config_name'].values, fontsize=7)
    axes[0].set_xlabel('Val Loss (MSE auf Log-Returns)', fontsize=10)
    axes[0].set_title('Val Loss pro Config (sortiert, klein = besser)', fontsize=11, fontweight='bold')
    axes[0].grid(True, axis='x', alpha=0.3)
    axes[0].invert_yaxis()  # beste Config oben

    # Val MARE
    axes[1].barh(range(n), df['val_mare'].values, color='#8E24AA', edgecolor='black', linewidth=0.3)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([''] * n)  # Labels nur links
    axes[1].set_xlabel('Val MARE (auf Preis-Ebene)', fontsize=10)
    axes[1].set_title('Val MARE pro Config (sortiert nach Val Loss)', fontsize=11, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.3)
    axes[1].invert_yaxis()

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def build_one_table(results_dir: Path, model_name: str, index_name: str, lookback: int = None):
    """Tabellen für eine (Index, Modell, [L])-Kombination.

    Wenn `lookback` gegeben ist, wird ein L<L>-Unterordner genutzt und eine
    Tabelle für genau dieses L erzeugt. Sonst wird der (legacy) flache
    Ordner verwendet.
    """
    if lookback is not None:
        model_dir = results_dir / model_name / index_name / f'L{lookback:02d}'
    else:
        model_dir = results_dir / model_name / index_name
    rows = collect_configs(model_dir, model_name)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    cols = ['config_name'] + MODEL_HP_COLUMNS[model_name] + METRIC_COLUMNS + ['training_time_s', 'epochs_trained']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('val_loss', na_position='last').reset_index(drop=True)

    title = f"{index_name} — {model_name}" + (f" — L={lookback}" if lookback is not None else "")
    write_csv(df, model_dir / 'tuning_table.csv')
    write_markdown(df, model_dir / 'tuning_table.md', title)
    plot_ranking(df, model_dir / 'tuning_table.png', title)
    return df


def build_summary(results_dir: Path, best_per_cell: dict):
    """Übersicht mit bester Config je (Index, Modell[, L]).

    Keys in best_per_cell können entweder (idx, model) ODER (idx, model, L) sein.
    """
    summary_rows = []
    for key, df in sorted(best_per_cell.items(), key=lambda kv: (str(kv[0][0]),
                                                                  str(kv[0][1]),
                                                                  (kv[0][2] if len(kv[0]) > 2 else -1) or -1)):
        if df is None or df.empty:
            continue
        idx = key[0]
        model = key[1]
        L = key[2] if len(key) > 2 else None
        best = df.iloc[0].to_dict()
        row = {
            'index': idx,
            'model': model,
            'best_config': best.get('config_name'),
            'val_loss': best.get('val_loss'),
            'val_mare': best.get('val_mare'),
            'train_loss': best.get('train_loss'),
            'train_mare': best.get('train_mare'),
            'n_configs': len(df),
        }
        if L is not None:
            row['L'] = L
        summary_rows.append(row)
    if not summary_rows:
        return
    sdf = pd.DataFrame(summary_rows)
    # L vorn
    if 'L' in sdf.columns:
        cols = ['index', 'model', 'L', 'best_config', 'val_loss', 'val_mare',
                'train_loss', 'train_mare', 'n_configs']
        sdf = sdf[[c for c in cols if c in sdf.columns]]
    sdf.to_csv(results_dir / 'tuning_summary.csv', index=False, float_format='%.6f')

    lines = ["# Tuning Summary — beste Config pro (Index, Modell)", "",
             "_Selektionsmetrik: `val_loss`. Train/Val MARE auf Preis-Ebene._", ""]
    display = sdf.copy()
    for col in ['train_loss', 'train_mare', 'val_loss', 'val_mare']:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    lines.append(_df_to_markdown(display))
    (results_dir / 'tuning_summary.md').write_text("\n".join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                        default=str(SCRIPT_DIR / 'results'),
                        help='Parameter-Tuning results Verzeichnis')
    parser.add_argument('--lookbacks', type=str, default='1-60',
                        help='Welche L-Werte auswerten (z.B. "1-60" oder "10,20,30"). '
                             'Leer/none = legacy (flache Struktur ohne L<L>/).')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"FEHLER: {results_dir} existiert nicht.")
        sys.exit(1)

    # Parse lookbacks
    lookbacks = []
    if args.lookbacks and args.lookbacks.lower() != 'none':
        for part in args.lookbacks.split(','):
            if '-' in part:
                a, b = part.split('-')
                lookbacks.extend(range(int(a), int(b) + 1))
            else:
                lookbacks.append(int(part))

    if not lookbacks:
        lookbacks = [None]  # legacy-Modus

    print(f"Lese Ergebnisse aus: {results_dir}")
    print(f"Lookbacks:           {lookbacks if lookbacks[0] is not None else 'legacy (flach)'}")
    print(f"{'Index':12s} {'Model':10s} {'L':>4s} {'N_configs':>10s} "
          f"{'BestValLoss':>14s} {'BestValMARE':>14s}")
    print("-" * 85)

    best_per_cell = {}
    total = 0
    for L in lookbacks:
        for idx in ALL_INDICES:
            for model in MODEL_DIRS:
                df = build_one_table(results_dir, model, idx, lookback=L)
                if df is None or df.empty:
                    continue
                best_per_cell[(idx, model, L)] = df
                total += len(df)
                vl = df['val_loss'].iloc[0]
                vm = df['val_mare'].iloc[0] if 'val_mare' in df else float('nan')
                L_disp = f'{L:>4d}' if L is not None else ' -- '
                print(f"{idx:12s} {model:10s} {L_disp} {len(df):>10d} "
                      f"{vl:>14.6f} {vm:>14.6f}")

    print("-" * 85)
    print(f"Summe: {total} Configs ausgewertet.")

    build_summary(results_dir, best_per_cell)
    print(f"\nSummary geschrieben: {results_dir/'tuning_summary.csv'}")
    print(f"Summary geschrieben: {results_dir/'tuning_summary.md'}")


if __name__ == '__main__':
    main()
