"""
Zoom-Plots mit echten Datumsangaben auf der x-Achse.

Standalone, read-only — nutzt die bereits geschriebenen `pred_test.npz` aus
`parameter_tuning/results/<MODEL>/<INDEX>/L<L>/` und mappt die Sequenz-Indizes
auf die original Datumsangaben aus den CSVs in
`stockData/preprocessedData/<INDEX>_historical_data.csv`.

Schreibt nach `comparison/results/<INDEX>/zoom_plots/zoom_<MODEL>_L<L>.png`.
Bestehende Pipeline-Daten/Plots werden NICHT modifiziert.

Usage:
  python comparison/plot_with_dates.py --index SP500 --L 1
  python comparison/plot_with_dates.py --index SP500            # nutzt best_L aus best_configurations.json
  python comparison/plot_with_dates.py --index ALL --L best     # alle Indizes mit best_L pro Modell
  python comparison/plot_with_dates.py --index SP500 --L 1 --models LSTM,CNN

Datums-Mapping:
  N raw prices, N-1 log-returns (returns[k] = ln(P(k+1)/P(k)))
  Splits auf Returns: 65/15/20  ->  test = returns[val_end : N-1]
  Sequenz k mit Lookback L  ->  zeigt y_pred für Datum prices[val_end + L + k + 1]
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
import matplotlib.dates as mdates

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import INDICES, ALL_INDICES  # noqa: E402

TUNING_RESULTS_DIR = PROJECT_ROOT / 'parameter_tuning' / 'results'
COMPARE_RESULTS_DIR = PROJECT_ROOT / 'comparison' / 'results'
CSV_DIR = PROJECT_ROOT / 'stockData' / 'preprocessedData'

MODEL_META = {
    'LSTM':     {'color': '#2196F3'},
    'CNN':      {'color': '#F44336'},
    'GRU':      {'color': '#4CAF50'},
    'INFORMER': {'color': '#9C27B0'},
}

# Deutsche Monatsnamen — locale-unabhängig (Windows liefert 'de_DE' nicht zuverlässig).
MONATE_KURZ = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
               'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']


def _format_date_de(d) -> str:
    """Formatiert ein Datum als DD.MM.YYYY"""
    ts = pd.Timestamp(d)
    return f"{ts.day:02d}.{ts.month:02d}.{ts.year}"


def load_test_dates(index_name, lookback):
    """Liest die CSV und extrahiert genau die Datumsangaben, die zu den Test-
    Sequenz-Predictions passen (deckt sich mit DataPreparator-Split-Logik).

    Returns: np.array of np.datetime64, gleiche Länge wie pred_test.npz Arrays.
    """
    csv_path = CSV_DIR / INDICES[index_name]
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    # Yahoo-Format: Spalte 1 heißt 'Price' aber enthält Datum.
    # Header-Zeilen 1+2 (Ticker, "Date,,,") werden durch pd.to_numeric -> NaN -> dropna
    # weggefiltert, exakt wie in DataPreparator.
    df = pd.read_csv(csv_path)
    close = pd.to_numeric(df['Close'], errors='coerce')
    valid_mask = close.notna()
    dates = pd.to_datetime(df.loc[valid_mask, 'Price'], errors='coerce').values
    prices = close.loc[valid_mask].values

    n_prices = len(prices)
    n_returns = n_prices - 1  # returns[k] = ln(P(k+1)/P(k))

    # DataPreparator.load_and_prepare Z. 65-71: Splits auf Returns 65/15/20
    train_end = int(0.65 * n_returns)
    val_end = train_end + int(0.15 * n_returns)

    # Test-Sequenzen: returns[val_end:n_returns] -> n_test_returns Stück
    # Nach create_sequences mit lookback L gibt es n_test_returns - L Sequenzen.
    # Sequenz k -> y entspricht returns[val_end + L + k] -> Preis-Index val_end + L + k + 1
    n_test_returns = n_returns - val_end
    n_seq = n_test_returns - lookback

    if n_seq <= 0:
        raise ValueError(f"Lookback {lookback} >= Anzahl Test-Returns {n_test_returns} "
                         f"für {index_name}")

    start_price_idx = val_end + lookback + 1
    end_price_idx = start_price_idx + n_seq
    test_dates = dates[start_price_idx:end_price_idx]

    return test_dates


def pick_windows(n_total, window_size):
    if n_total <= window_size:
        return [(0, n_total)]
    mid = max(0, (n_total - window_size) // 2)
    end = max(0, n_total - window_size)
    return [(0, window_size), (mid, mid + window_size), (end, end + window_size)]


def _center_narrow(wide_start, wide_end, deep_size, n_total):
    wide_mid = (wide_start + wide_end) // 2
    ds = max(0, wide_mid - deep_size // 2)
    de = min(n_total, ds + deep_size)
    ds = max(0, de - deep_size)
    return ds, de


def _format_date_axis(ax, dates_slice):
    """Setzt eine sinnvolle Datums-Achse je nach Spannweite (deutsche Beschriftung).

    Tick-Format:
      > 3 Jahre  -> Jahr (YYYY)
      > 1 Jahr   -> "Mon YYYY" (deutscher Monatskurzname)
      > 90 Tage  -> "Mon YYYY"
      > 30 Tage  -> "TT. Mon" (alle 2 Wochen, Mo)
      sonst      -> "TT. Mon" (wöchentlich, Mo)
    """
    if len(dates_slice) == 0:
        return
    span_days = (dates_slice[-1] - dates_slice[0]) / np.timedelta64(1, 'D')

    if span_days > 365 * 3:
        locator = mdates.YearLocator()
        def fmt_fn(x, pos=None):
            return f"{mdates.num2date(x).year}"
    elif span_days > 365:
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        def fmt_fn(x, pos=None):
            dt = mdates.num2date(x)
            return f"{MONATE_KURZ[dt.month - 1]} {dt.year}"
    elif span_days > 90:
        locator = mdates.MonthLocator()
        def fmt_fn(x, pos=None):
            dt = mdates.num2date(x)
            return f"{MONATE_KURZ[dt.month - 1]} {dt.year}"
    elif span_days > 30:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)
        def fmt_fn(x, pos=None):
            dt = mdates.num2date(x)
            return f"{dt.day:02d}. {MONATE_KURZ[dt.month - 1]}"
    else:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
        def fmt_fn(x, pos=None):
            dt = mdates.num2date(x)
            return f"{dt.day:02d}. {MONATE_KURZ[dt.month - 1]}"

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_fn))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha('right')


def plot_zoom_with_dates(y_actual, y_pred, dates, label, color, output_path,
                          wide=150, deep=40, dpi=300, title_prefix=''):
    """Erzeugt einen mehrteiligen Zoom-Plot (Überblick + 3× Weiter Zoom + 3× Detail-Zoom)
    mit echten Datumsangaben auf der x-Achse (DD.MM.YYYY, deutsche Monatsnamen).
    """
    n = len(y_actual)
    if len(dates) != n:
        raise ValueError(f"Date-Array Länge {len(dates)} != y-Länge {n}")

    wide_pairs = pick_windows(n, wide)[:3]
    deep_pairs = [_center_narrow(ws, we, deep, n) for (ws, we) in wide_pairs]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.1, 1.1],
                          hspace=0.50, wspace=0.22)

    # --- Zeile 1: Überblick ---
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates, y_actual, 'k-', linewidth=1.0, alpha=0.85,
             label='Tatsächlich', zorder=2)
    ax0.plot(dates, y_pred, color=color, linewidth=0.9, alpha=0.75,
             label=f'{label} (Vorhersage)', zorder=1)
    for (s, e) in wide_pairs:
        ax0.axvspan(dates[s], dates[min(e, n - 1)], color='gold', alpha=0.18, zorder=0)
    for (s, e) in deep_pairs:
        ax0.axvspan(dates[s], dates[min(e, n - 1)], color='red', alpha=0.25, zorder=0)
    ax0.set_title(f'{title_prefix} — Überblick ({_format_date_de(dates[0])} bis '
                  f'{_format_date_de(dates[-1])}, N={n} Test-Handelstage); '
                  f'gelb = Weiter Zoom {wide} Tage, rot = Detail-Zoom {deep} Tage',
                  fontsize=13, fontweight='bold')
    ax0.set_xlabel('Datum'); ax0.set_ylabel('Preis')
    ax0.legend(loc='lower right'); ax0.grid(True, alpha=0.3)
    _format_date_axis(ax0, dates)

    # --- Zeile 2: Weiter Zoom ---
    for i, (s, e) in enumerate(wide_pairs):
        ax = fig.add_subplot(gs[1, i])
        dxs = dates[s:e]
        ax.plot(dxs, y_actual[s:e], 'k-', linewidth=1.2, alpha=0.9, label='Tatsächlich')
        ax.plot(dxs, y_pred[s:e], color=color, linewidth=1.1, alpha=0.85,
                label=f'{label} (Vorhersage)')
        ax.fill_between(dxs, y_actual[s:e], y_pred[s:e], color=color, alpha=0.15)
        ds, de = deep_pairs[i]
        ax.axvspan(dates[ds], dates[min(de, n - 1)], color='red', alpha=0.15, zorder=0)
        ax.set_title(f'Weiter Zoom {i+1}: {_format_date_de(dxs[0])} bis '
                     f'{_format_date_de(dxs[-1])} ({e-s} Handelstage)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Datum'); ax.set_ylabel('Preis')
        ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)
        _format_date_axis(ax, dxs)

    # --- Zeile 3: Detail-Zoom — tägliche Auflösung ---
    for i, (s, e) in enumerate(deep_pairs):
        ax = fig.add_subplot(gs[2, i])
        dxs = dates[s:e]
        ax.plot(dxs, y_actual[s:e], 'k-o', linewidth=1.3, alpha=0.95,
                markersize=3.5, label='Tatsächlich')
        ax.plot(dxs, y_pred[s:e], color=color, marker='s', linewidth=1.2,
                alpha=0.9, markersize=3.5, label=f'{label} (Vorhersage)')
        ax.fill_between(dxs, y_actual[s:e], y_pred[s:e], color=color, alpha=0.18)
        for j in range(len(dxs)):
            ax.plot([dxs[j], dxs[j]], [y_actual[s+j], y_pred[s+j]],
                    color=color, alpha=0.35, linewidth=1.2, zorder=1)
        ax.set_title(f'Detail-Zoom {i+1}: {_format_date_de(dxs[0])} bis '
                     f'{_format_date_de(dxs[-1])} (tägliche Auflösung)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Datum'); ax.set_ylabel('Preis')
        ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)
        _format_date_axis(ax, dxs)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_path}", flush=True)


def _resolve_best_L(index_name, model_type):
    """Liest best_L aus best_configurations.json für (index, model)."""
    bc = TUNING_RESULTS_DIR / 'best_configurations.json'
    if not bc.is_file():
        return None
    with open(bc) as f:
        d = json.load(f)
    entry = d.get(index_name, {}).get(f'best_{model_type.lower()}')
    return int(entry['L']) if entry else None


def plot_one(index_name, model_type, lookback, wide=150, deep=40, dpi=300):
    """Plottet einen Zoom-Plot für (index, model, L) mit Datumsachse."""
    npz_path = (TUNING_RESULTS_DIR / model_type.upper() / index_name /
                f'L{lookback:02d}' / 'pred_test.npz')
    if not npz_path.is_file():
        print(f"  WARN: {npz_path} fehlt — skip {index_name}/{model_type}/L{lookback}")
        return False

    with np.load(npz_path) as d:
        y_actual = d['y_actual_prices'].copy()
        y_pred = d['y_pred_prices'].copy()

    test_dates = load_test_dates(index_name, lookback)
    if len(test_dates) != len(y_actual):
        print(f"  WARN: Datums-Länge {len(test_dates)} != y-Länge {len(y_actual)} "
              f"für {index_name}/{model_type}/L{lookback}")
        return False

    out_dir = COMPARE_RESULTS_DIR / index_name / 'zoom_plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'zoom_{model_type.upper()}_L{lookback:02d}.png'

    plot_zoom_with_dates(
        y_actual=y_actual, y_pred=y_pred, dates=test_dates,
        label=model_type.upper(), color=MODEL_META[model_type.upper()]['color'],
        output_path=str(out_path),
        wide=wide, deep=deep, dpi=dpi,
        title_prefix=f'{index_name} — {model_type.upper()} (L={lookback})',
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', required=True,
                        help='Index-Name oder "ALL" für alle Indizes')
    parser.add_argument('--L', default='best',
                        help='Lookback-L (int) oder "best" (Default) -> liest best_L '
                             'pro Modell aus best_configurations.json')
    parser.add_argument('--models', default='LSTM,CNN,GRU,INFORMER',
                        help='Komma-getrennte Modell-Liste (Default: alle 4)')
    parser.add_argument('--wide', type=int, default=150)
    parser.add_argument('--deep', type=int, default=40)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    indices = ALL_INDICES if args.index.upper() == 'ALL' else [args.index]
    models = [m.strip().upper() for m in args.models.split(',') if m.strip()]
    use_best = (str(args.L).lower() == 'best')
    fixed_L = None if use_best else int(args.L)

    total = 0
    ok = 0
    for idx in indices:
        if idx not in INDICES:
            print(f"WARN: Unbekannter Index '{idx}' — skip")
            continue
        for m in models:
            if m not in MODEL_META:
                print(f"WARN: Unbekanntes Modell '{m}' — skip")
                continue
            total += 1
            if use_best:
                L = _resolve_best_L(idx, m)
                if L is None:
                    print(f"  WARN: best_L für {idx}/{m} nicht in best_configurations.json "
                          f"— skip (nutze --L explizit)")
                    continue
            else:
                L = fixed_L
            if plot_one(idx, m, L, wide=args.wide, deep=args.deep, dpi=args.dpi):
                ok += 1

    print(f"\n{ok}/{total} Plots erzeugt unter {COMPARE_RESULTS_DIR}/<INDEX>/zoom_plots/")


if __name__ == '__main__':
    main()
