"""Alle 8 Indizes × 60 L-Werte parallel neu berechnen und Plots generieren.

Ablauf:
  1. Für jeden Index alle 60 L-Werte parallel trainieren (L_*.json erzeugen)
  2. Pro Index --collect aufrufen → MAE-Plot + Prediction-Plots (inkl. Informer)

Bereits vorhandene L_*.json werden übersprungen (Resume-Support).

Optimiert für AMD AI MAX+ 395 (32 logische Kerne) — 16 parallele Worker.
"""
import subprocess
import sys
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
SWEEP_SCRIPT = os.path.join(SCRIPT_DIR, 'lookback_window_sweep.py')
BEST_CONFIGS = os.path.join(PROJECT_ROOT, 'parameter_tuning', 'results', 'best_configurations.json')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

INDICES = [
    'SP500', 'DAX', 'NASDAQ', 'FTSE100',
    'HANG_SENG', 'NIKKEI', '10Y_Bond', '30Y_Bond',
]

MAX_WORKERS = 16


def get_env():
    """Umgebungsvariablen für TensorFlow-Worker."""
    env = os.environ.copy()
    env['TF_NUM_INTRAOP_THREADS'] = '2'
    env['TF_NUM_INTEROP_THREADS'] = '1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env['OMP_NUM_THREADS'] = '2'
    return env


def get_missing_tasks():
    """Ermittle fehlende (Index, L)-Paare anhand vorhandener L_*.json."""
    tasks = []
    for index in INDICES:
        index_dir = os.path.join(RESULTS_DIR, index)
        existing = set()
        if os.path.isdir(index_dir):
            for f in os.listdir(index_dir):
                if f.startswith('L_') and f.endswith('.json'):
                    try:
                        L = int(f.split('_')[1].split('.')[0])
                        existing.add(L)
                    except (ValueError, IndexError):
                        pass
        missing = [L for L in range(1, 61) if L not in existing]
        tasks.extend((index, L) for L in missing)
        if existing:
            print(f"  {index}: {len(missing)} fehlend, {len(existing)} vorhanden")
        else:
            print(f"  {index}: alle 60 L-Werte fehlen")
    return tasks


def run_one(index, L):
    """Einen einzelnen (Index, L)-Lauf als Subprocess starten."""
    cmd = [sys.executable, '-u', SWEEP_SCRIPT,
           '--index', index, '--lookback', str(L),
           '--best-configs', BEST_CONFIGS]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=get_env(), cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    status = 'OK' if result.returncode == 0 else 'FAIL'
    if status == 'FAIL':
        print(f"  FAIL {index}/L={L}: {result.stderr[-300:]}", flush=True)
    return index, L, status, elapsed


def collect_one(index):
    """Ergebnisse eines Index sammeln und alle Plots generieren."""
    cmd = [sys.executable, '-u', SWEEP_SCRIPT,
           '--index', index, '--collect',
           '--best-configs', BEST_CONFIGS]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=get_env(), cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    status = 'OK' if result.returncode == 0 else 'FAIL'
    if status == 'FAIL':
        print(f"  FAIL collect {index}: {result.stderr[-500:]}", flush=True)
    return index, status, elapsed


def main():
    parser = argparse.ArgumentParser(description='Alle Lookback-Sweeps neu berechnen')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Anzahl paralleler Worker (default: {MAX_WORKERS})')
    parser.add_argument('--collect-only', action='store_true',
                        help='Nur Plots generieren (L_*.json müssen vorhanden sein)')
    parser.add_argument('--indices', nargs='+', choices=INDICES, default=None,
                        help='Nur bestimmte Indizes bearbeiten')
    args = parser.parse_args()

    indices = args.indices or INDICES

    if not os.path.exists(BEST_CONFIGS):
        print(f"FEHLER: {BEST_CONFIGS} nicht gefunden.")
        print("Zuerst Parameter-Tuning ausführen oder --skip-tuning in run_pipeline.py nutzen.")
        sys.exit(1)

    # Schritt 1: Einzelne L-Werte parallel trainieren
    if not args.collect_only:
        print(f"\n{'='*70}")
        print(f"SCHRITT 1: Lookback-Sweep für {len(indices)} Indizes")
        print(f"{'='*70}")

        tasks = get_missing_tasks()
        # Nur gewählte Indizes
        tasks = [(idx, L) for idx, L in tasks if idx in indices]

        if not tasks:
            print("\nAlle L-Werte bereits vorhanden — überspringe Training.")
        else:
            print(f"\n{len(tasks)} Tasks mit {args.workers} Workern starten...")
            start_total = time.time()
            results = []

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_one, idx, L): (idx, L)
                           for idx, L in tasks}
                for future in as_completed(futures):
                    index, L, status, elapsed = future.result()
                    results.append((index, L, status, elapsed))
                    done = len(results)
                    total_elapsed = time.time() - start_total
                    avg = total_elapsed / done
                    remaining = avg * (len(tasks) - done)
                    print(f'  [{done}/{len(tasks)}] {index}/L={L:2d} {status} '
                          f'({elapsed:.0f}s) | '
                          f'Elapsed: {total_elapsed/60:.0f}min, '
                          f'ETA: {remaining/60:.0f}min',
                          flush=True)

            total_elapsed = time.time() - start_total
            failed = [r for r in results if r[2] == 'FAIL']
            print(f'\nSweep fertig: {len(tasks)-len(failed)}/{len(tasks)} OK '
                  f'in {total_elapsed/60:.1f}min')
            if failed:
                print(f'Fehlgeschlagen: {[(r[0], r[1]) for r in failed]}')
                print('Fehlgeschlagene Tasks können durch erneutes Ausführen nachgeholt werden.')

    # Schritt 2: Ergebnisse sammeln und Plots generieren
    print(f"\n{'='*70}")
    print(f"SCHRITT 2: Plots generieren (inkl. Informer)")
    print(f"{'='*70}")

    for index in indices:
        print(f"\nCollect & Plot: {index}...")
        index, status, elapsed = collect_one(index)
        print(f"  {index}: {status} ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print("FERTIG — Alle Plots wurden generiert.")
    print(f"Ergebnisse in: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
