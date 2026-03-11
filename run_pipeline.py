"""
Maximal parallelisierte Pipeline (Per-Index Tuning)
=====================================================
Fuehrt fuer jeden der 8 Indizes ein eigenes Hyperparameter-Tuning durch,
findet die beste Konfiguration pro Index und nutzt diese fuer den Lookback Sweep.

Schritte:
  1. Hyperparameter-Tuning: 8 Indizes × 208 Configs = 1664 Tasks parallel
  2. Ergebnisse sammeln: Beste Konfiguration pro Index ermitteln
  3. Lookback Sweep: 8 Indizes × 60 L-Werte = 480 Tasks parallel
     (jeder Index nutzt seine eigene beste Konfiguration)
  4. Cross-Index Vergleich: Konfigurationen und Ergebnisse vergleichen

Parallelisierung je nach Hardware (AMD Ryzen AI Max+ 395, 16C/32T, 48GB RAM):
  --workers 8   → 8 parallele Trainings, je ~4 TF-Threads (default)
  --workers 12  → aggressiver, voll ausgelastet
  --workers 16  → maximal

Aufruf:
  python run_pipeline.py                     # Alles parallel (8 Workers)
  python run_pipeline.py --workers 16        # Mehr Parallelität
  python run_pipeline.py --skip-tuning       # Tuning ueberspringen
  python run_pipeline.py --clean-only        # Nur aufraeumen
"""

import os
import sys
import json
import shutil
import subprocess
import time
import argparse
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
TUNING_DIR   = PROJECT_ROOT / 'parameter_tuning'
SWEEP_DIR    = PROJECT_ROOT / 'lookback_window_sweep'
COMPARE_DIR  = PROJECT_ROOT / 'comparison'
TUNING_RESULTS_DIR = TUNING_DIR / 'results'
SWEEP_RESULTS_DIR  = SWEEP_DIR / 'results'
COMPARE_RESULTS_DIR = COMPARE_DIR / 'results'
LOGS_DIR     = PROJECT_ROOT / 'logs'

ALL_INDICES = ['SP500', 'DAX', 'NASDAQ', 'FTSE100', 'HANG_SENG', 'NIKKEI', '10Y_Bond', '30Y_Bond']


# =============================================================================
# CLEANUP
# =============================================================================

def clean_results(full=False):
    """Delete old results.

    By default, preserves tuning results for crash-recovery.
    Use full=True (--clean-only) to delete everything including tuning.
    """
    dirs_to_clean = [
        COMPARE_RESULTS_DIR / 'evaluation',
    ]
    files_to_clean = [
        TUNING_RESULTS_DIR / 'best_configurations.json',
    ]

    if full:
        # Also delete tuning + sweep results (fresh start)
        dirs_to_clean.append(SWEEP_RESULTS_DIR)
        for index_name in ALL_INDICES:
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'LSTM' / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'CNN'  / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'GRU'  / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'INFORMER' / index_name)

    print("=" * 70)
    print("SCHRITT 0: Alte Ergebnisse loeschen")
    print("=" * 70)

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Geloescht: {d}")

    for f in files_to_clean:
        if f.exists():
            f.unlink()
            print(f"  Geloescht: {f}")

    print("Bereinigung abgeschlossen.\n")


# =============================================================================
# SUBPROCESS RUNNER
# =============================================================================

def run_subprocess(cmd, cwd, env, log_file=None):
    """Run a subprocess. Returns (returncode, elapsed_seconds)."""
    start = time.time()
    if log_file:
        log_dir = LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, cwd=str(cwd), env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, cwd=str(cwd), env=env)
    return result.returncode, time.time() - start


def make_env(tf_threads=4):
    """Create env dict with TF thread limits and unbuffered Python."""
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    env['TF_WORKER_THREADS'] = str(tf_threads)
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    return env


# =============================================================================
# STEP 1: PARALLEL HYPERPARAMETER TUNING (8 INDICES × 192 CONFIGS = 1536 TASKS)
# =============================================================================

def dispatch_single_config(args_tuple):
    """Worker function: train one hyperparameter config for one index."""
    index_name, model_type, config_name, tf_threads = args_tuple
    cmd = [sys.executable, '-u', str(TUNING_DIR / 'parameter_tuning.py'),
           '--model', model_type, '--config', config_name, '--index', index_name]
    env = make_env(tf_threads)
    log_name = f'tuning_{index_name}_{model_type}_{config_name}.log'
    returncode, elapsed = run_subprocess(cmd, PROJECT_ROOT, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] {index_name:10s} {model_type.upper():4s} {config_name} ({elapsed:.0f}s)")
    return (index_name, model_type, config_name, returncode, elapsed)


def generate_all_config_names():
    """Generate all config names for LSTM and CNN without importing TensorFlow."""
    import itertools

    lstm_configs = []
    for dropout, dense, lr, batch, epochs in itertools.product(
        [0.2, 0.4], [16, 32], [0.001, 0.005], [8, 16, 32], [50, 100]
    ):
        lstm_configs.append(('lstm', f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"))

    cnn_configs = []
    for kernel, pool, dropout, batch, epochs in itertools.product(
        [3, 5], [2, 4], [0.2, 0.4], [8, 16, 32], [50, 100]
    ):
        cnn_configs.append(('cnn', f"k{kernel}_p{pool}_d{dropout}_b{batch}_e{epochs}"))

    gru_configs = []
    for dropout, dense, lr, batch, epochs in itertools.product(
        [0.2, 0.4], [16, 32], [0.001, 0.005], [8, 16, 32], [50, 100]
    ):
        gru_configs.append(('gru', f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"))

    informer_configs = []
    for d_model, n_heads, dropout, lr, batch, epochs in itertools.product(
        [32, 64], [4, 8], [0.05, 0.1], [0.0001, 0.001], [16, 32], [50, 100]
    ):
        informer_configs.append(('informer', f"dm{d_model}_h{n_heads}_d{dropout}_lr{lr}_b{batch}_e{epochs}"))

    return lstm_configs + cnn_configs + gru_configs + informer_configs


def get_completed_configs():
    """Check which (index, model, config) triples are already done.

    Checks for individual results.json files in per-config directories
    under parameter_tuning/results/{MODEL}/{INDEX}/.
    """
    completed = set()
    for model_type in ['LSTM', 'CNN', 'GRU', 'INFORMER']:
        for index_name in ALL_INDICES:
            model_base = TUNING_RESULTS_DIR / model_type / index_name
            if model_base.exists():
                for config_dir in model_base.iterdir():
                    if config_dir.is_dir() and (config_dir / 'results.json').exists():
                        completed.add((index_name, model_type.lower(), config_dir.name))
    return completed


def run_parallel_tuning(max_workers, tf_threads):
    """Run all hyperparameter configs in parallel (8 indices × 240 configs)."""
    all_configs = generate_all_config_names()
    total_tasks = len(ALL_INDICES) * len(all_configs)

    print("\n" + "#" * 70)
    print(f"# SCHRITT 1: Per-Index Hyperparameter-Tuning")
    print(f"#   {len(ALL_INDICES)} Indizes × {len(all_configs)} Configs = {total_tasks} Tasks")
    print(f"#   {max_workers} parallele Worker")
    print("#" * 70)

    completed = get_completed_configs()

    # Generate all (index, model, config) tasks
    all_tasks = []
    for index_name in ALL_INDICES:
        for model_type, config_name in all_configs:
            if (index_name, model_type, config_name) not in completed:
                all_tasks.append((index_name, model_type, config_name))

    print(f"  Total: {total_tasks}, Bereits fertig: {len(completed)}, Ausstehend: {len(all_tasks)}")

    if not all_tasks:
        print("  Alle Configs bereits abgeschlossen!")
        return True

    start = time.time()
    tasks = [(idx, m, n, tf_threads) for idx, m, n in all_tasks]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dispatch_single_config, t) for t in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start
    failed = [r for r in results if r[3] != 0]
    print(f"\n  Tuning abgeschlossen: {len(results)-len(failed)}/{len(results)} OK "
          f"in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")

    if failed:
        print(f"  Fehlgeschlagen: {len(failed)}")
        for idx, _, name, _, _ in failed[:10]:
            print(f"    - {idx}/{name}")

    return len(failed) == 0


# =============================================================================
# STEP 2: COLLECT RESULTS & FIND BEST CONFIGS PER INDEX
# =============================================================================

def collect_results_and_find_best():
    """Read per-index config results and create best_configurations.json with per-index structure.

    Output format:
    {
        "timestamp": "...",
        "SP500": {
            "best_lstm": {"config_name": "...", "config": {...}, "metrics": {...}},
            "best_cnn":  {"config_name": "...", "config": {...}, "metrics": {...}},
            "total_lstm_configs": 72,
            "total_cnn_configs": 48
        },
        "DAX": { ... },
        ...
    }
    """
    print("\n" + "=" * 70)
    print("SCHRITT 2: Per-Index Ergebnisse sammeln & Beste Configs ermitteln")
    print("=" * 70)

    best_configs = {'timestamp': datetime.now().isoformat()}

    for index_name in ALL_INDICES:
        print(f"\n  --- {index_name} ---")
        index_best = {}

        for model_type, model_dir_name in [('lstm', 'LSTM'), ('cnn', 'CNN'), ('gru', 'GRU'), ('informer', 'INFORMER')]:
            model_base = TUNING_RESULTS_DIR / model_dir_name / index_name
            if not model_base.exists():
                print(f"    WARNUNG: {model_base} nicht gefunden!")
                continue

            # Collect all individual results.json files
            all_results = []
            for config_dir in sorted(model_base.iterdir()):
                results_file = config_dir / 'results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        all_results.append(json.load(f))

            # Save aggregated all_results.json
            agg_file = model_base / 'all_results.json'
            with open(agg_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            valid = [r for r in all_results if 'metrics' in r]
            index_best[f'total_{model_type}_configs'] = len(valid)

            if valid:
                best = min(valid, key=lambda x: x['metrics']['val_loss'])
                index_best[f'best_{model_type}'] = {
                    'config_name': best['config_name'],
                    'config': best['config'],
                    'metrics': best['metrics'],
                }
                rmse = best['metrics'].get('test_rmse', math.sqrt(best['metrics']['test_loss']))
                print(f"    Best {model_dir_name}: {best['config_name']}"
                      f"  (Val Loss: {best['metrics']['val_loss']:.6f},"
                      f" Test MAE: {best['metrics']['test_mae']:.6f},"
                      f" RMSE: {rmse:.6f})")

        best_configs[index_name] = index_best

    # Save best_configurations.json
    TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = TUNING_RESULTS_DIR / 'best_configurations.json'
    with open(best_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    print(f"\n  Gespeichert: {best_path}")

    return best_configs


# =============================================================================
# STEP 3: PARALLEL LOOKBACK SWEEP (480 individual tasks, per-index configs)
# =============================================================================

def dispatch_single_lookback(args_tuple):
    """Worker: train LSTM+CNN for one (index, L) pair with per-index best config."""
    index_name, L, best_configs_path, tf_threads = args_tuple
    cmd = [sys.executable, '-u', str(SWEEP_DIR / 'lookback_window_sweep.py'),
           '--index', index_name, '--lookback', str(L),
           '--best-configs', str(best_configs_path)]
    env = make_env(tf_threads)
    log_name = f'sweep_{index_name}_L{L:02d}.log'
    returncode, elapsed = run_subprocess(cmd, PROJECT_ROOT, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] {index_name} L={L:2d} ({elapsed:.0f}s)")
    return (index_name, L, returncode, elapsed)


def dispatch_sweep_collect(args_tuple):
    """Worker: collect results and generate plots for one index."""
    index_name, best_configs_path, tf_threads = args_tuple
    cmd = [sys.executable, '-u', str(SWEEP_DIR / 'lookback_window_sweep.py'),
           '--index', index_name, '--collect',
           '--best-configs', str(best_configs_path)]
    env = make_env(tf_threads)
    log_name = f'sweep_{index_name}_collect.log'
    returncode, elapsed = run_subprocess(cmd, PROJECT_ROOT, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] Collect {index_name} ({elapsed:.0f}s)")
    return (index_name, returncode, elapsed)


def get_completed_lookbacks():
    """Check which (index, L) pairs are already done (for resume after crash)."""
    completed = set()
    for index_name in ALL_INDICES:
        index_dir = SWEEP_RESULTS_DIR / index_name
        if index_dir.exists():
            for f in index_dir.glob('L_*.json'):
                try:
                    L = int(f.stem.split('_')[1])
                    completed.add((index_name, L))
                except (ValueError, IndexError):
                    pass
    return completed


def run_parallel_sweep(max_workers, tf_threads):
    """Run lookback sweep: 480 individual (index, L) tasks in parallel, then collect."""
    total_tasks = len(ALL_INDICES) * 60  # 8 indices × 60 L-values
    best_configs_path = TUNING_RESULTS_DIR / 'best_configurations.json'

    if not best_configs_path.exists():
        print("FEHLER: best_configurations.json nicht gefunden! Tuning zuerst ausfuehren.")
        return False

    print("\n" + "#" * 70)
    print(f"# SCHRITT 3a: Lookback Sweep ({total_tasks} Tasks, {max_workers} parallel)")
    print(f"#   Jeder Index nutzt seine eigene beste Konfiguration")
    print("#" * 70)

    # Generate all (index, L) pairs — interleave indices for even distribution
    all_tasks = [(idx, L) for L in range(1, 61) for idx in ALL_INDICES]
    completed = get_completed_lookbacks()
    pending = [(idx, L) for idx, L in all_tasks if (idx, L) not in completed]

    print(f"  Total: {len(all_tasks)}, Bereits fertig: {len(completed)}, Ausstehend: {len(pending)}")

    if pending:
        start = time.time()
        tasks = [(idx, L, best_configs_path, tf_threads) for idx, L in pending]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(dispatch_single_lookback, t) for t in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start
        failed = [r for r in results if r[2] != 0]
        print(f"\n  Sweep abgeschlossen: {len(results)-len(failed)}/{len(results)} OK "
              f"in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")

        if failed:
            for name, L, _, _ in failed[:10]:
                print(f"    FEHLER: {name} L={L} (siehe logs/sweep_{name}_L{L:02d}.log)")
    else:
        print("  Alle Tasks bereits abgeschlossen!")

    # Step 3b: Collect results and generate plots per index
    print("\n" + "#" * 70)
    print(f"# SCHRITT 3b: Ergebnisse sammeln & Plots (8 Indizes, {max_workers} parallel)")
    print("#" * 70)

    start = time.time()
    collect_tasks = [(idx, best_configs_path, tf_threads) for idx in ALL_INDICES]

    collect_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dispatch_sweep_collect, t) for t in collect_tasks]
        for future in as_completed(futures):
            collect_results.append(future.result())

    elapsed = time.time() - start
    failed = [r for r in collect_results if r[1] != 0]
    print(f"\n  Collect abgeschlossen: {len(collect_results)-len(failed)}/{len(collect_results)} OK "
          f"in {elapsed/60:.1f} min")

    if failed:
        for name, _, _ in failed:
            print(f"    FEHLER: {name} (siehe logs/sweep_{name}_collect.log)")

    return len(failed) == 0


# =============================================================================
# STEP 4: CROSS-INDEX COMPARISON
# =============================================================================

def run_cross_index_comparison():
    """Compare best hyperparameter configurations and results across all indices."""
    print("\n" + "#" * 70)
    print("# SCHRITT 4: Cross-Index Vergleich")
    print("#" * 70)

    best_path = TUNING_RESULTS_DIR / 'best_configurations.json'
    if not best_path.exists():
        print("  FEHLER: best_configurations.json nicht gefunden!")
        return

    with open(best_path, 'r') as f:
        best_configs = json.load(f)

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'lstm_configs': {},
        'cnn_configs': {},
        'gru_configs': {},
        'informer_configs': {},
        'lstm_sweep_results': {},
        'cnn_sweep_results': {},
        'gru_sweep_results': {},
        'informer_sweep_results': {},
    }

    # --- Hyperparameter Comparison ---
    print("\n" + "=" * 100)
    print("BESTE LSTM-KONFIGURATIONEN PRO INDEX")
    print("=" * 100)
    print(f"{'Index':12s} {'Config':35s} {'Val Loss':>10s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 100)

    for index_name in ALL_INDICES:
        if index_name not in best_configs:
            continue
        idx_best = best_configs[index_name]
        if 'best_lstm' in idx_best:
            b = idx_best['best_lstm']
            rmse = b['metrics'].get('test_rmse', math.sqrt(b['metrics']['test_loss']))
            print(f"{index_name:12s} {b['config_name']:35s} "
                  f"{b['metrics']['val_loss']:10.6f} {b['metrics']['test_mae']:10.6f} {rmse:10.6f}")
            comparison['lstm_configs'][index_name] = {
                'config_name': b['config_name'],
                'config': b['config'],
                'val_loss': b['metrics']['val_loss'],
                'test_mae': b['metrics']['test_mae'],
                'test_rmse': rmse,
            }

    print("\n" + "=" * 100)
    print("BESTE CNN-KONFIGURATIONEN PRO INDEX")
    print("=" * 100)
    print(f"{'Index':12s} {'Config':35s} {'Val Loss':>10s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 100)

    for index_name in ALL_INDICES:
        if index_name not in best_configs:
            continue
        idx_best = best_configs[index_name]
        if 'best_cnn' in idx_best:
            b = idx_best['best_cnn']
            rmse = b['metrics'].get('test_rmse', math.sqrt(b['metrics']['test_loss']))
            print(f"{index_name:12s} {b['config_name']:35s} "
                  f"{b['metrics']['val_loss']:10.6f} {b['metrics']['test_mae']:10.6f} {rmse:10.6f}")
            comparison['cnn_configs'][index_name] = {
                'config_name': b['config_name'],
                'config': b['config'],
                'val_loss': b['metrics']['val_loss'],
                'test_mae': b['metrics']['test_mae'],
                'test_rmse': rmse,
            }

    print("\n" + "=" * 100)
    print("BESTE GRU-KONFIGURATIONEN PRO INDEX")
    print("=" * 100)
    print(f"{'Index':12s} {'Config':35s} {'Val Loss':>10s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 100)

    for index_name in ALL_INDICES:
        if index_name not in best_configs:
            continue
        idx_best = best_configs[index_name]
        if 'best_gru' in idx_best:
            b = idx_best['best_gru']
            rmse = b['metrics'].get('test_rmse', math.sqrt(b['metrics']['test_loss']))
            print(f"{index_name:12s} {b['config_name']:35s} "
                  f"{b['metrics']['val_loss']:10.6f} {b['metrics']['test_mae']:10.6f} {rmse:10.6f}")
            comparison['gru_configs'][index_name] = {
                'config_name': b['config_name'],
                'config': b['config'],
                'val_loss': b['metrics']['val_loss'],
                'test_mae': b['metrics']['test_mae'],
                'test_rmse': rmse,
            }

    print("\n" + "=" * 100)
    print("BESTE INFORMER-KONFIGURATIONEN PRO INDEX")
    print("=" * 100)
    print(f"{'Index':12s} {'Config':35s} {'Val Loss':>10s} {'Test MAE':>10s} {'Test RMSE':>10s}")
    print("-" * 100)

    for index_name in ALL_INDICES:
        if index_name not in best_configs:
            continue
        idx_best = best_configs[index_name]
        if 'best_informer' in idx_best:
            b = idx_best['best_informer']
            rmse = b['metrics'].get('test_rmse', math.sqrt(b['metrics']['test_loss']))
            print(f"{index_name:12s} {b['config_name']:35s} "
                  f"{b['metrics']['val_loss']:10.6f} {b['metrics']['test_mae']:10.6f} {rmse:10.6f}")
            comparison['informer_configs'][index_name] = {
                'config_name': b['config_name'],
                'config': b['config'],
                'val_loss': b['metrics']['val_loss'],
                'test_mae': b['metrics']['test_mae'],
                'test_rmse': rmse,
            }

    # --- Analyze Config Similarity ---
    print("\n" + "=" * 90)
    print("HYPERPARAMETER-ANALYSE: Welche Parameter werden bevorzugt?")
    print("=" * 90)

    # LSTM parameter frequency
    lstm_params = {'dropout': {}, 'dense_units': {}, 'lr': {}, 'batch': {}, 'epochs': {}}
    for idx, data in comparison['lstm_configs'].items():
        cfg = data['config']
        for param in lstm_params:
            val = cfg[param]
            lstm_params[param][val] = lstm_params[param].get(val, [])
            lstm_params[param][val].append(idx)

    print("\nLSTM - Haeufigkeit der besten Parameter:")
    for param, values in lstm_params.items():
        print(f"  {param}:")
        for val, indices in sorted(values.items(), key=lambda x: -len(x[1])):
            print(f"    {val}: {len(indices)}x ({', '.join(indices)})")

    # CNN parameter frequency
    cnn_params = {'kernel': {}, 'pool': {}, 'dropout': {}, 'batch': {}, 'epochs': {}}
    for idx, data in comparison['cnn_configs'].items():
        cfg = data['config']
        for param in cnn_params:
            val = cfg[param]
            cnn_params[param][val] = cnn_params[param].get(val, [])
            cnn_params[param][val].append(idx)

    print("\nCNN - Haeufigkeit der besten Parameter:")
    for param, values in cnn_params.items():
        print(f"  {param}:")
        for val, indices in sorted(values.items(), key=lambda x: -len(x[1])):
            print(f"    {val}: {len(indices)}x ({', '.join(indices)})")

    # GRU parameter frequency
    gru_params = {'dropout': {}, 'dense_units': {}, 'lr': {}, 'batch': {}, 'epochs': {}}
    for idx, data in comparison['gru_configs'].items():
        cfg = data['config']
        for param in gru_params:
            val = cfg[param]
            gru_params[param][val] = gru_params[param].get(val, [])
            gru_params[param][val].append(idx)

    print("\nGRU - Haeufigkeit der besten Parameter:")
    for param, values in gru_params.items():
        print(f"  {param}:")
        for val, indices in sorted(values.items(), key=lambda x: -len(x[1])):
            print(f"    {val}: {len(indices)}x ({', '.join(indices)})")

    # Informer parameter frequency
    informer_params = {'d_model': {}, 'n_heads': {}, 'dropout': {}, 'lr': {}, 'batch': {}, 'epochs': {}}
    for idx, data in comparison['informer_configs'].items():
        cfg = data['config']
        for param in informer_params:
            val = cfg.get(param)
            if val is not None:
                informer_params[param][val] = informer_params[param].get(val, [])
                informer_params[param][val].append(idx)

    print("\nInformer - Haeufigkeit der besten Parameter:")
    for param, values in informer_params.items():
        print(f"  {param}:")
        for val, indices in sorted(values.items(), key=lambda x: -len(x[1])):
            print(f"    {val}: {len(indices)}x ({', '.join(indices)})")

    # --- Sweep Results Comparison ---
    print("\n" + "=" * 90)
    print("LOOKBACK SWEEP: Bestes L pro Index")
    print("=" * 90)

    sweep_base = SWEEP_RESULTS_DIR
    print(f"\n{'Index':12s} {'LSTM L':>7s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'CNN L':>7s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'GRU L':>7s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'Inf L':>7s} {'MAE':>10s} {'RMSE':>10s}")
    print("-" * 130)

    for index_name in ALL_INDICES:
        results_file = sweep_base / index_name / 'lookback_evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                sweep_res = json.load(f)

            lstm_l = sweep_res.get('best_l_lstm', {}).get('l_value', '?')
            lstm_mae = sweep_res.get('best_l_lstm', {}).get('mae', float('nan'))
            lstm_rmse = sweep_res.get('best_l_lstm', {}).get('rmse', math.sqrt(lstm_mae) if not math.isnan(lstm_mae) else float('nan'))
            cnn_l = sweep_res.get('best_l_cnn', {}).get('l_value', '?')
            cnn_mae = sweep_res.get('best_l_cnn', {}).get('mae', float('nan'))
            cnn_rmse = sweep_res.get('best_l_cnn', {}).get('rmse', math.sqrt(cnn_mae) if not math.isnan(cnn_mae) else float('nan'))
            gru_l = sweep_res.get('best_l_gru', {}).get('l_value', '?')
            gru_mae = sweep_res.get('best_l_gru', {}).get('mae', float('nan'))
            gru_rmse = sweep_res.get('best_l_gru', {}).get('rmse', math.sqrt(gru_mae) if not math.isnan(gru_mae) else float('nan'))
            inf_l = sweep_res.get('best_l_informer', {}).get('l_value', '?')
            inf_mae = sweep_res.get('best_l_informer', {}).get('mae', float('nan'))
            inf_rmse = sweep_res.get('best_l_informer', {}).get('rmse', float('nan'))

            print(f"{index_name:12s} {str(lstm_l):>7s} {lstm_mae:10.6f} {lstm_rmse:10.6f} "
                  f"{str(cnn_l):>7s} {cnn_mae:10.6f} {cnn_rmse:10.6f} "
                  f"{str(gru_l):>7s} {gru_mae:10.6f} {gru_rmse:10.6f} "
                  f"{str(inf_l):>7s} {inf_mae:10.6f} {inf_rmse:10.6f}")

            comparison['lstm_sweep_results'][index_name] = {'best_L': lstm_l, 'mae': lstm_mae, 'rmse': lstm_rmse}
            comparison['cnn_sweep_results'][index_name] = {'best_L': cnn_l, 'mae': cnn_mae, 'rmse': cnn_rmse}
            comparison['gru_sweep_results'][index_name] = {'best_L': gru_l, 'mae': gru_mae, 'rmse': gru_rmse}
            comparison['informer_sweep_results'][index_name] = {'best_L': inf_l, 'mae': inf_mae, 'rmse': inf_rmse}
        else:
            print(f"{index_name:12s} {'?':>7s} {'':>10s} {'':>10s} {'?':>7s} {'':>10s} {'':>10s} {'?':>7s} {'':>10s} {'':>10s} {'?':>7s}")

    # --- Config Deviation Analysis ---
    print("\n" + "=" * 90)
    print("KONFIGURATIONSABWEICHUNGEN")
    print("=" * 90)

    # Check how many unique configs exist
    unique_lstm = set(d['config_name'] for d in comparison['lstm_configs'].values())
    unique_cnn = set(d['config_name'] for d in comparison['cnn_configs'].values())
    print(f"\n  LSTM: {len(unique_lstm)} verschiedene Konfigurationen aus {len(comparison['lstm_configs'])} Indizes")
    for cfg_name in sorted(unique_lstm):
        indices_with = [idx for idx, d in comparison['lstm_configs'].items() if d['config_name'] == cfg_name]
        print(f"    {cfg_name}: {', '.join(indices_with)}")

    print(f"\n  CNN: {len(unique_cnn)} verschiedene Konfigurationen aus {len(comparison['cnn_configs'])} Indizes")
    for cfg_name in sorted(unique_cnn):
        indices_with = [idx for idx, d in comparison['cnn_configs'].items() if d['config_name'] == cfg_name]
        print(f"    {cfg_name}: {', '.join(indices_with)}")

    unique_gru = set(d['config_name'] for d in comparison['gru_configs'].values())
    print(f"\n  GRU: {len(unique_gru)} verschiedene Konfigurationen aus {len(comparison['gru_configs'])} Indizes")
    for cfg_name in sorted(unique_gru):
        indices_with = [idx for idx, d in comparison['gru_configs'].items() if d['config_name'] == cfg_name]
        print(f"    {cfg_name}: {', '.join(indices_with)}")

    unique_informer = set(d['config_name'] for d in comparison['informer_configs'].values())
    print(f"\n  Informer: {len(unique_informer)} verschiedene Konfigurationen aus {len(comparison['informer_configs'])} Indizes")
    for cfg_name in sorted(unique_informer):
        indices_with = [idx for idx, d in comparison['informer_configs'].items() if d['config_name'] == cfg_name]
        print(f"    {cfg_name}: {', '.join(indices_with)}")

    # Save comparison
    COMPARE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARE_RESULTS_DIR / 'cross_index_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Vergleich gespeichert: {comparison_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Masterarbeit Pipeline (Per-Index Tuning)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallele Prozesse (default: 8)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Hyperparameter-Tuning ueberspringen')
    parser.add_argument('--clean-only', action='store_true',
                        help='Nur aufraeumen')
    args = parser.parse_args()

    # Calculate TF threads per worker: total_threads / workers
    total_threads = os.cpu_count() or 16
    tf_threads = max(2, total_threads // args.workers)

    total_start = time.time()

    print("\n" + "=" * 70)
    print("MASTERARBEIT PIPELINE (PER-INDEX TUNING)")
    print("=" * 70)
    print(f"Gestartet:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Kerne:     {total_threads}")
    print(f"Workers:       {args.workers}")
    print(f"TF Threads/W:  {tf_threads}")
    print(f"Indizes:       {len(ALL_INDICES)}")
    print(f"Tuning Tasks:  {len(ALL_INDICES)} × 208 = {len(ALL_INDICES) * 208}")
    print(f"Sweep Tasks:   {len(ALL_INDICES)} × 60  = {len(ALL_INDICES) * 60}")
    print()

    # Step 0: Clean (preserve tuning results for crash-recovery)
    clean_results(full=args.clean_only)

    if args.clean_only:
        print("--clean-only: Fertig.")
        return

    # Step 1: Parallel Hyperparameter Tuning (per index)
    if not args.skip_tuning:
        run_parallel_tuning(args.workers, tf_threads)

        # Step 2: Collect results & find best per index
        collect_results_and_find_best()
    else:
        print("\n--skip-tuning: Schritte 1+2 uebersprungen.\n")

    # Step 3: Parallel Lookback Sweep (per-index best configs)
    run_parallel_sweep(args.workers, tf_threads)

    # Step 4: Cross-Index Comparison
    run_cross_index_comparison()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE KOMPLETT ABGESCHLOSSEN")
    print("=" * 70)
    print(f"Gesamtdauer: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
    print(f"Tuning:      {TUNING_RESULTS_DIR}")
    print(f"Sweep:       {SWEEP_RESULTS_DIR}")
    print(f"Vergleich:   {COMPARE_RESULTS_DIR}")
    print(f"Logs:        {LOGS_DIR}")


if __name__ == "__main__":
    main()
