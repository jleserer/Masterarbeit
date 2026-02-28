"""
Maximal parallelisierte Pipeline
=================================
Nutzt alle verfügbaren CPU-Kerne für maximale Geschwindigkeit.

Parallelisierung:
  Schritt 1: Alle 120 Hyperparameter-Configs parallel (bis zu --workers Prozesse)
             Jeder Prozess: 1 Config laden → trainieren → speichern
  Schritt 3a: Alle 480 (Index, L)-Paare parallel (8 Indizes × 60 L-Werte)
              Jeder Prozess: 1 (Index, L) laden → LSTM+CNN trainieren → speichern
  Schritt 3b: Pro Index Ergebnisse sammeln & Plots generieren (parallel)

Hardware-Empfehlung (AMD Ryzen AI Max+ 395, 16C/32T, 48GB RAM):
  --workers 8   → 8 parallele Trainings, je ~4 TF-Threads (default)
  --workers 12  → aggressiver, voll ausgelastet
  --workers 4   → konservativ, gut für Nebenbei-Arbeit

Aufruf:
  python run_pipeline.py                     # Alles parallel (8 Workers)
  python run_pipeline.py --workers 12        # Mehr Parallelität
  python run_pipeline.py --skip-tuning       # Tuning überspringen
  python run_pipeline.py --clean-only        # Nur aufräumen
"""

import os
import sys
import json
import re
import shutil
import subprocess
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / 'models'
RESULTS_DIR  = PROJECT_ROOT / 'results'
TUNING_RESULTS_DIR = RESULTS_DIR / 'tuning'

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
        RESULTS_DIR / 'evaluation',
        RESULTS_DIR / 'lookback_evaluation',
    ]
    files_to_clean = [
        RESULTS_DIR / 'best_configurations.json',
        TUNING_RESULTS_DIR / 'best_configurations.json',
    ]

    if full:
        # Also delete tuning + sweep results (fresh start)
        dirs_to_clean += [
            RESULTS_DIR / 'lookback_sweep',
            TUNING_RESULTS_DIR / 'LSTM' / 'SP500',
            TUNING_RESULTS_DIR / 'CNN'  / 'SP500',
        ]

    print("=" * 70)
    print("SCHRITT 0: Alte Ergebnisse löschen")
    print("=" * 70)

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Gelöscht: {d}")

    for f in files_to_clean:
        if f.exists():
            f.unlink()
            print(f"  Gelöscht: {f}")

    print("Bereinigung abgeschlossen.\n")


# =============================================================================
# SUBPROCESS RUNNER
# =============================================================================

def run_subprocess(cmd, cwd, env, log_file=None):
    """Run a subprocess. Returns (returncode, elapsed_seconds)."""
    start = time.time()
    if log_file:
        log_dir = RESULTS_DIR / 'logs'
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
# STEP 1: PARALLEL HYPERPARAMETER TUNING (ALL 120 CONFIGS)
# =============================================================================

def dispatch_single_config(args_tuple):
    """Worker function: train one hyperparameter config."""
    model_type, config_name, tf_threads = args_tuple
    cmd = [sys.executable, '-u', 'hyperparameter_tuning.py',
           '--model', model_type, '--config', config_name]
    env = make_env(tf_threads)
    log_name = f'tuning_{model_type}_{config_name}.log'
    returncode, elapsed = run_subprocess(cmd, MODELS_DIR, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] {model_type.upper():4s} {config_name} ({elapsed:.0f}s)")
    return (model_type, config_name, returncode, elapsed)


def generate_all_config_names():
    """Generate all config names for LSTM and CNN without importing TensorFlow."""
    import itertools

    lstm_configs = []
    for dropout, dense, lr, batch, epochs in itertools.product(
        [0.2, 0.4], [16, 32, 64], [0.001, 0.005], [8, 16, 32], [50, 100]
    ):
        lstm_configs.append(('lstm', f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"))

    cnn_configs = []
    for kernel, pool, dropout, batch, epochs in itertools.product(
        [3, 5], [2, 4], [0.2, 0.4], [8, 16, 32], [50, 100]
    ):
        cnn_configs.append(('cnn', f"k{kernel}_p{pool}_d{dropout}_b{batch}_e{epochs}"))

    return lstm_configs + cnn_configs


def get_completed_configs():
    """Check which configs are already done (for resume after crash).

    Checks for individual results.json files in per-config directories
    under models/results/ (where hyperparameter_tuning.py saves them).
    """
    completed = set()
    for model_type in ['LSTM', 'CNN']:
        model_base = TUNING_RESULTS_DIR / model_type / 'SP500'
        if model_base.exists():
            for config_dir in model_base.iterdir():
                if config_dir.is_dir() and (config_dir / 'results.json').exists():
                    completed.add((model_type.lower(), config_dir.name))
    return completed


def run_parallel_tuning(max_workers, tf_threads):
    """Run all 120 hyperparameter configs in parallel."""
    print("\n" + "#" * 70)
    print(f"# SCHRITT 1: Hyperparameter-Tuning (120 Configs, {max_workers} parallel)")
    print("#" * 70)

    all_configs = generate_all_config_names()
    completed = get_completed_configs()
    pending = [(m, n) for m, n in all_configs if (m, n) not in completed]

    print(f"  Total: {len(all_configs)}, Bereits fertig: {len(completed)}, Ausstehend: {len(pending)}")

    if not pending:
        print("  Alle Configs bereits abgeschlossen!")
        return True

    start = time.time()
    tasks = [(m, n, tf_threads) for m, n in pending]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dispatch_single_config, t) for t in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start
    failed = [r for r in results if r[2] != 0]
    print(f"\n  Tuning abgeschlossen: {len(results)-len(failed)}/{len(results)} OK "
          f"in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")

    if failed:
        print(f"  Fehlgeschlagen: {len(failed)}")
        for _, name, _, _ in failed[:5]:
            print(f"    - {name}")

    return len(failed) == 0


# =============================================================================
# STEP 2: COLLECT RESULTS & FIND BEST CONFIGS
# =============================================================================

def collect_results_and_find_best():
    """Read individual config results and create aggregated JSON + best_configurations.json."""
    print("\n" + "=" * 70)
    print("SCHRITT 2: Ergebnisse sammeln & Beste Configs ermitteln")
    print("=" * 70)

    best_configs = {'timestamp': datetime.now().isoformat()}

    for model_type, model_dir_name in [('lstm', 'LSTM'), ('cnn', 'CNN')]:
        model_base = TUNING_RESULTS_DIR / model_dir_name / 'SP500'
        if not model_base.exists():
            print(f"  WARNUNG: {model_base} nicht gefunden!")
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
        print(f"  {model_dir_name}: {len(valid)} gültige Configs gesammelt")

        if valid:
            best = min(valid, key=lambda x: x['metrics']['val_loss'])
            key = f'best_{model_type}'
            best_configs[key] = {
                'config_name': best['config_name'],
                'config': best['config'],
                'metrics': best['metrics'],
            }
            best_configs[f'total_{model_type}_configs'] = len(valid)
            print(f"  Best {model_dir_name}: {best['config_name']}")
            print(f"    Val Loss: {best['metrics']['val_loss']:.6f}, "
                  f"Test MAE: {best['metrics']['test_mae']:.6f}")

    # Save best_configurations.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = RESULTS_DIR / 'best_configurations.json'
    with open(best_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    print(f"  Gespeichert: {best_path}")

    # Update sweep script
    update_sweep_configs(best_configs)

    return best_configs


def update_sweep_configs(best_configs):
    """Update BEST_LSTM_CONFIG / BEST_CNN_CONFIG in evaluate_lookback_window_sweep.py."""
    sweep_path = MODELS_DIR / 'evaluate_lookback_window_sweep.py'
    code = sweep_path.read_text(encoding='utf-8')

    if 'best_lstm' in best_configs:
        cfg = best_configs['best_lstm']['config']
        new_block = (
            f"BEST_LSTM_CONFIG = {{\n"
            f"    'name': '{best_configs['best_lstm']['config_name']}',\n"
            f"    'dropout': {cfg['dropout']},\n"
            f"    'dense_units': {cfg['dense_units']},\n"
            f"    'lr': {cfg['lr']},\n"
            f"    'batch': {cfg['batch']},\n"
            f"    'epochs': {cfg['epochs']}\n"
            f"}}"
        )
        code = re.sub(r"BEST_LSTM_CONFIG\s*=\s*\{[^}]*\}", new_block, code, flags=re.DOTALL)
        print(f"  Sweep-Script: BEST_LSTM_CONFIG -> {best_configs['best_lstm']['config_name']}")

    if 'best_cnn' in best_configs:
        cfg = best_configs['best_cnn']['config']
        new_block = (
            f"BEST_CNN_CONFIG = {{\n"
            f"    'name': '{best_configs['best_cnn']['config_name']}',\n"
            f"    'kernel': {cfg['kernel']},\n"
            f"    'pool': {cfg['pool']},\n"
            f"    'dropout': {cfg['dropout']},\n"
            f"    'batch': {cfg['batch']},\n"
            f"    'epochs': {cfg['epochs']}\n"
            f"}}"
        )
        code = re.sub(r"BEST_CNN_CONFIG\s*=\s*\{[^}]*\}", new_block, code, flags=re.DOTALL)
        print(f"  Sweep-Script: BEST_CNN_CONFIG  -> {best_configs['best_cnn']['config_name']}")

    sweep_path.write_text(code, encoding='utf-8')


# =============================================================================
# STEP 3: PARALLEL LOOKBACK SWEEP (480 individual tasks)
# =============================================================================

def dispatch_single_lookback(args_tuple):
    """Worker: train LSTM+CNN for one (index, L) pair."""
    index_name, L, tf_threads = args_tuple
    cmd = [sys.executable, '-u', 'evaluate_lookback_window_sweep.py',
           '--index', index_name, '--lookback', str(L)]
    env = make_env(tf_threads)
    log_name = f'sweep_{index_name}_L{L:02d}.log'
    returncode, elapsed = run_subprocess(cmd, MODELS_DIR, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] {index_name} L={L:2d} ({elapsed:.0f}s)")
    return (index_name, L, returncode, elapsed)


def dispatch_sweep_collect(args_tuple):
    """Worker: collect results and generate plots for one index."""
    index_name, tf_threads = args_tuple
    cmd = [sys.executable, '-u', 'evaluate_lookback_window_sweep.py',
           '--index', index_name, '--collect']
    env = make_env(tf_threads)
    log_name = f'sweep_{index_name}_collect.log'
    returncode, elapsed = run_subprocess(cmd, MODELS_DIR, env, log_name)
    status = 'OK' if returncode == 0 else 'FAIL'
    print(f"  [{status}] Collect {index_name} ({elapsed:.0f}s)")
    return (index_name, returncode, elapsed)


def get_completed_lookbacks():
    """Check which (index, L) pairs are already done (for resume after crash)."""
    completed = set()
    sweep_base = RESULTS_DIR / 'lookback_sweep'
    for index_name in ALL_INDICES:
        index_dir = sweep_base / index_name
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

    print("\n" + "#" * 70)
    print(f"# SCHRITT 3a: Lookback Sweep ({total_tasks} Tasks, {max_workers} parallel)")
    print("#" * 70)

    # Generate all (index, L) pairs — interleave indices for even distribution
    all_tasks = [(idx, L) for L in range(1, 61) for idx in ALL_INDICES]
    completed = get_completed_lookbacks()
    pending = [(idx, L) for idx, L in all_tasks if (idx, L) not in completed]

    print(f"  Total: {len(all_tasks)}, Bereits fertig: {len(completed)}, Ausstehend: {len(pending)}")

    if pending:
        start = time.time()
        tasks = [(idx, L, tf_threads) for idx, L in pending]

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
                print(f"    FEHLER: {name} L={L} (siehe results/logs/sweep_{name}_L{L:02d}.log)")
    else:
        print("  Alle Tasks bereits abgeschlossen!")

    # Step 3b: Collect results and generate plots per index
    print("\n" + "#" * 70)
    print(f"# SCHRITT 3b: Ergebnisse sammeln & Plots (8 Indizes, {max_workers} parallel)")
    print("#" * 70)

    start = time.time()
    collect_tasks = [(idx, tf_threads) for idx in ALL_INDICES]

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
            print(f"    FEHLER: {name} (siehe results/logs/sweep_{name}_collect.log)")

    return len(failed) == 0


# =============================================================================
# STEP 4: FINAL EVALUATION
# =============================================================================

def run_final_evaluation():
    """Run evaluate_best_models.py."""
    print("\n" + "#" * 70)
    print("# SCHRITT 4: Finaler Modellvergleich")
    print("#" * 70)

    cmd = [sys.executable, '-u', 'evaluate_best_models.py']
    env = make_env(tf_threads=8)
    returncode, elapsed = run_subprocess(cmd, MODELS_DIR, env, 'evaluate_best_models.log')

    status = 'OK' if returncode == 0 else 'FEHLER'
    print(f"  [{status}] evaluate_best_models.py ({elapsed/60:.1f} min)")
    return returncode == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Masterarbeit Pipeline (max. parallel)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallele Prozesse (default: 8)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Hyperparameter-Tuning überspringen')
    parser.add_argument('--clean-only', action='store_true',
                        help='Nur aufräumen')
    args = parser.parse_args()

    # Calculate TF threads per worker: total_threads / workers
    total_threads = os.cpu_count() or 16
    tf_threads = max(2, total_threads // args.workers)

    total_start = time.time()

    print("\n" + "=" * 70)
    print("MASTERARBEIT PIPELINE (MAXIMAL PARALLEL)")
    print("=" * 70)
    print(f"Gestartet:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Kerne:     {total_threads}")
    print(f"Workers:       {args.workers}")
    print(f"TF Threads/W:  {tf_threads}")
    print()

    # Step 0: Clean (preserve tuning results for crash-recovery)
    clean_results(full=args.clean_only)

    if args.clean_only:
        print("--clean-only: Fertig.")
        return

    # Step 1: Parallel Hyperparameter Tuning
    if not args.skip_tuning:
        run_parallel_tuning(args.workers, tf_threads)

        # Step 2: Collect results & find best
        collect_results_and_find_best()
    else:
        print("\n--skip-tuning: Schritte 1+2 übersprungen.\n")

    # Step 3: Parallel Lookback Sweep (480 individual tasks, not limited by index count)
    run_parallel_sweep(args.workers, tf_threads)

    # Step 4: Final Evaluation
    run_final_evaluation()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE KOMPLETT ABGESCHLOSSEN")
    print("=" * 70)
    print(f"Gesamtdauer: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
    print(f"Ergebnisse:  {RESULTS_DIR}")
    print(f"Logs:        {RESULTS_DIR / 'logs'}")


if __name__ == "__main__":
    main()
