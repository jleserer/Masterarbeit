"""
Maximal parallelisierte Pipeline (Per-Index Tuning + L-Sweep)
=============================================================
Führt für jeden der 6 Indizes ein eigenes Hyperparameter-Tuning durch,
wählt die global beste Config pro (Index, Modell) und predicted dann
auf Test über L=1..60.

Schritte:
  1. Coarse Tuning: 6 × 160 Configs × 7 Coarse-L = 6 720 Tasks (Train+Val)
     Configs = LSTM 32 + CNN 32 + GRU 32 + Informer 64 = 160.
  2. Global Best Auswahl: pro (Index, Modell) lowest val_loss -> global_best.json
  3. L=1..60 Test-Predict-Sweep: 6 × 4 × 60 = 1 440 Tasks
     - L in Coarse-Liste -> LOAD model.keras/.h5/.pt aus Phase 1 (kein Re-Train)
     - L sonst           -> Re-Train mit best_cfg bei sequence_length=L
  4. Aggregation: best_configurations.json (Val-Kurve coarse + Test-Kurve fine)
  5. Plots pro Index aus den gespeicherten Daten
  6. Cross-Index Vergleich

Parallelisierung (Hardware-Cap WORKER_HARD_CAP = 16):
  AMD Ryzen AI Max+ 395, 16C/32T, 48 GB RAM. Werte > 16 werden geclamped,
  weil 32 Worker × ~1.2 GB TF-Init RAM-Footprint hat und auf 48 GB OOM
  ausgelöst hat.

Aufruf:
  python run_pipeline.py --workers 16        # Empfehlung
  python run_pipeline.py --workers 8         # Konservativ
  python run_pipeline.py --no-clean          # Resume nach Crash
  python run_pipeline.py --clean-only        # Nur aufräumen
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
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
TUNING_DIR   = PROJECT_ROOT / 'parameter_tuning'
COMPARE_DIR  = PROJECT_ROOT / 'comparison'
TUNING_RESULTS_DIR = TUNING_DIR / 'results'
COMPARE_RESULTS_DIR = COMPARE_DIR / 'results'
LOGS_DIR     = PROJECT_ROOT / 'logs'

sys.path.insert(0, str(PROJECT_ROOT))
from config import ALL_INDICES  # noqa: E402


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
        # Also delete tuning + results (fresh start)
        for index_name in ALL_INDICES:
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'LSTM' / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'CNN'  / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'GRU'  / index_name)
            dirs_to_clean.append(TUNING_RESULTS_DIR / 'INFORMER' / index_name)

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
    """Run a subprocess. Returns (returncode, elapsed_seconds).

    Auf Windows kann eine große Zahl gleichzeitiger subprocess.Popen-Aufrufe
    zu OSError [WinError 87] (Falscher Parameter beim OpenProcess) und/oder
    Access-Violations führen — Process-Handle-Verwaltung gerät unter Druck.
    Mit einem kleinen Jitter (0–500ms) entzerren wir den Spawn-Druck.
    """
    import random
    time.sleep(random.uniform(0, 0.5))

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

_WORKER_INITIALIZED = False


def _pool_initializer(tf_threads):
    """Wird einmal pro Pool-Worker beim Start aufgerufen.

    Importiert TF/Torch genau EINMAL pro Worker-Prozess und konfiguriert Threads,
    bevor irgendein Task läuft. Damit haben wir maximal `max_workers` parallele
    TF-Init-Operationen, die alle gleichzeitig zu Start stattfinden — danach werden
    Tasks im selben Prozess seriell abgearbeitet ohne erneutes TF-Laden.
    """
    global _WORKER_INITIALIZED
    if _WORKER_INITIALIZED:
        return
    import os as _os
    _os.environ['TF_WORKER_THREADS'] = str(tf_threads)
    _os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _os.environ['PYTHONHASHSEED'] = '42'
    import random as _r
    _r.seed(42)
    import numpy as _np
    _np.random.seed(42)
    try:
        import tensorflow as _tf
        _tf.random.set_seed(42)
        _tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        _tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass
    # torch wird NICHT im Pool-Initializer importiert — Informer-Tasks laden es
    # lazy beim ersten Aufruf. So zahlt nur ein Informer-Worker den Import-Cost,
    # nicht alle 16 parallel beim Pool-Spawn.
    _WORKER_INITIALIZED = True


def _release_worker_resources():
    """Nach jedem Task: TF-Graph + Python-GC leeren, damit ein wiederverwendeter
    Pool-Worker nicht über Hunderte Tasks hinweg Modell-Graphen akkumuliert.

    Hauptursache für den OOM-Drift: Keras-Backend-Default speichert pro Worker
    einen globalen Default-Graph; ohne clear_session wächst er monoton.
    """
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass


def dispatch_single_config(args_tuple):
    """In-process Worker: trainiert eine Config direkt im aktuellen Prozess.

    Vermeidet subprocess.Popen pro Task — keine N×TF-Imports = kein Windows-DLL-
    Deadlock. Logging via parameter_tuning.run_single_config in dessen Streams.

    Task-Tupel: (index_name, model_type, config_name, L, tf_threads)
    """
    index_name, model_type, config_name, L, tf_threads = args_tuple
    log_name = f'tuning_{index_name}_{model_type}_L{L:02d}_{config_name}.log'
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    start = time.time()
    rc = 0
    try:
        sys.path.insert(0, str(TUNING_DIR))
        from parameter_tuning import run_single_config

        # Per-Index-CSV ermitteln
        from config import INDICES
        data_path = str(PROJECT_ROOT / 'stockData' / 'preprocessedData' / INDICES[index_name])

        # stdout in Logfile umlenken
        with open(log_path, 'w', encoding='utf-8') as f:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = f, f
            try:
                run_single_config(data_path, model_type, config_name,
                                  index_name=index_name, lookback=L)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
    except Exception as e:
        rc = 1
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[POOL WORKER EXCEPTION]: {type(e).__name__}: {e}\n")
            import traceback
            f.write(traceback.format_exc())
    finally:
        _release_worker_resources()

    elapsed = time.time() - start
    status = 'OK' if rc == 0 else 'FAIL'
    print(f"  [{status}] {index_name:10s} {model_type.upper():4s} L={L:02d} "
          f"{config_name} ({elapsed:.0f}s)")
    return (index_name, model_type, config_name, L, rc, elapsed)


def dispatch_best_predict(args_tuple):
    """In-process Worker: best-predict direkt im Pool-Worker.

    Task-Tupel: (index_name, model_type, L, tf_threads)
    """
    index_name, model_type, L, tf_threads = args_tuple
    log_name = f'predict_{index_name}_{model_type}_L{L:02d}.log'
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    start = time.time()
    rc = 0
    try:
        sys.path.insert(0, str(TUNING_DIR))
        from parameter_tuning import run_best_per_L_test
        from config import INDICES
        data_path = str(PROJECT_ROOT / 'stockData' / 'preprocessedData' / INDICES[index_name])

        with open(log_path, 'w', encoding='utf-8') as f:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = f, f
            try:
                run_best_per_L_test(data_path, model_type, index_name, L)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
    except Exception as e:
        rc = 1
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[POOL WORKER EXCEPTION]: {type(e).__name__}: {e}\n")
            import traceback
            f.write(traceback.format_exc())
    finally:
        _release_worker_resources()

    elapsed = time.time() - start
    status = 'OK' if rc == 0 else 'FAIL'
    print(f"  [{status}] Predict {index_name:10s} {model_type.upper():4s} L={L:02d} "
          f"({elapsed:.0f}s)")
    return (index_name, model_type, L, rc, elapsed)


ALL_MODEL_TYPES = ('lstm', 'cnn', 'gru', 'informer')


def generate_all_config_names(models=None):
    """Generate all config names without importing TensorFlow.

    Args:
        models: None oder iterable von Modell-Keys ('lstm','cnn','gru','informer').
                None = alle Modelle. Sonst nur die genannten — für das gestaffelte
                Phase-1-Vorgehen (z.B. erst LSTM/CNN/GRU, dann Informer).

    DUPLIKATIONS-WARNUNG: Suchraum ist auch in parameter_tuning.py:57-84 definiert.
    Wenn dort etwas geändert wird, muss diese Funktion synchron gehalten werden —
    sonst dispatched die Pipeline andere Config-Namen, als der Tuner erwartet.
    Grund für die Duplizierung: parameter_tuning.py importiert TF auf Modulebene;
    der Pipeline-Orchestrator soll TF NICHT vor dem Pool-Spawn laden.
    """
    import itertools

    requested = set(ALL_MODEL_TYPES) if models is None else set(models)
    out = []

    if 'lstm' in requested:
        for dropout, dense, lr, batch, epochs in itertools.product(
            [0.2, 0.4], [16, 32], [0.001, 0.005], [8, 32], [50, 100]
        ):
            out.append(('lstm', f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"))

    if 'cnn' in requested:
        for kernel, pool, dropout, batch, epochs in itertools.product(
            [3, 5], [2, 4], [0.2, 0.4], [8, 32], [50, 100]
        ):
            out.append(('cnn', f"k{kernel}_p{pool}_d{dropout}_b{batch}_e{epochs}"))

    if 'gru' in requested:
        for dropout, dense, lr, batch, epochs in itertools.product(
            [0.2, 0.4], [16, 32], [0.001, 0.005], [8, 32], [50, 100]
        ):
            out.append(('gru', f"d{dropout}_u{dense}_lr{lr}_b{batch}_e{epochs}"))

    if 'informer' in requested:
        for d_model, n_heads, dropout, lr, batch, epochs in itertools.product(
            [32, 64], [4, 8], [0.05, 0.1], [0.0001, 0.001], [8, 32], [50, 100]
        ):
            out.append(('informer',
                        f"dm{d_model}_h{n_heads}_d{dropout}_lr{lr}_b{batch}_e{epochs}"))

    return out


def get_completed_configs(lookbacks):
    """Check which (index, model, config, L) tuples are already done.

    Unter parameter_tuning/results/{MODEL}/{INDEX}/L{L}/{config}/results.json
    """
    completed = set()
    for model_type in ['LSTM', 'CNN', 'GRU', 'INFORMER']:
        for index_name in ALL_INDICES:
            idx_base = TUNING_RESULTS_DIR / model_type / index_name
            if not idx_base.exists():
                continue
            for L in lookbacks:
                l_dir = idx_base / f'L{L:02d}'
                if not l_dir.exists():
                    continue
                for config_dir in l_dir.iterdir():
                    if config_dir.is_dir() and (config_dir / 'results.json').exists():
                        completed.add((index_name, model_type.lower(), config_dir.name, L))
    return completed


def get_completed_best_predicts(lookbacks):
    """Check which (index, model, L) best-predict tasks are done (pred_test.npz)."""
    completed = set()
    for model_type in ['LSTM', 'CNN', 'GRU', 'INFORMER']:
        for index_name in ALL_INDICES:
            idx_base = TUNING_RESULTS_DIR / model_type / index_name
            if not idx_base.exists():
                continue
            for L in lookbacks:
                if (idx_base / f'L{L:02d}' / 'pred_test.npz').exists():
                    completed.add((index_name, model_type.lower(), L))
    return completed


COARSE_LOOKBACKS = [1, 10, 20, 30, 40, 50, 60]
FINE_LOOKBACKS = list(range(1, 61))  # 1..60 für Phase-3-Test-Predict-Sweep
LOOKBACKS = COARSE_LOOKBACKS  # Backward-compat alias für get_completed_configs

# Hardware-Cap: 16 Worker. Hoeher führt auf 48 GB RAM zu OOM-Crashes (32x TF-Init
# allein ~38 GB). --workers wird hier hart auf <= WORKER_HARD_CAP geclamped.
WORKER_HARD_CAP = 16

# Wird in main() durch --throttle gesetzt; default off. Throttle-Logik liest das.
_ARGS_THROTTLE = False


def _cpu_throttle_ready(min_idle_threads, idle_threshold_pct, sample_interval_s):
    """psutil-basierter Check: True wenn mind. min_idle_threads logische
    CPU-Threads aktuell unter idle_threshold_pct Auslastung liegen.

    Blockt sample_interval_s Sekunden für die Messung — das ist gewollt: das
    Polling-Intervall ist exakt die Messung.

    Returns: (ready_bool, idle_count, total_threads)
    """
    import psutil
    cpu_pct = psutil.cpu_percent(percpu=True, interval=sample_interval_s)
    idle_count = sum(1 for c in cpu_pct if c < idle_threshold_pct)
    return idle_count >= min_idle_threads, idle_count, len(cpu_pct)


def _throttled_dispatch_loop(executor, tasks, max_workers,
                              min_idle_threads=4, idle_threshold_pct=50.0,
                              sample_interval_s=3.0, dispatch_cooldown_s=30.0):
    """Dispatched Tasks an einen ProcessPoolExecutor erst dann, wenn die CPU-Last
    es zulässt. Verlässt sich auf die `_cpu_throttle_ready`-Bedingung.

    - `min_idle_threads`: wie viele logische CPU-Threads gleichzeitig unter
      `idle_threshold_pct` sein müssen, damit ein neuer Task dispatched wird.
    - `dispatch_cooldown_s`: Mindestabstand zwischen zwei aufeinanderfolgenden
      Dispatches — gibt dem letzten Worker Zeit, sich auf CPU zu zeigen, bevor
      die nächste Messung passiert.

    Returns: list of future-results.
    """
    results = []
    pending = list(tasks)
    active = {}  # future -> task
    last_dispatch_ts = 0.0
    n_initial = max(1, min_idle_threads // 2)  # Bootstrap

    # Initial-Dispatch (sofort, ohne CPU-Check) — sonst könnten wir Stunden
    # warten, bevor der erste Task startet.
    for _ in range(min(n_initial, len(pending))):
        t = pending.pop(0)
        future = executor.submit(dispatch_single_config, t)
        active[future] = t
    last_dispatch_ts = time.time()
    print(f"    [throttle] initial dispatch: {len(active)} tasks")

    while pending or active:
        # Sammle fertige Futures (non-blocking via .done())
        for f in [f for f in list(active.keys()) if f.done()]:
            try:
                results.append(f.result())
            except Exception as e:
                t = active[f]
                print(f"    WORKER EXCEPTION {t[0]}/{t[1]}/L{t[3]:02d}/{t[2]}: {e}")
            del active[f]

        # Neuer Dispatch nur wenn (a) was zu tun (b) Pool-Slot frei (c) cooldown
        # abgelaufen (d) CPU-Idle-Bedingung erfüllt.
        now = time.time()
        if (pending and len(active) < max_workers
                and now - last_dispatch_ts >= dispatch_cooldown_s):
            ready, idle, total = _cpu_throttle_ready(
                min_idle_threads, idle_threshold_pct, sample_interval_s)
            if ready:
                t = pending.pop(0)
                future = executor.submit(dispatch_single_config, t)
                active[future] = t
                last_dispatch_ts = time.time()
                print(f"    [throttle] dispatch  (active={len(active)}/{max_workers}, "
                      f"idle {idle}/{total}, pending={len(pending)}): "
                      f"{t[0]}/{t[1]} L{t[3]:02d} {t[2]}")
                continue  # sofort weiter prüfen
            else:
                print(f"    [throttle] wait      (active={len(active)}/{max_workers}, "
                      f"idle {idle}/{total} < {min_idle_threads}, "
                      f"pending={len(pending)})")
        else:
            # Kein Dispatch möglich. Schlafen.
            time.sleep(sample_interval_s)

    return results


def run_parallel_tuning(max_workers, tf_threads, models=None):
    """Phase 1: Coarse Tuning für (Index, Modell, Config) bei 7 L-Werten.

    Volles Soll: 6 Indizes × 160 Configs × 7 L = 6.720 Tasks (alle Modelle).
    Wenn `models` (Subset von ('lstm','cnn','gru','informer')) gesetzt ist,
    werden nur die Configs der genannten Modelle dispatched. Damit kann Phase 1
    in mehreren Etappen abgearbeitet werden (z.B. erst LSTM/CNN/GRU, dann
    Informer separat).

    CPU-Throttling: wenn `informer` im models-Filter ist, dispatched die
    Pipeline jeden neuen Task erst dann, wenn psutil meldet, dass mind. 4
    logische CPU-Threads weniger als 50% Auslastung haben. Das verhindert das
    Vollausreizen bei zähen Informer-Tasks auf CPU (Median ~30 min/Task).

    Train + Val nur, kein Test. L-Werte: COARSE_LOOKBACKS.
    """
    all_configs = generate_all_config_names(models=models)
    total_tasks = len(ALL_INDICES) * len(all_configs) * len(COARSE_LOOKBACKS)
    models_label = 'all' if models is None else ','.join(sorted(models))
    # Throttle ist OPT-IN über die --throttle CLI-Flag, NICHT mehr automatisch
    # an Informer gebunden. Grund: max_tasks_per_child=3 führte bei aggressivem
    # Recycle zu überlappenden Worker-Generationen und RAM-Explosion (37 GB in
    # 1.5 h). Lieber Hard-Cap via --workers, kein Throttle.
    throttle = bool(_ARGS_THROTTLE)

    print("\n" + "#" * 70)
    print(f"# SCHRITT 1: Coarse Hyperparameter-Tuning  (models={models_label})")
    print(f"#   {len(ALL_INDICES)} Indizes × {len(all_configs)} Configs × "
          f"{len(COARSE_LOOKBACKS)} L = {total_tasks} Tasks")
    print(f"#   COARSE L = {COARSE_LOOKBACKS}")
    print(f"#   {max_workers} parallele Worker")
    if throttle:
        print(f"#   THROTTLE aktiv (--throttle): dispatch nur wenn >=4 Threads <50% busy")
    print("#" * 70)

    completed = get_completed_configs(COARSE_LOOKBACKS)

    all_tasks = []
    for index_name in ALL_INDICES:
        for model_type, config_name in all_configs:
            for L in COARSE_LOOKBACKS:
                if (index_name, model_type, config_name, L) not in completed:
                    all_tasks.append((index_name, model_type, config_name, L))

    print(f"  Total: {total_tasks}, Bereits fertig: {len(completed)}, "
          f"Ausstehend: {len(all_tasks)}")

    if not all_tasks:
        print("  Alle Configs bereits abgeschlossen!")
        return True

    # Resilient dispatch mit Retry bei BrokenProcessPool
    start = time.time()
    remaining = list(all_tasks)
    max_retries = 5
    retry = 0
    results = []
    while remaining and retry < max_retries:
        tasks = [(idx, m, n, L, tf_threads) for idx, m, n, L in remaining]
        # max_tasks_per_child wurde entfernt: führte zu überlappenden Worker-
        # Generationen während Recycle und damit zu RAM-Explosion. Lieber
        # Hard-Cap der Worker-Anzahl via --workers.
        try:
            with ProcessPoolExecutor(max_workers=max_workers,
                                     initializer=_pool_initializer,
                                     initargs=(tf_threads,)) as executor:
                if throttle:
                    # CPU-aware throttled dispatch (siehe Funktion für Details)
                    results.extend(_throttled_dispatch_loop(executor, tasks, max_workers))
                else:
                    # Klassischer Bulk-Submit: alle Tasks gleichzeitig in die Pool-Queue
                    future_to_task = {executor.submit(dispatch_single_config, t): t
                                      for t in tasks}
                    for future in as_completed(future_to_task):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            t = future_to_task[future]
                            print(f"    WORKER EXCEPTION {t[0]}/{t[1]}/L{t[3]:02d}/{t[2]}: {e}")
        except BrokenProcessPool as e:
            print(f"\n  WARN: ProcessPool abgestürzt ({e}). Starte neu.")
        finally:
            completed_now = get_completed_configs(COARSE_LOOKBACKS)
            remaining = [(idx, m, n, L) for idx, m, n, L in all_tasks
                         if (idx, m, n, L) not in completed_now]
            retry += 1
            if remaining:
                print(f"  Retry {retry}/{max_retries}: {len(remaining)} ausstehend.")

    elapsed = time.time() - start
    print(f"\n  Tuning abgeschlossen in {elapsed/60:.1f} min ({elapsed/3600:.1f}h); "
          f"{len(completed) + len(results) - len(remaining)} fertig, {len(remaining)} offen")

    return len(remaining) == 0


def pick_global_best_configs():
    """Phase 2a: Pickt für jede (Index, Modell)-Kombination die GLOBAL beste Config
    über die 7 coarse L-Werte (Selektionsmetrik: val_loss).

    Schreibt parameter_tuning/results/<MODEL>/<INDEX>/global_best.json mit:
      {
        "config_name": ...,
        "config": {...},
        "selected_at_L": <coarse_L>,
        "val_loss": ...,
        "val_mare": ...
      }

    Diese Datei wird in Phase 3 gelesen, damit der Sweep über L=1..60 immer
    DIESELBE Config nutzt.
    """
    print("\n" + "#" * 70)
    print(f"# SCHRITT 2: Global beste Config pro (Index, Modell) über coarse L")
    print("#" * 70)

    for index_name in ALL_INDICES:
        for model_type, model_dir_name in [('lstm', 'LSTM'), ('cnn', 'CNN'),
                                            ('gru', 'GRU'), ('informer', 'INFORMER')]:
            idx_base = TUNING_RESULTS_DIR / model_dir_name / index_name
            if not idx_base.exists():
                continue

            # Sammle alle (config, L_coarse) Resultate
            candidates = []  # list of (val_loss, L, results.json content)
            for L in COARSE_LOOKBACKS:
                l_dir = idx_base / f'L{L:02d}'
                if not l_dir.exists():
                    continue
                for cfg_dir in l_dir.iterdir():
                    rj = cfg_dir / 'results.json'
                    if not rj.is_file():
                        continue
                    try:
                        with open(rj) as f:
                            r = json.load(f)
                        if 'metrics' in r and r['metrics'].get('val_loss') is not None:
                            candidates.append((r['metrics']['val_loss'], L, r))
                    except Exception:
                        continue

            if not candidates:
                print(f"  {index_name:10s} {model_dir_name:8s}: KEIN Coarse-Tuning-Ergebnis gefunden")
                continue

            best_val_loss, best_L, best_r = min(candidates, key=lambda x: x[0])
            entry = {
                'config_name': best_r['config_name'],
                'config': best_r['config'],
                'selected_at_L': best_L,
                'val_loss': best_val_loss,
                'val_mare': best_r['metrics'].get('val_mare'),
                'train_loss': best_r['metrics'].get('train_loss'),
                'train_mare': best_r['metrics'].get('train_mare'),
            }
            with open(idx_base / 'global_best.json', 'w') as f:
                json.dump(entry, f, indent=2)
            print(f"  {index_name:10s} {model_dir_name:8s}: best={best_r['config_name']} "
                  f"@ L={best_L:2d} (val_loss={best_val_loss:.6f})")


def run_parallel_best_predict(max_workers, tf_threads):
    """Phase 3: Sweep über L=1..60 mit der GLOBAL besten Config (aus Phase 2).
    Pro (Index, Modell, L) ein Predict auf Train+Val+Test.

    Modell-Beschaffung pro Task (siehe parameter_tuning.run_best_per_L_test):
      - Wenn das in Phase 1 gespeicherte best_cfg-Modell unter
        <MODEL>/<INDEX>/L<L>/<best_cfg>/{model.keras|model.h5|model.pt} vorliegt
        -> LOAD (gilt für L in COARSE_LOOKBACKS = {1,10,20,30,40,50,60}).
        Bilateral Loader: .keras zuerst, .h5 als Fallback.
      - Sonst: build + train mit best_cfg bei sequence_length=L.

    Erzeugt pro (Index, Modell, L):
      - best_test_metrics.json
      - pred_test.npz, pred_train.npz, pred_val.npz

    Tasks: 6 Indizes × 4 Modelle × 60 L = 1 440.
    """
    print("\n" + "#" * 70)
    print(f"# SCHRITT 3: L=1..60 Test-Predict-Sweep")
    print(f"#   {len(ALL_INDICES)} Indizes × 4 Modelle × {len(FINE_LOOKBACKS)} L "
          f"= {len(ALL_INDICES) * 4 * len(FINE_LOOKBACKS)} Tasks")
    print(f"#   Modell-Beschaffung: LOAD bei L in {COARSE_LOOKBACKS} (Phase-1-Cache),")
    print(f"#                       sonst Re-Train mit best_cfg bei sequence_length=L")
    print(f"#   {max_workers} parallele Worker")
    print("#" * 70)

    # Verifiziere global_best.json für jede (Index, Modell)-Kombination
    ready_pairs = []
    for index_name in ALL_INDICES:
        for model_type in ['lstm', 'cnn', 'gru', 'informer']:
            gb_path = TUNING_RESULTS_DIR / model_type.upper() / index_name / 'global_best.json'
            if not gb_path.exists():
                print(f"  WARN: {gb_path} fehlt — skip {index_name}/{model_type}")
                continue
            try:
                with open(gb_path) as f:
                    json.load(f)  # Lesbarkeits-Check
            except Exception as e:
                print(f"  WARN: {gb_path} unleserlich ({e}) — skip")
                continue
            ready_pairs.append((index_name, model_type))

    completed = get_completed_best_predicts(FINE_LOOKBACKS)

    all_tasks = []
    for (idx, model) in ready_pairs:
        for L in FINE_LOOKBACKS:
            if (idx, model, L) not in completed:
                all_tasks.append((idx, model, L))

    total_tasks = len(ready_pairs) * len(FINE_LOOKBACKS)
    print(f"  Total: {total_tasks}, Bereits fertig: {len(completed)}, "
          f"Ausstehend: {len(all_tasks)}")

    if not all_tasks:
        print("  Alle Predictions bereits abgeschlossen!")
        return True

    start = time.time()
    remaining = list(all_tasks)
    max_retries = 5
    retry = 0
    while remaining and retry < max_retries:
        tasks = [(idx, m, L, tf_threads) for idx, m, L in remaining]
        try:
            with ProcessPoolExecutor(max_workers=max_workers,
                                     initializer=_pool_initializer,
                                     initargs=(tf_threads,)) as executor:
                future_to_task = {executor.submit(dispatch_best_predict, t): t for t in tasks}
                for future in as_completed(future_to_task):
                    try:
                        future.result()
                    except Exception as e:
                        t = future_to_task[future]
                        print(f"    WORKER EXCEPTION {t[0]}/{t[1]}/L{t[2]:02d}: {e}")
        except BrokenProcessPool as e:
            print(f"\n  WARN: ProcessPool abgestürzt ({e}). Starte neu.")
        finally:
            completed_now = get_completed_best_predicts(FINE_LOOKBACKS)
            remaining = [(idx, m, L) for idx, m, L in all_tasks
                         if (idx, m, L) not in completed_now]
            retry += 1
            if remaining:
                print(f"  Retry {retry}/{max_retries}: {len(remaining)} ausstehend.")

    elapsed = time.time() - start
    print(f"\n  Predictions abgeschlossen in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    return len(remaining) == 0


# =============================================================================
# STEP 2: COLLECT RESULTS & FIND BEST CONFIGS PER INDEX
# =============================================================================

def collect_results_and_find_best():
    """Phase 4: Aggregation. Liest für jede (Index, Modell)-Kombination:
      - die Val-Kurve der globalen Best-Config bei den 7 Coarse-L-Werten (Phase 1)
      - die Test-Kurve über alle 60 L-Werte (Phase 3)
      - den Test-Score bei best_L (Phase 3, für Backward-Compat als top-level Felder)

    Output parameter_tuning/results/best_configurations.json:
      'SP500': {
         'best_lstm': {
            'L': <best_L>, 'config_name': ..., 'config': {...},
            'val_loss', 'val_mare',                            # bei best_L
            'test_loss', 'test_mare', 'test_rmse',             # bei best_L
            'val_curve_coarse': [ {'L', 'val_loss', 'val_mare'}, ... ]  # 7 Punkte
            'test_curve_fine':  [ {'L', 'test_loss', 'test_mare', 'test_rmse'}, ... ]  # 60 Punkte
         }, ...
      }
    """
    print("\n" + "=" * 70)
    print("SCHRITT 4: Aggregation (Val-Kurve coarse + Test-Kurve fine)")
    print("=" * 70)

    best_configs = {'timestamp': datetime.now().isoformat()}

    for index_name in ALL_INDICES:
        print(f"\n  --- {index_name} ---")
        index_best = {}

        for model_type, model_dir_name in [('lstm', 'LSTM'), ('cnn', 'CNN'),
                                            ('gru', 'GRU'), ('informer', 'INFORMER')]:
            idx_base = TUNING_RESULTS_DIR / model_dir_name / index_name
            if not idx_base.exists():
                print(f"    WARN: {idx_base} fehlt"); continue

            gb_path = idx_base / 'global_best.json'
            if not gb_path.exists():
                print(f"    WARN: global_best.json fehlt für {model_dir_name}"); continue
            with open(gb_path) as f:
                gb = json.load(f)
            cfg_name = gb['config_name']
            best_L = int(gb['selected_at_L'])

            # Val-Kurve coarse: gleiche Config bei jedem Coarse-L
            val_curve = []
            for L in COARSE_LOOKBACKS:
                rj = idx_base / f'L{L:02d}' / cfg_name / 'results.json'
                if not rj.exists():
                    continue
                try:
                    with open(rj) as f:
                        r = json.load(f)
                    m = r.get('metrics', {})
                    val_curve.append({
                        'L': L,
                        'val_loss': m.get('val_loss'),
                        'val_mare': m.get('val_mare'),
                        'train_loss': m.get('train_loss'),
                        'train_mare': m.get('train_mare'),
                    })
                except Exception:
                    continue

            # Test-Kurve fine: best_test_metrics.json für L=1..60
            test_curve = []
            for L in FINE_LOOKBACKS:
                tp = idx_base / f'L{L:02d}' / 'best_test_metrics.json'
                if not tp.exists():
                    continue
                try:
                    with open(tp) as f:
                        tm_l = json.load(f)
                    test_curve.append({
                        'L': L,
                        'test_loss': tm_l.get('test_loss'),
                        'test_mare': tm_l.get('test_mare'),
                        'test_rmse': tm_l.get('test_rmse'),
                        'model_loaded_from_disk': tm_l.get('model_loaded_from_disk', False),
                    })
                except Exception:
                    continue

            # Test bei best_L (Top-Level für Backward-Compat)
            test_path = idx_base / f'L{best_L:02d}' / 'best_test_metrics.json'
            if not test_path.exists():
                print(f"    WARN: Test-Resultat fehlt bei L={best_L} für {model_dir_name}"); continue
            with open(test_path) as f:
                tm = json.load(f)

            index_best[f'best_{model_type}'] = {
                'L': best_L,
                'config_name': cfg_name,
                'config': gb['config'],
                'val_loss': tm.get('val_loss_used_for_selection'),
                'val_mare': tm.get('val_mare'),
                'test_loss': tm.get('test_loss'),
                'test_mare': tm.get('test_mare'),
                'test_rmse': tm.get('test_rmse'),
                'val_curve_coarse': val_curve,
                'test_curve_fine': test_curve,
            }
            print(f"    Best {model_dir_name}: L={best_L:2d} {cfg_name} "
                  f"(Val Loss: {tm.get('val_loss_used_for_selection'):.6f}, "
                  f"Val MARE: {tm.get('val_mare'):.6f}, "
                  f"Test MARE: {tm.get('test_mare'):.6f}, "
                  f"Test-Kurve: {len(test_curve)}/{len(FINE_LOOKBACKS)} L)")

        best_configs[index_name] = index_best

    TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = TUNING_RESULTS_DIR / 'best_configurations.json'
    with open(best_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    print(f"\n  Gespeichert: {best_path}")

    return best_configs


# =============================================================================
# STEP 4: PLOT-GENERATION AUS GESPEICHERTEN DATEN
# =============================================================================

def generate_plots_per_index():
    """Phase 4: erzeugt pro Index alle Plots aus den bereits gespeicherten
    per-L Daten (best_test_metrics.json + pred_test.npz).

    Kein Re-Training. Plots pro Index:
      - 01_mape_vs_lookback.png             (Val + Test MAPE je Modell über L=1..60)
      - 02_predictions_vs_actual.png        (best-L je Modell, Preis-Ebene)
      - 03_scatter_predictions.png          (Scatter pred vs actual)
      - zoom_plots/zoom_<model>_L<best_L>.png  (Zoom-Plot mit Datumsachse)
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "#" * 70)
    print(f"# SCHRITT 4: Plot-Generation pro Index (aus gespeicherten Daten)")
    print("#" * 70)

    best_path = TUNING_RESULTS_DIR / 'best_configurations.json'
    if not best_path.exists():
        print("  FEHLER: best_configurations.json fehlt — Phase 3 nicht gelaufen?")
        return False
    with open(best_path) as f:
        best_configs = json.load(f)

    model_colors = {'lstm': '#2196F3', 'cnn': '#F44336',
                    'gru': '#4CAF50', 'informer': '#9C27B0'}
    model_markers = {'lstm': 'o', 'cnn': 's', 'gru': '^', 'informer': 'D'}

    # Legenden-Position für 01_mape_vs_lookback.png pro Index. Die Kurven liegen
    # je nach Index unterschiedlich, daher manuell pro Index gesetzt, damit die
    # Legende nicht überlappt. Default (nicht gelistet): rechts auf 1/3 Höhe.
    # Werte sind (loc, bbox_to_anchor) — bbox=None -> Standard-loc ohne Anchor.
    mape_legend_pos = {
        'DAX':    ('upper left',    None),
        'NIKKEI': ('upper left',    None),
        'NASDAQ': ('center right',  (0.99, 0.47)),
    }
    mape_legend_default = ('center right', (0.99, 0.33))

    # Zoom-Plot-Helper mit Datumsachse (aus comparison/plot_with_dates.py)
    sys.path.insert(0, str(COMPARE_DIR))
    try:
        from plot_with_dates import plot_one as plot_zoom_dates
    except Exception as e:
        print(f"  WARN: plot_with_dates import fehlgeschlagen ({e})")
        plot_zoom_dates = None

    for index_name in ALL_INDICES:
        idx_cfg = best_configs.get(index_name)
        if not idx_cfg:
            print(f"  {index_name}: kein Eintrag in best_configurations.json")
            continue

        out_dir = COMPARE_RESULTS_DIR / index_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 1) MAPE vs L Plot: Val-Kurve coarse (7 Punkte, gestrichelt) + Test-Kurve fine (60 Punkte) ---
        # MAPE = MARE * 100 (relativer Fehler in Prozent, identische Definition).
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_type in ['lstm', 'cnn', 'gru', 'informer']:
            entry = idx_cfg.get(f'best_{model_type}')
            if not entry:
                continue
            color = model_colors[model_type]
            best_L = entry['L']
            test_mare_at_best = entry.get('test_mare')
            test_mape_at_best = (test_mare_at_best * 100
                                 if test_mare_at_best is not None else None)

            # Val-Kurve coarse (7 Punkte) — als Referenz der Hyperparameter-Selektion
            if entry.get('val_curve_coarse'):
                vc = sorted(entry['val_curve_coarse'], key=lambda x: x['L'])
                vals = [e.get('val_mare') for e in vc]
                ax.plot([e['L'] for e in vc],
                        [v * 100 if v is not None else None for v in vals],
                        marker=model_markers[model_type], linestyle='--',
                        linewidth=1.2, color=color, alpha=0.55,
                        label=f"{model_type.upper()} Val coarse (Best L={best_L})")

            # Test-Kurve fine (60 Punkte) — die eigentliche Sweep-Kurve
            if entry.get('test_curve_fine'):
                tc = sorted(entry['test_curve_fine'], key=lambda x: x['L'])
                tvals = [e.get('test_mare') for e in tc]
                ax.plot([e['L'] for e in tc],
                        [v * 100 if v is not None else None for v in tvals],
                        linestyle='-', linewidth=1.8, color=color, alpha=0.9,
                        label=f"{model_type.upper()} Test sweep "
                              f"(Test MAPE @ L={best_L} = "
                              f"{(test_mape_at_best if test_mape_at_best is not None else float('nan')):.2f} %)")

            # best_L als Stern auf der Test-Kurve hervorheben
            if test_mape_at_best is not None:
                ax.scatter([best_L], [test_mape_at_best], marker='*',
                           s=220, color=color, edgecolor='black', linewidth=1.3,
                           zorder=5)

        ax.set_title(f'{index_name} — Val MAPE coarse (gestrichelt) + Test MAPE Sweep L=1..60 (durchgezogen)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Lookback Window L', fontsize=10)
        ax.set_ylabel('MAPE in % (auf Preis-Ebene)', fontsize=10)
        ax.grid(True, alpha=0.3)
        # Legende INNERHALB der Achse — hält sie im Diagrammbereich, damit
        # bbox_inches='tight' die Plotfläche nicht verkleinert (gleiche Breite wie bisher).
        # Position pro Index aus mape_legend_pos (siehe oben).
        leg_loc, leg_bbox = mape_legend_pos.get(index_name, mape_legend_default)
        if leg_bbox is not None:
            ax.legend(loc=leg_loc, bbox_to_anchor=leg_bbox, fontsize=8, framealpha=0.9)
        else:
            ax.legend(loc=leg_loc, fontsize=8, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(out_dir / '01_mape_vs_lookback.png', dpi=200, bbox_inches='tight')
        plt.close()

        # --- 2) Predictions vs Actual (best L pro Modell, Preis-Ebene) ---
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        for ax, model_type in zip(axes, ['lstm', 'cnn', 'gru', 'informer']):
            entry = idx_cfg.get(f'best_{model_type}')
            if not entry:
                ax.set_visible(False)
                continue
            best_L = entry['L']
            npz_path = (TUNING_RESULTS_DIR / model_type.upper() / index_name /
                        f'L{best_L:02d}' / 'pred_test.npz')
            if not npz_path.exists():
                ax.text(0.5, 0.5, f"pred_test.npz fehlt für {model_type.upper()} L={best_L}",
                        ha='center', va='center')
                continue
            d = np.load(npz_path)
            y_actual = d['y_actual_prices']
            y_pred = d['y_pred_prices']
            ax.plot(y_actual, 'k-', linewidth=1.0, alpha=0.8, label='Actual')
            ax.plot(y_pred, color=model_colors[model_type], linewidth=0.9, alpha=0.75,
                    label=f'{model_type.upper()} (L={best_L})')
            ax.set_title(f'{index_name} — {model_type.upper()} Best L={best_L} '
                         f'— Test MARE={entry["test_mare"]:.4f}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Test Day')
            ax.set_ylabel('Preis')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            d.close()
        plt.tight_layout()
        plt.savefig(out_dir / '02_predictions_vs_actual.png', dpi=200, bbox_inches='tight')
        plt.close()

        # --- 3) Scatter predictions ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for ax, model_type in zip(axes.flat, ['lstm', 'cnn', 'gru', 'informer']):
            entry = idx_cfg.get(f'best_{model_type}')
            if not entry:
                ax.set_visible(False)
                continue
            best_L = entry['L']
            npz_path = (TUNING_RESULTS_DIR / model_type.upper() / index_name /
                        f'L{best_L:02d}' / 'pred_test.npz')
            if not npz_path.exists():
                continue
            d = np.load(npz_path)
            ax.scatter(d['y_actual_prices'], d['y_pred_prices'],
                       s=8, alpha=0.4, color=model_colors[model_type])
            lo = min(d['y_actual_prices'].min(), d['y_pred_prices'].min())
            hi = max(d['y_actual_prices'].max(), d['y_pred_prices'].max())
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5)
            ax.set_title(f'{model_type.upper()} (L={best_L}, MARE={entry["test_mare"]:.4f})',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Actual Preis')
            ax.set_ylabel('Predicted Preis')
            ax.grid(True, alpha=0.3)
            d.close()
        plt.suptitle(f'{index_name} — Predicted vs Actual Preise (best L pro Modell)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / '03_scatter_predictions.png', dpi=200, bbox_inches='tight')
        plt.close()

        # --- 4) Zoom-Plots pro Modell (mit Datumsachse) ---
        if plot_zoom_dates is not None:
            for model_type in ['lstm', 'cnn', 'gru', 'informer']:
                entry = idx_cfg.get(f'best_{model_type}')
                if not entry:
                    continue
                best_L = entry['L']
                try:
                    plot_zoom_dates(index_name, model_type.upper(), best_L,
                                    wide=150, deep=40, dpi=300)
                except Exception as e:
                    print(f"    WARN: Zoom-Plot {index_name}/{model_type.upper()}/"
                          f"L={best_L} fehlgeschlagen: {e}")

        print(f"  {index_name:10s}: Plots erzeugt in {out_dir}")

    return True


# =============================================================================
# STEP 4: CROSS-INDEX COMPARISON
# =============================================================================

_MODEL_PARAM_KEYS = {
    'lstm':     ['dropout', 'dense_units', 'lr', 'batch', 'epochs'],
    'cnn':      ['kernel', 'pool', 'dropout', 'batch', 'epochs'],
    'gru':      ['dropout', 'dense_units', 'lr', 'batch', 'epochs'],
    'informer': ['d_model', 'n_heads', 'dropout', 'lr', 'batch', 'epochs'],
}


def _print_best_config_table(model_type, best_configs, comparison):
    """Tabelle: beste Config pro Index für ein Modell. Mutiert comparison[<m>_configs]."""
    label = model_type.upper()
    print("\n" + "=" * 100)
    print(f"BESTE {label}-KONFIGURATIONEN PRO INDEX")
    print("=" * 100)
    print(f"{'Index':12s} {'Config':35s} {'Val Loss':>10s} {'Val MARE':>10s} "
          f"{'Test MARE':>10s} {'Sel L':>6s}")
    print("-" * 100)
    key = f'best_{model_type}'
    for index_name in ALL_INDICES:
        idx_best = best_configs.get(index_name, {})
        if key not in idx_best:
            continue
        b = idx_best[key]
        print(f"{index_name:12s} {b['config_name']:35s} "
              f"{b['val_loss']:10.6f} {b['val_mare']:10.6f} "
              f"{b['test_mare']:10.6f} {b['L']:6d}")
        comparison[f'{model_type}_configs'][index_name] = {
            'config_name': b['config_name'],
            'config': b['config'],
            'val_loss': b['val_loss'],
            'val_mare': b['val_mare'],
            'test_mare': b['test_mare'],
            'L': b['L'],
        }


def _print_param_frequency(model_type, comparison):
    """Häufigkeit der gewählten Hyperparameter über alle Indizes."""
    label = model_type.upper() if model_type != 'informer' else 'Informer'
    keys = _MODEL_PARAM_KEYS[model_type]
    counts = {p: {} for p in keys}
    for idx, data in comparison[f'{model_type}_configs'].items():
        cfg = data['config']
        for p in keys:
            v = cfg.get(p)
            if v is None:
                continue
            counts[p].setdefault(v, []).append(idx)

    print(f"\n{label} - Häufigkeit der besten Parameter:")
    for p, values in counts.items():
        print(f"  {p}:")
        for v, indices in sorted(values.items(), key=lambda x: -len(x[1])):
            print(f"    {v}: {len(indices)}x ({', '.join(indices)})")


def _print_test_sweep_table(best_configs, comparison):
    """Bestes L im Test-Sweep (L=1..60) pro (Index, Modell). Quelle:
    best_configurations.json[index][best_<m>]['test_curve_fine']. Schreibt
    pro Modell in comparison['<m>_sweep_results'][index] = {best_L, mae, rmse, mare}.
    """
    print("\n" + "=" * 90)
    print("LOOKBACK SWEEP: Bestes L pro Index (min Test MARE aus L=1..60)")
    print("=" * 90)
    print(f"\n{'Index':12s} {'LSTM L':>7s} {'MARE':>10s} {'RMSE':>10s} "
          f"{'CNN L':>7s} {'MARE':>10s} {'RMSE':>10s} "
          f"{'GRU L':>7s} {'MARE':>10s} {'RMSE':>10s} "
          f"{'Inf L':>7s} {'MARE':>10s} {'RMSE':>10s}")
    print("-" * 130)

    for index_name in ALL_INDICES:
        idx_best = best_configs.get(index_name, {})
        row = {}
        for model_type in ('lstm', 'cnn', 'gru', 'informer'):
            entry = idx_best.get(f'best_{model_type}', {})
            curve = entry.get('test_curve_fine') or []
            if curve:
                # min by test_mare; ignoriere Einträge ohne Wert
                valid = [e for e in curve if e.get('test_mare') is not None]
                if valid:
                    best = min(valid, key=lambda e: e['test_mare'])
                    row[model_type] = (best['L'], best['test_mare'],
                                       best.get('test_rmse', float('nan')))
                    comparison[f'{model_type}_sweep_results'][index_name] = {
                        'best_L': best['L'],
                        'mare': best['test_mare'],
                        'rmse': best.get('test_rmse'),
                        'loss': best.get('test_loss'),
                    }
                    continue
            row[model_type] = ('?', float('nan'), float('nan'))

        def _fmt(v):
            return f"{v:10.6f}" if isinstance(v, float) and not math.isnan(v) else f"{'':>10s}"

        print(f"{index_name:12s} "
              f"{str(row['lstm'][0]):>7s} {_fmt(row['lstm'][1])} {_fmt(row['lstm'][2])} "
              f"{str(row['cnn'][0]):>7s}  {_fmt(row['cnn'][1])} {_fmt(row['cnn'][2])} "
              f"{str(row['gru'][0]):>7s}  {_fmt(row['gru'][1])} {_fmt(row['gru'][2])} "
              f"{str(row['informer'][0]):>7s}  {_fmt(row['informer'][1])} {_fmt(row['informer'][2])}")


def _print_config_deviations(comparison):
    """Welche Configs werden über welche Indizes hinweg wiederholt gewählt?"""
    print("\n" + "=" * 90)
    print("KONFIGURATIONSABWEICHUNGEN")
    print("=" * 90)
    for model_type in ('lstm', 'cnn', 'gru', 'informer'):
        label = model_type.upper() if model_type != 'informer' else 'Informer'
        cfgs = comparison[f'{model_type}_configs']
        unique = sorted({d['config_name'] for d in cfgs.values()})
        print(f"\n  {label}: {len(unique)} verschiedene Konfigurationen aus {len(cfgs)} Indizes")
        for cfg_name in unique:
            indices_with = [idx for idx, d in cfgs.items() if d['config_name'] == cfg_name]
            print(f"    {cfg_name}: {', '.join(indices_with)}")


def run_cross_index_comparison():
    """Phase 6: Cross-Index-Vergleich. Liest ausschließlich aus
    parameter_tuning/results/best_configurations.json (Phase 4) — keine Abhängigkeit
    mehr von der toten lookback_evaluation_results.json.
    """
    print("\n" + "#" * 70)
    print("# SCHRITT 6: Cross-Index Vergleich")
    print("#" * 70)

    best_path = TUNING_RESULTS_DIR / 'best_configurations.json'
    if not best_path.exists():
        print("  FEHLER: best_configurations.json nicht gefunden!")
        return

    with open(best_path, 'r') as f:
        best_configs = json.load(f)

    comparison = {
        'timestamp': datetime.now().isoformat(),
        **{f'{m}_configs': {} for m in ('lstm', 'cnn', 'gru', 'informer')},
        **{f'{m}_sweep_results': {} for m in ('lstm', 'cnn', 'gru', 'informer')},
    }

    # 1) Beste Config pro Modell
    for model_type in ('lstm', 'cnn', 'gru', 'informer'):
        _print_best_config_table(model_type, best_configs, comparison)

    # 2) Hyperparameter-Frequenz
    print("\n" + "=" * 90)
    print("HYPERPARAMETER-ANALYSE: Welche Parameter werden bevorzugt?")
    print("=" * 90)
    for model_type in ('lstm', 'cnn', 'gru', 'informer'):
        _print_param_frequency(model_type, comparison)

    # 3) Test-Sweep-Tabelle (best L im L=1..60-Sweep)
    _print_test_sweep_table(best_configs, comparison)

    # 4) Config-Deviation
    _print_config_deviations(comparison)

    # Save
    COMPARE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = COMPARE_RESULTS_DIR / 'cross_index_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Vergleich gespeichert: {comparison_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Masterarbeit Pipeline (L-aware Tuning)')
    parser.add_argument('--workers', type=int, default=8,
                        help=f'Parallele Prozesse (default: 8, max: {WORKER_HARD_CAP})')
    parser.add_argument('--clean-only', action='store_true',
                        help='Nur aufräumen, nicht laufen')
    parser.add_argument('--no-clean', action='store_true',
                        help='Cleanup überspringen (Resume nach Crash)')
    parser.add_argument('--models', type=str, default='all',
                        help='Komma-getrennte Modell-Liste für Phase 1 '
                             '(lstm,cnn,gru,informer). Default "all" läuft die '
                             'gesamte Pipeline (Phase 1-6). Wenn != all, stoppt die '
                             'Pipeline NACH Phase 1 (Phase 2-6 brauchen alle Modelle).')
    parser.add_argument('--throttle', action='store_true',
                        help='Opt-in CPU-Throttle: dispatch nur wenn >=4 logische '
                             'Threads <50%% busy. Default aus. Empfohlen NUR mit '
                             'kleiner --workers Zahl wenn man dynamisch laden lassen will.')
    args = parser.parse_args()

    # Throttle-Flag global verfügbar machen (wird in run_parallel_tuning gelesen)
    global _ARGS_THROTTLE
    _ARGS_THROTTLE = args.throttle

    # --models parsen
    if args.models == 'all':
        selected_models = None  # = alle
    else:
        selected_models = tuple(m.strip().lower() for m in args.models.split(',') if m.strip())
        for m in selected_models:
            if m not in ALL_MODEL_TYPES:
                print(f"FEHLER: Unbekanntes Modell '{m}'. Erlaubt: {ALL_MODEL_TYPES} oder 'all'.")
                sys.exit(1)

    if args.workers > WORKER_HARD_CAP:
        print(f"WARN: --workers {args.workers} > Hardware-Cap {WORKER_HARD_CAP} — "
              f"clamp auf {WORKER_HARD_CAP}.")
        args.workers = WORKER_HARD_CAP
    if args.workers < 1:
        args.workers = 1

    total_threads = os.cpu_count() or 16
    tf_threads = max(2, total_threads // args.workers)

    total_start = time.time()

    n_configs = len(generate_all_config_names(models=selected_models))
    tuning_tasks = len(ALL_INDICES) * n_configs * len(COARSE_LOOKBACKS)
    predict_tasks = len(ALL_INDICES) * 4 * len(FINE_LOOKBACKS)  # L=1..60 Sweep

    print("\n" + "=" * 70)
    print("MASTERARBEIT PIPELINE (Coarse Tuning + L=1..60 Test-Predict-Sweep)")
    print("=" * 70)
    print(f"Gestartet:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Kerne:       {total_threads}")
    print(f"Workers:         {args.workers} (Cap: {WORKER_HARD_CAP})")
    print(f"TF Threads/W:    {tf_threads}")
    print(f"Indizes:         {len(ALL_INDICES)}")
    print(f"Models (Phase 1):{args.models}")
    print(f"Coarse L:        {COARSE_LOOKBACKS}")
    print(f"Fine L:          1..{FINE_LOOKBACKS[-1]}  ({len(FINE_LOOKBACKS)} Werte)")
    print(f"Tuning Tasks:    {len(ALL_INDICES)} × {n_configs} Configs × "
          f"{len(COARSE_LOOKBACKS)} L = {tuning_tasks}")
    if selected_models is None:
        print(f"Predict Tasks:   {len(ALL_INDICES)} × 4 Modelle × {len(FINE_LOOKBACKS)} L "
              f"= {predict_tasks}")
        print(f"Gesamt-Tasks:    {tuning_tasks + predict_tasks}")
    else:
        print(f"Predict Tasks:   --models != all  ->  Phase 2-6 werden übersprungen")
    print()

    # Schritt 0: Bei jedem Pipeline-Lauf komplett aufräumen (Fresh-Start)
    if not args.no_clean:
        clean_results(full=True)

    if args.clean_only:
        print("--clean-only: Fertig.")
        return

    # Schritt 1: Coarse Hyperparameter-Tuning (Train + Val) bei 7 L-Werten
    run_parallel_tuning(args.workers, tf_threads, models=selected_models)

    # Wenn ein Subset von Modellen angegeben wurde -> nach Phase 1 stoppen.
    # Phase 2-6 brauchen alle 4 Modelle (global_best, Cross-Index-Vergleich).
    if selected_models is not None:
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print(f"PHASE 1 für Modelle '{args.models}' ABGESCHLOSSEN")
        print("=" * 70)
        print(f"Dauer: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
        print(f"Nächster Schritt: pipeline_watchdog.py --models <rest> oder ohne --models für "
              f"Phase 2-6.")
        return

    # Schritt 2: Global beste Config pro (Index, Modell) über alle Coarse L
    pick_global_best_configs()

    # Schritt 3: Sweep über L=1..60 mit Global-Best-Config: Train + Test-Predict
    run_parallel_best_predict(args.workers, tf_threads)

    # Schritt 4: Aggregieren über L, best_configurations.json schreiben
    collect_results_and_find_best()

    # Schritt 5: Plots pro Index aus gespeicherten Daten (kein Re-Training)
    generate_plots_per_index()

    # Schritt 6: Cross-Index-Vergleich
    run_cross_index_comparison()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE KOMPLETT ABGESCHLOSSEN")
    print("=" * 70)
    print(f"Gesamtdauer: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
    print(f"Tuning:      {TUNING_RESULTS_DIR}")
    print(f"Vergleich:   {COMPARE_RESULTS_DIR}")
    print(f"Logs:        {LOGS_DIR}")


if __name__ == "__main__":
    main()
