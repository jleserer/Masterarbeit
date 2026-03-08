"""Re-run SP500 sweep for all 60 L-values using subprocess parallelism.
Optimized for AMD AI MAX+ 395 (32 logical cores) — 16 parallel workers.
"""
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BEST_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parameter_tuning', 'results', 'best_configurations.json')
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MAX_WORKERS = 16  # 32 logical cores / 2 = 16 workers (each TF task uses ~2 threads)


def run_one(L):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookback_window_sweep.py')
    cmd = [sys.executable, '-u', script,
           '--index', 'SP500', '--lookback', str(L),
           '--best-configs', BEST_CONFIGS]
    env = os.environ.copy()
    env['TF_NUM_INTRAOP_THREADS'] = '2'
    env['TF_NUM_INTEROP_THREADS'] = '1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env['OMP_NUM_THREADS'] = '2'
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    status = 'OK' if result.returncode == 0 else 'FAIL'
    if status == 'FAIL':
        print(f"  FAIL L={L}: {result.stderr[-300:]}", flush=True)
    return L, status, elapsed


if __name__ == '__main__':
    # Check which L-values are already done
    sweep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'SP500')
    existing = set()
    for f in os.listdir(sweep_dir):
        if f.startswith('L_') and f.endswith('.json'):
            try:
                L = int(f.split('_')[1].split('.')[0])
                existing.add(L)
            except (ValueError, IndexError):
                pass

    tasks = [L for L in range(1, 61) if L not in existing]
    print(f'SP500 sweep: {len(tasks)} L-values to run (already done: {len(existing)})')
    print(f'Using {MAX_WORKERS} parallel workers on AMD AI MAX+ 395 (32 cores)')
    print(f'Each worker: TF_INTRAOP=2, OMP=2 threads')

    start_total = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, L): L for L in tasks}
        for future in as_completed(futures):
            L, status, elapsed = future.result()
            results.append((L, status, elapsed))
            done = len(results)
            total_elapsed = time.time() - start_total
            avg = total_elapsed / done
            remaining = avg * (len(tasks) - done)
            print(f'  [{done}/{len(tasks)}] L={L:2d} {status} ({elapsed:.0f}s) | '
                  f'Elapsed: {total_elapsed/60:.0f}min, ETA: {remaining/60:.0f}min',
                  flush=True)

    total_elapsed = time.time() - start_total
    failed = [r for r in results if r[1] == 'FAIL']
    print(f'\nSP500 sweep done: {len(tasks)-len(failed)}/{len(tasks)} OK '
          f'in {total_elapsed/60:.1f}min ({total_elapsed/3600:.1f}h)')
    if failed:
        print(f'Failed L-values: {sorted([r[0] for r in failed])}')
