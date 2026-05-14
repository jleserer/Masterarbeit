"""
Auto-Restart-Watchdog für run_pipeline.py.

Funktionen:
1. Crash-Erkennung:  Pipeline-Exit != 0 -> Restart mit --no-clean (Resume).
2. Hang-Erkennung:   Wenn HANG_TIMEOUT_S lang kein neuer results.json/pred_test.npz
                     produziert wurde -> Pipeline killen + Restart.
3. Cleanup vor Restart: cleanup_incomplete.py wird bei Hang-Restart aufgerufen
                        (mit mtime-Schutz, siehe cleanup_incomplete.py).

run_pipeline.py clamped --workers hart auf WORKER_HARD_CAP (= 24). Der Watchdog
gibt --workers 1:1 weiter; höhere Werte werden ignoriert/clamped. Die
Crash-Härte ist primär durch Tier-1-Fixes (clear_session, single TF-Threading-
Setup) bereits in der Pipeline selbst adressiert — der Watchdog ist Sicherheits-
netz, kein Workaround mehr.

Usage:
  python pipeline_watchdog.py [--workers 24] [--max-restarts 30] [--hang-timeout 600]
"""
import os
import sys
import time
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
TUNING_RESULTS = PROJECT_ROOT / 'parameter_tuning' / 'results'


def _newest_artifact_mtime():
    """Jüngste Modifikationszeit unter den Pipeline-Output-Dateien.

    Robust gegen WinError 1450 (zu viele File-Handles bei tiefem Recursive-Walk).
    Statt über tausende Dateien zu walken, schauen wir nur auf das Verzeichnis-
    Mtime der <MODEL>/<INDEX>/ Ordner — das wird vom OS aktualisiert, sobald
    eine neue results.json darunter geschrieben wird.
    """
    newest = 0.0
    if not TUNING_RESULTS.exists():
        return 0.0
    try:
        for model_dir in TUNING_RESULTS.iterdir():
            if not model_dir.is_dir() or model_dir.name not in ('LSTM', 'CNN', 'GRU', 'INFORMER'):
                continue
            for index_dir in model_dir.iterdir():
                if not index_dir.is_dir():
                    continue
                # Statt allen Dateien zu lauschen: prüfe nur die L<L>-Ordner-Mtime
                # (parent dir wird beim Schreiben einer Datei darunter aktualisiert)
                try:
                    for l_dir in index_dir.iterdir():
                        if not l_dir.is_dir():
                            continue
                        try:
                            mt = l_dir.stat().st_mtime
                            if mt > newest:
                                newest = mt
                        except OSError:
                            continue
                except OSError:
                    continue
    except OSError:
        # Falls FS gerade erschöpft ist: liefere letzten bekannten Wert (0)
        return newest
    return newest


def _hang_watcher(proc_holder, stop_event, hang_timeout_s, log_fn):
    """Hintergrund-Thread: prüft alle 60s, ob neue Artefakte entstehen."""
    last_artifact_mtime = _newest_artifact_mtime()
    last_check = time.time()
    while not stop_event.is_set():
        time.sleep(30)
        if stop_event.is_set():
            break
        cur = _newest_artifact_mtime()
        now = time.time()
        if cur > last_artifact_mtime:
            last_artifact_mtime = cur
            last_check = now
            continue
        # Keine neue Aktivität
        if now - last_check >= hang_timeout_s:
            log_fn(f"  HANG DETECTED — kein neues Artefakt seit {int(now - last_check)}s "
                   f"(Threshold {hang_timeout_s}s) -> Pipeline killen")
            try:
                proc = proc_holder.get('proc')
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception as e:
                log_fn(f"  Watcher-Kill-Exception: {e}")
            return


def _ram_watcher(proc_holder, stop_event, ram_limit_gb, log_fn):
    """Hintergrund-Thread: kills Pipeline-Subprocess wenn System-RAM-Free
    unter ram_limit_gb fällt. Watchdog macht danach automatisch Restart (mit
    --no-clean Resume + cleanup_incomplete).

    Hintergrund: Informer-Trainings haben einen RAM-Leak, der pro Worker etwa
    1 GB/h akkumuliert. Auto-Recycle gibt das OS-RAM komplett frei und die
    Pipeline kann von Disk-State weiterlaufen.
    """
    try:
        import psutil
    except ImportError:
        log_fn("  RAM-Watcher inaktiv: psutil nicht installiert.")
        return
    while not stop_event.is_set():
        time.sleep(30)
        if stop_event.is_set():
            break
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        if free_gb < ram_limit_gb:
            log_fn(f"  RAM-LIMIT erreicht: nur {free_gb:.2f} GB frei "
                   f"(Threshold {ram_limit_gb} GB) -> Pipeline killen")
            try:
                proc = proc_holder.get('proc')
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception as e:
                log_fn(f"  RAM-Watcher-Kill-Exception: {e}")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24,
                        help='Wird in run_pipeline.py auf WORKER_HARD_CAP=24 geclamped.')
    parser.add_argument('--max-restarts', type=int, default=30,
                        help='Maximale Anzahl an Restarts (default: 30)')
    parser.add_argument('--restart-wait', type=int, default=15,
                        help='Sekunden zwischen Restarts (default: 15)')
    parser.add_argument('--hang-timeout', type=int, default=600,
                        help='Hang-Erkennung: kein neues Artefakt seit N Sekunden (default: 600 = 10 min)')
    parser.add_argument('--start-fresh', action='store_true',
                        help='Erster Run wischt alles. Default: Resume.')
    parser.add_argument('--models', type=str, default='all',
                        help='1:1 weitergereicht an run_pipeline.py. Default "all" '
                             'läuft die gesamte Pipeline. lstm,cnn,gru bzw. informer '
                             'für gestaffeltes Phase-1-Vorgehen.')
    parser.add_argument('--throttle', action='store_true',
                        help='Weiterreichung an run_pipeline.py: opt-in CPU-Throttle.')
    parser.add_argument('--ram-limit-gb', type=float, default=None,
                        help='Wenn System-RAM-Free unter diesen Wert (in GB) fällt, '
                             'wird die Pipeline gekillt und vom Watchdog neu gestartet '
                             '(Auto-Recycle gegen Memory-Leaks). Default: deaktiviert.')
    args = parser.parse_args()

    pipeline = PROJECT_ROOT / 'run_pipeline.py'
    cleanup = PROJECT_ROOT / 'cleanup_incomplete.py'
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PIPELINE WATCHDOG")
    print("=" * 70)
    print(f"Workers:       {args.workers}")
    print(f"Max restarts:  {args.max_restarts}")
    print(f"Restart wait:  {args.restart_wait}s")
    print(f"Hang timeout:  {args.hang_timeout}s")
    print(f"Start fresh:   {args.start_fresh}")
    print(f"Models:        {args.models}")
    print(f"RAM limit:     {args.ram_limit_gb} GB" if args.ram_limit_gb else "RAM limit:     disabled")
    print()

    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {msg}", flush=True)

    attempt = 0
    while attempt <= args.max_restarts:
        attempt += 1
        is_first = (attempt == 1)
        log(f"=== RUN #{attempt} ({'fresh' if (is_first and args.start_fresh) else 'resume'}) ===")

        # Vor jedem Restart inkomplette Configs aufräumen (ausser allerersten Lauf)
        if attempt > 1 and cleanup.exists():
            try:
                subprocess.run([sys.executable, str(cleanup)], cwd=str(PROJECT_ROOT),
                               capture_output=True, timeout=300)
                log("  cleanup_incomplete.py ausgeführt")
            except Exception as e:
                log(f"  cleanup-Exception: {e}")

        cmd = [sys.executable, '-u', str(pipeline),
               '--workers', str(args.workers),
               '--models', args.models]
        if args.throttle:
            cmd.append('--throttle')
        if not (is_first and args.start_fresh):
            cmd.append('--no-clean')

        log_path = log_dir / f'watchdog_run_{attempt:02d}.log'

        # Pipeline starten + Hang-Watcher (und optional RAM-Watcher) in Threads
        proc_holder = {'proc': None}
        stop_event = threading.Event()
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                                        stdout=f, stderr=subprocess.STDOUT)
                proc_holder['proc'] = proc
                hang_thr = threading.Thread(
                    target=_hang_watcher,
                    args=(proc_holder, stop_event, args.hang_timeout, log),
                    daemon=True,
                )
                hang_thr.start()
                ram_thr = None
                if args.ram_limit_gb is not None:
                    ram_thr = threading.Thread(
                        target=_ram_watcher,
                        args=(proc_holder, stop_event, args.ram_limit_gb, log),
                        daemon=True,
                    )
                    ram_thr.start()
                rc = proc.wait()
                stop_event.set()
                hang_thr.join(timeout=2)
                if ram_thr is not None:
                    ram_thr.join(timeout=2)
        except Exception as e:
            log(f"  Watchdog-Exception: {e}")
            rc = -1

        log(f"Run #{attempt} beendet, exit={rc}, log={log_path}")

        if rc == 0:
            print("\nPIPELINE ERFOLGREICH ABGESCHLOSSEN.")
            return 0

        # Vor Restart Python aufräumen — eigene UND Parent-PID ausschließen,
        # sonst killt der Watchdog sich selbst und/oder einen umschließenden
        # Outer-Watchdog (alle sind python.exe).
        own_pid = os.getpid()
        try:
            parent_pid = os.getppid()
        except Exception:
            parent_pid = own_pid
        log(f"  Cleanup (Stop alle Python-Prozesse ausser PID={own_pid}, parent={parent_pid}) ...")
        ps_cmd = (f"Get-Process python -ErrorAction SilentlyContinue "
                  f"| Where-Object {{ $_.Id -ne {own_pid} -and $_.Id -ne {parent_pid} }} "
                  f"| Stop-Process -Force")
        subprocess.run(['powershell.exe', '-Command', ps_cmd],
                       cwd=str(PROJECT_ROOT))
        time.sleep(args.restart_wait)

    print(f"\nMax-Restarts ({args.max_restarts}) erreicht — gebe auf.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
