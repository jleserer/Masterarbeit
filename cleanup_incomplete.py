"""Findet und löscht unvollständige Tuning-Resultate.

Per-Config Verzeichnis wird gelöscht, wenn:
  - results.json fehlt (aber Verzeichnis existiert)
  - results.json existiert, hat aber keinen 'metrics'-Key
  - results.json hat 'error' (Subprocess crashte vor Trainings-Ende)

Plus: Phase-3 Prüfung — best_test_metrics.json XOR pred_test.npz inkonsistent?

Race-Condition-Schutz:
  Einträge die jünger als MTIME_GUARD_S sind, werden nicht angefasst — sie
  könnten gerade von einem laufenden Pool-Worker geschrieben werden. Wenn der
  Watchdog gerade einen Restart macht, kann das anderenfalls noch laufende
  Subprozesse killen, deren halb-geschriebene Files fälschlich als 'bad'
  klassifiziert werden.

Optionen:
  --dry-run     Liste was gelöscht WUERDE, aber nicht löschen
  --mtime-guard Sekunden-Schwelle (default 60). Files jünger als das werden
                nicht angefasst.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TUNING = ROOT / 'parameter_tuning' / 'results'

DEFAULT_MTIME_GUARD_S = 60


def _is_protected_by_mtime(path: Path, now: float, guard_s: int) -> bool:
    """True wenn die jüngste Datei unter `path` (rekursiv 1 Level) jünger als
    guard_s Sekunden ist. Schützt gerade-laufende Worker."""
    try:
        if path.is_file():
            return (now - path.stat().st_mtime) < guard_s
        if not path.is_dir():
            return False
        # jüngste Datei direkt im Verzeichnis
        for entry in path.iterdir():
            try:
                if (now - entry.stat().st_mtime) < guard_s:
                    return True
            except OSError:
                continue
        # plus mtime des Verzeichnisses selbst
        if (now - path.stat().st_mtime) < guard_s:
            return True
    except OSError:
        # Wenn wir das Verzeichnis nicht lesen können, lieber schützen
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Nichts löschen, nur listen.')
    parser.add_argument('--mtime-guard', type=int, default=DEFAULT_MTIME_GUARD_S,
                        help=f'Files jünger als N Sekunden nicht anfassen '
                             f'(default: {DEFAULT_MTIME_GUARD_S}).')
    args = parser.parse_args()

    if not TUNING.exists():
        print(f"  {TUNING} existiert nicht — nichts zu tun")
        return 0

    now = time.time()
    removed = 0
    ok = 0
    incomplete_phase3 = 0
    protected = 0

    def _maybe_remove_dir(p: Path, reason: str):
        nonlocal removed, protected
        if _is_protected_by_mtime(p, now, args.mtime_guard):
            protected += 1
            print(f"  PROTECT (jünger als {args.mtime_guard}s): {p.relative_to(ROOT)}  [{reason}]")
            return
        if args.dry_run:
            print(f"  WOULD-REMOVE: {p.relative_to(ROOT)}  [{reason}]")
        else:
            shutil.rmtree(p, ignore_errors=True)
            print(f"  REMOVE: {p.relative_to(ROOT)}  [{reason}]")
        removed += 1

    def _maybe_remove_file(p: Path, reason: str):
        nonlocal protected
        if _is_protected_by_mtime(p, now, args.mtime_guard):
            protected += 1
            return False
        if args.dry_run:
            print(f"  WOULD-UNLINK: {p.relative_to(ROOT)}  [{reason}]")
        else:
            try:
                p.unlink()
                print(f"  UNLINK: {p.relative_to(ROOT)}  [{reason}]")
            except OSError:
                pass
        return True

    for model_dir in TUNING.iterdir():
        if not model_dir.is_dir() or model_dir.name not in ('LSTM', 'CNN', 'GRU', 'INFORMER'):
            continue
        for index_dir in model_dir.iterdir():
            if not index_dir.is_dir():
                continue
            for l_dir in index_dir.iterdir():
                if not l_dir.is_dir() or not l_dir.name.startswith('L'):
                    continue

                # Phase-3-Spuren prüfen: beide Files müssen vorhanden sein
                test_metrics = l_dir / 'best_test_metrics.json'
                pred_test = l_dir / 'pred_test.npz'
                if test_metrics.exists() != pred_test.exists():
                    cleaned = False
                    if test_metrics.exists():
                        cleaned |= _maybe_remove_file(test_metrics, 'phase3-inconsistent')
                    if pred_test.exists():
                        cleaned |= _maybe_remove_file(pred_test, 'phase3-inconsistent')
                    if cleaned:
                        incomplete_phase3 += 1

                # Phase-1 Configs prüfen
                for cfg_dir in l_dir.iterdir():
                    if not cfg_dir.is_dir():
                        continue
                    rj = cfg_dir / 'results.json'
                    bad = False
                    reason = ''
                    if not rj.exists():
                        bad, reason = True, 'no results.json'
                    else:
                        try:
                            with open(rj) as f:
                                r = json.load(f)
                            if 'metrics' not in r:
                                bad, reason = True, 'no metrics key'
                            elif 'error' in r:
                                bad, reason = True, f'error in results.json: {r.get("error")}'
                        except Exception as e:
                            bad, reason = True, f'json parse error: {e}'

                    if bad:
                        _maybe_remove_dir(cfg_dir, reason)
                    else:
                        ok += 1

    print(f"\nOK Configs:           {ok}")
    print(f"{'WOULD-REMOVE' if args.dry_run else 'Removed':22s}{removed}")
    print(f"Phase-3 inkonsistent: {incomplete_phase3} L-Subdirs aufgeräumt")
    print(f"Protected (mtime):    {protected} files/dirs zu jung — übersprungen")
    if args.dry_run:
        print("\n(Dry-Run: keine Änderungen vorgenommen.)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
