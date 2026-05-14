"""
DEMO der neuen Pipeline (Coarse + Sweep) mit stark reduzierter Konfiguration,
damit man in wenigen Minuten die später erzeugten Artefakte zur Diskussion stellen kann.

Reduktion gegenüber Production:
  - Indizes:   8         -> 1   (SP500)
  - Modelle:   4         -> 4   (alle, damit Output-Struktur vollständig sichtbar)
  - Configs:   208       -> 2 pro Modell (= 8 Configs)
  - Coarse L:  7         -> 2   ([5, 30])
  - Fine L:    60        -> 3   ([5, 15, 30])

Tasks total:
  Phase 1 (Tuning): 1 Index × 2 Configs × 4 Modelle × 2 L = 16
  Phase 3 (Sweep):  1 Index × 4 Modelle × 3 L            = 12
  -> ~28 Trainings-Runs. Bei 8 Workern Ø 30s -> ~5 Minuten.

Output landet in den üblichen Verzeichnissen (parameter_tuning/results/,
comparison/results/) — anschließend kannst
du das normale `run_pipeline.py --clean-only` aufrufen, um aufzuräumen.

Usage:
  python pipeline_demo.py [--workers 8]
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='Pipeline-Demo')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallele Worker (default: 8)')
    args = parser.parse_args()

    # Module nach Imports patchen — die Globals werden zur Laufzeit gelesen
    import run_pipeline

    DEMO_INDICES = ['SP500']
    DEMO_COARSE_L = [5, 30]
    DEMO_FINE_L = [5, 15, 30]
    DEMO_CONFIGS_PER_MODEL = 2

    run_pipeline.ALL_INDICES = DEMO_INDICES
    run_pipeline.COARSE_LOOKBACKS = DEMO_COARSE_L
    run_pipeline.FINE_LOOKBACKS = DEMO_FINE_L
    run_pipeline.LOOKBACKS = DEMO_COARSE_L  # legacy alias

    _orig_gen = run_pipeline.generate_all_config_names

    def _demo_generate_configs():
        all_cfgs = _orig_gen()
        seen = {}
        out = []
        for model, name in all_cfgs:
            seen.setdefault(model, 0)
            if seen[model] < DEMO_CONFIGS_PER_MODEL:
                out.append((model, name))
                seen[model] += 1
        return out

    run_pipeline.generate_all_config_names = _demo_generate_configs

    total_threads = os.cpu_count() or 16
    tf_threads = max(2, total_threads // args.workers)

    print("\n" + "=" * 70)
    print("PIPELINE DEMO (reduzierte Konfiguration)")
    print("=" * 70)
    print(f"Indizes:         {DEMO_INDICES}")
    print(f"Coarse L:        {DEMO_COARSE_L}")
    print(f"Fine L:          {DEMO_FINE_L}")
    print(f"Configs/Modell:  {DEMO_CONFIGS_PER_MODEL}")
    print(f"Workers:         {args.workers}")
    print(f"TF Threads/W:    {tf_threads}")
    n_configs = len(_demo_generate_configs())
    n_tuning = len(DEMO_INDICES) * n_configs * len(DEMO_COARSE_L)
    n_sweep = len(DEMO_INDICES) * 4 * len(DEMO_FINE_L)
    print(f"Tuning Tasks:    {len(DEMO_INDICES)} × {n_configs} × {len(DEMO_COARSE_L)} = {n_tuning}")
    print(f"Sweep Tasks:     {len(DEMO_INDICES)} × 4 × {len(DEMO_FINE_L)}      = {n_sweep}")
    print()

    # Schritt 0: Aufräumen
    run_pipeline.clean_results(full=True)

    # Schritt 1: Coarse Tuning
    run_pipeline.run_parallel_tuning(args.workers, tf_threads)

    # Schritt 2: Global Best
    run_pipeline.pick_global_best_configs()

    # Schritt 3: Sweep
    run_pipeline.run_parallel_best_predict(args.workers, tf_threads)

    # Schritt 4: Aggregation
    run_pipeline.collect_results_and_find_best()

    # Schritt 5: Plots pro Index
    run_pipeline.generate_plots_per_index()

    # Schritt 6: Cross-Index (entartet bei 1 Index, aber Tabellen werden geschrieben)
    run_pipeline.run_cross_index_comparison()

    print("\n" + "=" * 70)
    print("DEMO ABGESCHLOSSEN")
    print("=" * 70)
    print("\nErzeugte Artefakte:")
    print(f"  parameter_tuning/results/")
    print(f"    <MODEL>/SP500/L05/, L30/                  -> Phase-1 Tuning-Resultate")
    print(f"    <MODEL>/SP500/L05/<config>/results.json   -> Train+Val Metriken")
    print(f"    <MODEL>/SP500/L05/<config>/<config>_mare.png -> Train/Val MARE Bars")
    print(f"    <MODEL>/SP500/global_best.json            -> Phase-2 Auswahl")
    print(f"    <MODEL>/SP500/L05/best_test_metrics.json  -> Phase-3 Sweep + Test")
    print(f"    <MODEL>/SP500/L05/pred_test.npz           -> Test-Predictions")
    print(f"    best_configurations.json                  -> Phase-4 Aggregation")
    print(f"  comparison/results/SP500/")
    print(f"    01_mare_vs_lookback.png                          -> MAE-vs-L Plot")
    print(f"    02_predictions_vs_actual.png                     -> Predictions Plot")
    print(f"    03_scatter_predictions.png                       -> Scatter Plot")
    print(f"    zoom_plots/zoom_<MODEL>_L<best_L>.png            -> Zoom-Plots mit Datumsachse (300 DPI)")


if __name__ == '__main__':
    main()
