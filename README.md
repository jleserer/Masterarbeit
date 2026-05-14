# Masterarbeit — Stock Price Prediction mit Deep Learning

Vergleich von **LSTM, CNN, GRU und Informer** zur Vorhersage von 6 Finanzindizes auf Basis von **Log-Returns**.

## Pipeline-Architektur

Die Pipeline kombiniert eine **coarse Hyperparameter-Suche** über sieben Lookback-Window-Größen mit einem **vollen L=1..60 Test-Predict-Sweep** auf der jeweils global besten Config. Sechs Phasen, Test-Daten werden **erst in Phase 3** angefasst (Data-Leakage-Vermeidung).

```
SCHRITT 1: Coarse Hyperparameter-Tuning
   6 Indizes × 160 Configs × 7 L-Werte = 6 720 Tasks
   Configs = LSTM 32 + CNN 32 + GRU 32 + Informer 64 = 160
   COARSE_LOOKBACKS = [1, 10, 20, 30, 40, 50, 60]
   Trainiert jede Config bei jedem coarse L. Nur Train + Val.
   Speichert pro Config: results.json, model.keras (bzw. model.pt für Informer), history.json, pred_train.npz, pred_val.npz.

SCHRITT 2: Global-Best-Auswahl
   Pro (Index, Modell): wähle (config, L) mit niedrigstem Val-Loss
   → schreibt <MODEL>/<INDEX>/global_best.json

SCHRITT 3: L=1..60 Test-Predict-Sweep
   6 Indizes × 4 Modelle × 60 L = 1 440 Tasks (unabhängig von der Configs-Anzahl)
   Pro Task: global-best Config bei sequence_length = L, predict auf Test.
     - L ∈ COARSE_LOOKBACKS  →  LOAD Phase-1-Modell (model.keras / .h5 fallback / model.pt)
     - sonst                 →  Re-Train mit best_cfg bei sequence_length = L
   → schreibt L<L>/{best_test_metrics.json, pred_test.npz, pred_train.npz, pred_val.npz}

SCHRITT 4: Aggregation
   Sammelt Val-Kurve coarse (7 Punkte) + Test-Kurve fine (60 Punkte)
   → parameter_tuning/results/best_configurations.json

SCHRITT 5: Plots pro Index — landen in comparison/results/<INDEX>/
   01_mare_vs_lookback.png   Val coarse (gestrichelt) + Test sweep L=1..60 (durchgezogen) + best_L Stern
   02_predictions_vs_actual  best-L Predictions, Preis-Ebene
   03_scatter_predictions    Predicted vs Actual Preise
   zoom_plots/zoom_<MODEL>_L<best_L>.png  (300 DPI, mit Datumsachse)

SCHRITT 6: Cross-Index Vergleich
   cross_index_comparison.json + Pro-Index extended_metrics.json
   (Directional Accuracy, MAE-Improvement vs. Baseline, R²)
```

### Aufruf

Hardware-Cap: `--workers` wird hart auf 24 begrenzt (32 Worker hatten auf 48 GB RAM mehrfach OOM-Crashes verursacht).

```bash
# Empfohlen: Watchdog (Auto-Restart bei Crashes)
python pipeline_watchdog.py --workers 24 --max-restarts 30

# Direkt (ohne Watchdog)
python run_pipeline.py --workers 24

# Resume nach Crash (lässt fertige Configs unangetastet)
python run_pipeline.py --workers 24 --no-clean

# Nur aufräumen (alle Ergebnisse löschen)
python run_pipeline.py --clean-only
```

### Demo

```bash
python pipeline_demo.py --workers 8     # ~5 Minuten, 1 Index, 2 Configs/Modell
```

## Wissenschaftliche Methodik

### Datenfluss

```
preprocessedData/*.csv  (absolute Schlusskurse)
   ├─ Log-Returns:  r(t) = ln(P(t) / P(t-1))
   ├─ Chronologischer Split (kein Shuffle): 65 % Train / 15 % Val / 20 % Test
   ├─ Sequenzen pro Split (Lookback L)
   ├─ Training auf Train, Selektion auf Val, Reporting auf Test
   └─ Inverse-Transform Returns → Preise:  P(t) = P(t-1) · exp(r(t))
```

### Selektion und Reporting (kein Data-Leakage)

| Set | Rolle | Anzahl Evaluationen |
|---|---|---|
| **Train** | Gewichts-Optimierung | per Epoche |
| **Val** | Hyperparameter- und L-Auswahl (Phase 2) | einmal pro Config × Coarse-L |
| **Test** | Reporting (Phase 3) | 60× pro (Index, Modell) — L=1..60 — niemals für Selektion |

Wichtig: Test wird **erst nach abgeschlossener Selektion** angefasst. Phase 2 wählt
`best_cfg` und `best_L` ausschließlich anhand `val_loss`. Phase 3 fixiert diese
beiden Größen und predicted dann über `L=1..60` auf Test — das ist Reporting,
keine Selektion. Best-L bleibt der in Phase 2 anhand Val gewählte Wert.

### Metriken

- **MSE-Loss** auf Log-Returns (Optimierungszielgröße)
- **MARE** (Mean Absolute Relative Error) **auf Preisen** nach Inverse-Transform
  - `MARE = mean(|p_pred - p_true| / (|p_true| + ε))`
  - Auf Preis-Ebene interpretierbar (≈ relativer Preisfehler in %)
- Phase 6 zusätzlich: Directional Accuracy, MAE vs. Naive-Zero-Baseline, R²

### Reproduzierbarkeit

- Seeds (`random`, `numpy`, `tf.random`, `torch.manual_seed`, `PYTHONHASHSEED`) in jedem Subprozess gesetzt
- Per-Index Best Config (kein Shared Hyperparameter-Bias zwischen Indizes)
- Resume-Support (`--no-clean`) und Watchdog mit Auto-Restart

## Modelle

| Modell | Framework | Architektur | Tuning-Configs | Tunable HPs |
|---|---|---|---|---|
| LSTM | TensorFlow/Keras | 2× LSTM (128, 64) + Dense | 32 | dropout × dense × lr × batch × epochs |
| CNN | TensorFlow/Keras | 3× Conv1D (64, 128, 256) + Dense | 32 | kernel × pool × dropout × batch × epochs |
| GRU | TensorFlow/Keras | 2× GRU (128, 64) + Dense | 32 | dropout × dense × lr × batch × epochs |
| Informer | PyTorch | ProbSparse Attention, Encoder-Decoder | 64 | d_model × n_heads × dropout × lr × batch × epochs |

`batch ∈ {8, 32}` für alle Modelle (16 wurde nach Auswertung der ersten Indizes
entfernt — verlor durchgängig in allen drei Keras-Modellen).

Konstanten zentral in [config.py](config.py). Detaillierte Architektur: [models/README.md](models/README.md). Die Standalone-`models/*_model.py` lesen die Konstanten ebenfalls aus `config.py` — keine Duplikate mehr.

## Verzeichnisstruktur

```
Masterarbeit/
├── run_pipeline.py                    Pipeline-Orchestrator (6 Phasen)
├── pipeline_watchdog.py               Auto-Restart-Wrapper (Inner)
├── pipeline_demo.py                   Reduzierte Demo (~5 min)
├── cleanup_incomplete.py              Bereinigt halbe Phase-1/3-Outputs (mtime-safe)
├── config.py                          Zentrale Konstanten (INDICES, Architektur, Seeds)
├── quick_verify.py                    Smoke-Test (nutzt config-Architektur)
│
├── data_preparation/
│   └── data_preparation.py            DataPreparator: Log-Returns + Splits + Sequenzen
│
├── parameter_tuning/
│   ├── parameter_tuning.py            FullGridSearchTuner (--lookback L, --mode tune|best-predict)
│   ├── generate_tuning_tables.py      Pro (Index, Modell, L): CSV/MD/PNG-Tabellen
│   └── results/
│       ├── <MODEL>/<INDEX>/
│       │   ├── L<L>/<config>/         results.json, model.keras (Keras 3) oder model.pt,
│       │   │                          history.json, <config>_mare.png, pred_*.npz
│       │   ├── L<L>/                  best_test_metrics.json, pred_test.npz, pred_train.npz, pred_val.npz
│       │   │                          (für alle L=1..60 nach Phase 3)
│       │   └── global_best.json       Phase-2 Auswahl
│       └── best_configurations.json   Phase-4 Aggregation (val_curve_coarse + test_curve_fine)
│
├── comparison/
│   ├── post_processing.py             Phase 6: erweiterte Metriken — liest pred_test.npz (kein Re-Train mehr)
│   ├── plot_with_dates.py             Zoom-Plot-Generator mit Datumsachse (von run_pipeline importiert + standalone nutzbar)
│   └── results/
│       ├── <INDEX>/                   Pipeline-Phase-5 Plots (Val/Test, Predictions):
│       │                                01_mare_vs_lookback.png
│       │                                02_predictions_vs_actual.png
│       │                                03_scatter_predictions.png
│       │                              Phase-6 erweiterte Metriken:
│       │                                01_directional_accuracy.png, 02_mae_vs_baseline.png,
│       │                                03_cumulative_returns.png, 04_returns_overlay.png,
│       │                                extended_metrics.json
│       │                              Zoom-Plots (300 DPI, mit Datumsachse):
│       │                                zoom_plots/zoom_<MODEL>_L<best_L>.png
│       ├── predictions/<INDEX>/       predictions.npz (für Plot-Regeneration)
│       ├── cross_index_comparison.json
│       └── cross_index_*.png          DA, MAE-Improvement, R² Heatmaps
│
├── models/
│   ├── base_keras_model.py            BaseKerasTimeSeriesModel (gemeinsame Pipeline)
│   ├── lstm_model.py / cnn_model.py / gru_model.py     TensorFlow/Keras (Subklassen)
│   ├── informer_model.py                                PyTorch
│   └── README.md
│
└── stockData/
    ├── sourceData/                    Yahoo-Finance-Rohdaten
    ├── preprocessedData/              Bereinigte CSVs (gemeinsamer Startzeitpunkt)
    ├── get-data/                      Download- und Normalize-Skripte
    └── plots/                         Datenvisualisierungen
```

## Schnellstart

### Voraussetzungen

```bash
pip install tensorflow torch pandas numpy matplotlib seaborn yfinance
```

### Datenabruf (einmalig)

```bash
python stockData/get-data/download_stock_data.py
python stockData/get-data/normalize_data.py
```

### Pipeline starten

```bash
python pipeline_watchdog.py --workers 24
```

Auf 16 C / 32 T:
- **Phase 1 (Coarse Tuning):** 6 720 Tasks → ca. 2–3 h (dominiert die Laufzeit).
- **Phase 3 (L=1..60 Sweep):** 1 440 Tasks, davon 6×4×7 = 168 reine Loads aus Phase 1. Die übrigen ~1 272 Tasks sind Re-Trainings mit best_cfg bei nicht-coarse L. Geschätzt 25–40 min mit 24 Workern.
- **Plots + Cross-Index:** Sekunden.

Werte über `--workers 24` werden geclamped (siehe oben).

### Manuelles Re-Plotting

Alle Plots können aus den gespeicherten `.npz` und `.json` ohne Re-Training regeneriert werden:

```bash
python parameter_tuning/generate_tuning_tables.py    # CSV/MD/PNG Tabellen pro (Index, Modell, L)
python comparison/plot_with_dates.py --index SP500 --L 30      # Zoom-Plots mit Datumsachse (DD.MM.YYYY)
python comparison/plot_with_dates.py --index ALL --L best      # alle Indizes mit best_L pro Modell
```

## 6 Indizes

| Index | Datei |
|---|---|
| S&P 500 | `SP500_historical_data.csv` |
| NASDAQ | `NASDAQ_historical_data.csv` |
| DAX | `DAX_historical_data.csv` |
| Nikkei 225 | `NIKKEI_historical_data.csv` |
| Hang Seng | `HANG_SENG_historical_data.csv` |
| 10Y Bond | `10-Year Bond_historical_data.csv` |

## Referenzen

- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *arXiv:1406.1078*.
- LeCun, Y., Bengio, Y., Hinton, G. (2015). Deep learning. *Nature*, 521(7553).
- Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021* (Best Paper).
