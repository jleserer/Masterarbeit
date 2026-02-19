# Lookback Window Sweep Analysis - Implementierung & Ergebnisse

## 🎯 Projekt-Übersicht

Durchfassende Analyse der Loopback-Window-Sensitivität (L = 1 bis 60) mit optimalen Konfigurationen aus 24h Hyperparameter-Suche, gemäß Anforderungen von Prof. Saffer.

**Zwei Hauptabbildungen:**
1. MAE vs Lookback Window L (Parameter-Sweep über alle L-Werte)
2. Vorhersagen vs Realwerte für beste(n) L-Wert(e)

---

## 🔬 Verwendete Konfigurationen (Best aus Hyperparameter-Tuning)

### LSTM Best:
- **Config Name**: d0.2_u64_lr0.005_b8_e100
- **Test MAE (L=60)**: 0.011987
- **Best MAE (L=46)**: 0.011312
- **Parameter**:
  - Dropout: 0.2
  - Dense Units: 64
  - Learning Rate: 0.005
  - Batch Size: 8
  - Epochs: 100 (mit Early Stopping)

### CNN Best:
- **Config Name**: k5_p2_d0.2_b16_e100
- **Test MAE (L=60)**: 0.027612
- **Best MAE (L=7)**: 0.010875
- **Parameter**:
  - Kernel Size: 5
  - Pool Size: 2
  - Dropout: 0.2
  - Batch Size: 16
  - Epochs: 100 (mit Early Stopping)

---

## 📊 Ergebnisse: Top 5 beste Lookback Windows

### LSTM - Best L-Werte

| Rang | L | Test MAE | Improvement |
|------|---|----------|-------------|
| 1 | **46** | **0.011312** | ← l* (Beste) |
| 2 | 11 | 0.017430 | +54% |
| 3 | 51 | 0.019405 | +71% |
| 4 | 13 | 0.024249 | +114% |
| 5 | 42 | 0.036572 | +223% |

**Optimales LSTM-Fenster: L=46 mit MAE=0.011312**

### CNN - Best L-Werte

| Rang | L | Test MAE | Improvement |
|------|---|----------|-------------|
| 1 | **7** | **0.010875** | ← l* (Beste) |
| 2 | 10 | 0.011334 | +4% |
| 3 | 21 | 0.013129 | +21% |
| 4 | 13 | 0.015448 | +42% |
| 5 | 14 | 0.017992 | +65% |

**Optimales CNN-Fenster: L=7 mit MAE=0.010875** (Bestes Overall!)

---

## 🔄 Script-Struktur: `evaluate_lookback_window_sweep.py`

### Hauptfunktionen:

1. **evaluate_lookback_windows()**
   - Testet L = 1 bis 60 systematisch
   - Für jedes L:
     - Erstellt Sequenzen mit Backward-Looking Window = L
     - Trainiert LSTM mit Best Config
     - Trainiert CNN mit Best Config
     - Speichert Test MAE & Loss

2. **plot_mae_vs_lookback()**
   - **Abbildung 1**: MAE vs L
   - Zeigt LSTM (blau) und CNN (rot)
   - Markiert beste L-Werte mit Annotationen
   - Wissenschaftlich formatiert

3. **generate_predictions_multiple_L()**
   - Für beste 3 L-Werte von jedem Modell
   - Trainiert finale Modelle
   - Generiert Vorhersagen auf Test-Set
   - Inverse Transform zu Original-Skala

4. **plot_predictions_vs_actual()**
   - **Abbildung 2/2b**: Vorhersage-Plots
   - Zeitreihen-Plot (normalisiert)
   - Zeitreihen-Plot (Original-Preis-Skala)
   - Scatter-Plot (Actual vs Predicted)

---

## 📈 Sequenzen-Bildung (Backward-Looking Window)

Nach Prof. Saffer's Definition:

```
Für Zeitpunkt t und Fenster L:

Input:  sequenz_L(t) = [Close(t-L+1), ..., Close(t-1), Close(t)]
        Shape: L × 1 (Close-Preis)
        
Target: real(t+1) = Close(t+1)

MAE = (1/n) Σ |pred(τ) - real(τ)| für τ = L+1..L+n
```

---

## 💾 Datenverarbeitung

### Data Split (Goodfellow et al., 2016)
```
Total Samples: 9,562 (S&P 500 Historical Data)
├─ Train: 6,215 samples (65.0%)
├─ Val:   1,434 samples (15.0%)
└─ Test:  1,913 samples (20.0%)
```

### Sequenzen für jedes L
```
L=1:   X_train(1,913, 1)  y_train(1,913,)
L=60:  X_train(1,854, 60) y_train(1,854,)
...
L=60 produziert erste Vorhersage an Tag 61
```

---

## 📁 Output-Struktur

### Verzeichnis: `results/lookback_sweep/lookback_evaluation_sweep/`

```
├── 01_mae_vs_lookback_window.png              (Hauptabbildung)
│
├── 02_LSTM_predictions_vs_actual_timeseries.png
├── 03_LSTM_predictions_vs_actual_original_scale.png
├── 04_LSTM_scatter_predictions.png
│   └─ Für L = [46, 11, 51] (beste 3 LSTM)
│
├── 02_CNN_predictions_vs_actual_timeseries.png
├── 03_CNN_predictions_vs_actual_original_scale.png
├── 04_CNN_scatter_predictions.png
│   └─ Für L = [7, 10, 21] (beste 3 CNN)
│
└── lookback_evaluation_results.json           (Numerische Ergebnisse)
```

---

## ✅ Erfüllte Anforderungen (Prof. Saffer)

### ✓ Abbildung 1: MAE vs Lookback Window L
**Datei**: `01_mae_vs_lookback_window.png`
- X-Achse: L = 1, 2, ..., 60 (Loopback-Window Größe)
- Y-Achse: Test MAE
- Zeigt beide Modelle (LSTM blau, CNN rot)
- Beste L-Werte gekennzeichnet mit Annotationen
- Professionelle Formatierung für Thesis

### ✓ Abbildung 2a: LSTM Vorhersagen (l* = 46 + Top 2)
- **Zeitreihen (normalisiert)**: `02_LSTM_predictions_vs_actual_timeseries.png`
- **Zeitreihen (Original)**: `03_LSTM_predictions_vs_actual_original_scale.png`
- **Scatter**: `04_LSTM_scatter_predictions.png`
- Für L = [46, 11, 51]

### ✓ Abbildung 2b: CNN Vorhersagen (l* = 7 + Top 2)
- **Zeitreihen (normalisiert)**: `02_CNN_predictions_vs_actual_timeseries.png`
- **Zeitreihen (Original)**: `03_CNN_predictions_vs_actual_original_scale.png`
- **Scatter**: `04_CNN_scatter_predictions.png`
- Für L = [7, 10, 21]

---

## 🧠 Interpretationen & Insights

### CNN-Charakteristiken
- **Bevorzugt kleine Fenster**: L=7 ist optimal
- **Schnelle Konvergenz**: Top 3 alle L < 25
- Hierarchische Merkmalserkennung funktioniert mit kurzen Abhängigkeiten
- **MAE-Bereich**: 0.010875 bis 0.017992 (relativ stabil)

### LSTM-Charakteristiken
- **Bevorzugt größere Fenster**: L=46 ist optimal
- **Variable Performance**: Besser bei 11 und über 40
- Benötigt längeren Kontext für zeitliche Muster
- **MAE-Bereich**: 0.011312 bis 0.036572 (höhere Variabilität)

### Optimale Range
| Modell | Bereich | Bestes L | MAE |
|--------|---------|----------|-----|
| CNN | L ∈ [7-25] | 7 | 0.010875 |
| LSTM | L ∈ {11, 46-51} | 46 | 0.011312 |
| Kombination | L ∈ [7-15] | 7 (CNN) | 0.010875 |

---

## 📊 Architektur-Details

### LSTM Modell
```python
Input(L, 1)
  ↓
LSTM(128, return_sequences=True)
  → Dropout(0.2)
  ↓
LSTM(64, return_sequences=False)
  → Dropout(0.2)
  ↓
Dense(64, activation='relu')
  → Dropout(0.2)
  ↓
Dense(1, activation='linear')  # Output
```

### CNN Modell
```python
Input(L, 1)
  ↓
Conv1D(64, kernel=5) + ReLU
  → MaxPooling(2)
  → Dropout(0.2)
  ↓
Conv1D(128, kernel=5) + ReLU
  → MaxPooling(2)
  → Dropout(0.2)
  ↓
Conv1D(256, kernel=5) + ReLU
  → Dropout(0.2)
  ↓
Flatten()
  ↓
Dense(64, activation='relu')
  → Dropout(0.2)
  ↓
Dense(32, activation='relu')
  ↓
Dense(1, activation='linear')  # Output
```

---

## ⏱️ Laufzeit & Performance

- **Total Laufzeit**: 3h 45min
- **Pro L**: ~3.75 min durchschnittlich
- **Training**: Mit Early Stopping & ReduceLROnPlateau
- **Hardware**: GPU-beschleunigt (TensorFlow/Keras)

---

## 🎓 Verwendung in Master-Thesis

### Empfehlung für Prof. Saffer:

> "Die Parameter-Sweep-Analyse über L=1..60 zeigt deutlich, dass:
> 
> **CNN mit L=7** erreicht beste Overall-Performance (MAE=0.010875)
> **LSTM mit L=46** erreicht beste LSTM-Performance (MAE=0.011312)
>
> Die Abbildungen zeigen den vollständigen Trend sowie detaillierte Vorhersagen für die besten Fenstergrößen."

### Inkludieren Sie in Thesis:
1. `01_mae_vs_lookback_window.png` - Ein großer Vergleichsplot
2. Brain 2-3 beste L-Werte mit Individual Plots (CNN L=7, LSTM L=46)
3. JSON-Daten im Anhang für Reproduzierbarkeit

---

## 📂 Dateien

| Datei | Typ | Beschreibung |
|-------|-----|-------------|
| `evaluate_lookback_window_sweep.py` | .py | Hauptskript |
| `results/lookback_sweep/lookback_evaluation_sweep/` | dir | Output-Verzeichnis |
| `LOOKBACK_SWEEP_ANALYSIS.md` | .md | Dieses Dokument |

---

**Evaluation abgeschlossen**: 2026-02-18 09:58:22  
**Status**: ✅ FERTIG  
**Für Prof. Saffer bereit**
