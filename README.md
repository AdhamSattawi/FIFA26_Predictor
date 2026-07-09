# FIFA 2026 World Cup Predictor

A player-level neural network system that predicts World Cup match outcomes (Home Win / Draw / Away Win) by modeling the tactical potential of individual players — not team-level aggregates.

**Core thesis:** National teams are ephemeral. Rosters change every 4 years, and team-level statistics (league position, cumulative points) suffer from severe concept drift. By modeling from the individual player up — using their recent club performance — we bypass this trap entirely.

---

## 🧠 Architecture Overview

```
Match Input
├── home_players (11, F)  ← Player club stats (goals/90, assists/90, minutes%, age, discipline…)
├── away_players (11, F)  ← Same, ordered by canonical position (GK → RB → CB → LB → CDM → CM → CAM → RW → LW → ST)
└── context (C,)          ← 102 pre-computed features: Elo, rolling form (L5/L10/L20), H2H, fatigue, penalty composure
```

Three progressive neural architectures and a gradient-boosted tree baseline are trained and compared:

| # | Model | Key Idea |
|---|---|---|
| 1 | **Baseline MLP** | Average 11 player vectors → simple team representation |
| 2 | **Tactical CNN** | 1D convolutions scan adjacent positions, learning tactical synergies |
| 3 | **Attention CNN** | Self-attention learns which positions carry predictive signal, with Focal Loss + class weights |
| 4 | **XGBoost baseline**| Gradient-boosted trees trained on context features (highly optimized tabular benchmark) |


---

## 📁 Project Structure

```
FIFA26_Predictor/
├── world_cup_features_dataset.csv     # 49K+ matches with 100 ML features (Gulati dataset)
├── config.py                          # All paths, constants, feature lists, hyperparams
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── lineups/                   # Scraped starting XIs (all_lineups.csv, scrape_progress.csv)
│   │   ├── player_stats/              # Scraped player club stats (all_player_stats.csv)
│   │   └── player_elo/               # PlayerElo snapshot (players.csv, coaches.csv)
│   ├── processed/                     # Merged datasets + player matrices
│   └── features/                      # Normalized train/val/test tensors (.npz)
│
├── src/
│   ├── scraping/
│   │   ├── scrape_lineups.py          # Scrape starting XIs (WC only)
│   │   ├── scrape_lineups_full.py     # Full-coverage resume-safe scraper (WC + Qualifiers + Friendlies)
│   │   ├── scrape_player_stats.py     # Scrape player club stats
│   │   └── utils.py                   # Shared scraping utilities + name mapping
│   ├── processing/
│   │   ├── position_mapping.py        # Canonical 11-slot position ordering
│   │   ├── merge_data.py             # Join Gulati + lineups + player stats
│   │   └── feature_engineering.py    # Build tensors, clip outliers, temporal split
│   ├── models/
│   │   ├── baseline_mlp.py           # Model 1: Naive Average MLP
│   │   ├── tactical_cnn.py           # Model 2: 1D Tactical CNN
│   │   └── attention_cnn.py          # Model 3: Attention CNN
│   ├── train.py                       # Training loop with Focal Loss, mixup, and weighted loss
│   ├── train_xgb.py                   # Train XGBoost baseline model
│   ├── train_ensemble.py              # Optimizes blending weights via SLSQP Log Loss minimization
│   ├── evaluate.py                    # Metrics, confusion matrices, plots (supports test split)
│   ├── predict_2026.py               # Generate WC 2026 predictions using optimal blend
│   └── ensemble_2026.py              # PlayerElo ensemble for 2026
│
└── outputs/
    ├── models/                        # Saved checkpoints (.pt, .json, .pkl) and ensemble weights
    ├── plots/                         # Confusion matrices, loss curves, comparison plots
    └── predictions/                   # Match probability CSVs (test, 2026 predictions)

```

---

## ⚙️ Pipeline

```
1. scrape_lineups.py
2. scrape_player_stats.py
        ↓
3. merge_data.py
        ↓
4. feature_engineering.py
        ↓
5. train.py
        ↓
6. evaluate.py
        ↓
7. predict_2026.py → ensemble_2026.py
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
playwright install chromium
```

### Manual Downloads (Required)

1. **Gulati 102-feature dataset** — already included as `world_cup_features_dataset.csv`

2. **PlayerElo snapshot** (for 2026 ensemble only):
   - Download from https://www.kaggle.com/datasets/playerelo/playerelo
   - Place `players.csv` and `coaches.csv` in `data/raw/player_elo/`

### Running the Pipeline

You can run the entire pipeline (scraping, processing, training, evaluation, and prediction) using the master orchestrator script:

```bash
# Run the full pipeline (including web scraping)
python run_pipeline.py

# Skip the web scraping phase and run processing + training + predictions
# (Uses zero-filled player features to establish a strong team-level baseline immediately)
python run_pipeline.py --skip-scraping

# Run only model training and validation evaluation
python run_pipeline.py --only-train
```

Alternatively, you can run each step manually:

```bash
# 1. Scrape lineups (Choose full-coverage version for training dataset coverage)
python src/scraping/scrape_lineups_full.py

# 2. Scrape player club stats
python src/scraping/scrape_player_stats.py

# 3. Merge data sources
python src/processing/merge_data.py

# 4. Feature engineering and scaling
python src/processing/feature_engineering.py

# 5. Train MLP, CNN, and Attention models (neural networks)
python src/train.py --loss focal --mixup-alpha 0.2 --noise-std 0.05

# 6. Train XGBoost baseline
python src/train_xgb.py

# 7. Learn optimal ensemble stacking weights
python src/train_ensemble.py

# 8. Evaluate and save plots
python src/evaluate.py

# 9. Generate 2026 predictions (runs weighted ensemble)
python src/predict_2026.py

# 10. Run PlayerElo ensemble (requires player_elo data downloaded)
python src/ensemble_2026.py
```


---

## 📊 Data Sources

| Source | Contents | Usage |
|---|---|---|
| [Kriish Gulati / Kaggle](https://www.kaggle.com/datasets/kriishgulati/football-match-results-1872-2026-with-ml-features) | 49K+ international matches, 102 leak-free ML features | Context vector (Elo, form, H2H, fatigue) |
| [Transfermarkt](https://www.transfermarkt.com) | Starting lineups + player club stats | Player-level feature matrices |
| [PlayerElo / Kaggle](https://www.kaggle.com/datasets/playerelo/playerelo) | Daily Elo ratings for 70K+ players | 2026 ensemble only |

---

## 📈 Model Comparison

| Model | Accuracy | Log Loss | F1-Home | F1-Draw | F1-Away | Macro F1 |
|---|---|---|---|---|---|---|
| **XGBoost Baseline** | **60.10%** | **0.8543** | **0.7290** | 0.2759 | 0.6195 | **0.5414** |
| **Tactical CNN** | 49.80% | 0.9572 | 0.5211 | 0.3563 | 0.6387 | 0.5054 |
| **Attention CNN** | 43.27% | 0.9998 | 0.3391 | 0.3569 | 0.6310 | 0.4423 |
| **Baseline MLP** | 43.72% | 0.9843 | 0.3997 | **0.3598** | 0.6004 | 0.4533 |

*(Evaluated on unbiased 2022 World Cup cycle test split. Player matrix coverage has been increased to 31.3% (1,852 matches). Neural network performance will reach its peak once the remaining player stats are scraped to replace the current zero-filled stats.)*


---

## ⚠️ Limitations

Football is a chaotic sport. This model cannot predict early red cards, sudden injuries, or dressing room chemistry. It produces a **probabilistic tactical assessment** of each match based on player quality and form — not a deterministic oracle.

---

## 📄 Background

This project is the successor to the [Ligat Ha'Al Predictive Model](https://github.com/AdhamSattawi/ligat-alifot-analytical-predictive-model), built to overcome the concept drift problem of team-level features in international football prediction.