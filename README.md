# GNN-Based Football Tactical Analysis System

A Graph Attention Network + LSTM system that takes StatsBomb event data, constructs graph representations of 5-minute match windows, and predicts the **next window's** tactical state — 5 regression targets and 2 classification targets simultaneously.


## How It Works

Raw StatsBomb events (passes, carries, dribbles, defensive actions) are split into overlapping 5-minute windows. Each window is converted into a graph with 12 nodes (one per tactical role), multi-event edges, and global context features. The GAT-LSTM processes sequences of 5 consecutive graphs and predicts what will happen in the 6th window.


## Quick Start

```bash
# 1. Download raw event data
python preprocessing/download_raw.py

# 2. Build graph dataset
python -m preprocessing.dataset

# 3. Train the model
python train.py

# 4. Evaluate on held-out test set
python inference.py

# 5. Rolling next-window prediction for a single match
python match_predictor.py
python match_predictor.py --match_id 3869685
python match_predictor.py --team "France"
```


## File Inventory

```
├── train.py                        # Training (3-way split, focal loss, early stopping)
├── inference.py                    # Evaluation on held-out test set
├── match_predictor.py              # Rolling next-window prediction for a single match
├── baseline_diagnostic.py          # Random vs trained vs mean baseline comparison
├── train_val_test_split.json       # Saved 3-way split (created by train.py)
├── models/
│   ├── __init__.py
│   └── model.py                    # TacticalGATLSTM (GAT spatial + LSTM temporal)
└── preprocessing/
    ├── __init__.py
    ├── download_raw.py             # Fetches StatsBomb open data
    ├── data_pipeline.py            # Extracts passes, shots, defense, carries, dribbles
    ├── utils.py                    # Feature encoding (position → 12 roles, heights, patterns)
    ├── window_slicer.py            # Rolling windows + tactical metric computation
    ├── graph_builder.py            # Window → PyG graph (nodes, edges, globals, targets)
    ├── dataset.py                  # PyG InMemoryDataset builder
    └── sequence_dataset.py         # Groups consecutive windows into (input_seq, target) pairs
```


## Architecture

```
Input: Sequence of 5 consecutive 5-minute window graphs
  │
  ├─ Each window contains: Passes + Carries + Dribbles + Defensive Actions
  │
  ▼
Graph Construction (per window)
  ├─ 12 Nodes (tactical roles: GK, LB, CB_L, CB_R, RB, DM, CM_L, CM_R, AM, LW, RW, ST)
  │   └─ 12 features each (active flag, avg x/y, action share, directional tendency,
  │       pressure ratio, avg distance, high pass ratio, receive share, set piece ratio,
  │       degree centrality, betweenness centrality)
  ├─ Multi-event Edges (passes + carries + dribbles + defense)
  │   └─ 9 features each (action type, distance, angle, pressure, height, pattern,
  │       ΔxT, time gap, cumulative chain xT)
  └─ 5 Global Features (opponent defensive density, counterpress intensity,
      half indicator, score differential, match progress)
  │
  ▼
GAT Spatial Encoder (shared across all 5 timesteps)
  ├─ GATv2Conv Layer 1: 12 → 64 × 4 heads = 256-dim, ELU, dropout 0.15
  ├─ GATv2Conv Layer 2: 256 → 128-dim, 1 head, ELU, dropout 0.10
  ├─ Global Mean Pool → 128-dim team embedding
  └─ Concatenate with 5-dim global context → 133-dim per window
  │
  ▼
LSTM Temporal Module
  ├─ Input: 5 × 133-dim (one per window in the sequence)
  ├─ 2-layer LSTM, 128 hidden, dropout 0.20
  └─ Output: last hidden state → 128-dim
  │
  ▼
Shared MLP Trunk (128 → 64, ReLU)
  │
  ├──→ Regression Head (64 → 5):  Cumulative xT, Press Height, Field Tilt, Verticality, Tempo
  ├──→ Def Posture Head (64 → 3): Low Block / Mid Block / High Press
  └──→ Off Style Head (64 → 3):   Patient / Balanced / Counter
```


## Dataset

- **Source**: StatsBomb Open Data (free)
- **Competitions**: World Cup 2018 & 2022, Euro 2020 & 2024, Copa America 2024
- **Scale**: ~230 matches, ~80K+ window-graphs
- **Dataset name**: `offline_mix_v7_suite`


## Training Details

- **Split**: 70% train / 15% validation / 15% test (match-level, no leakage)
- **Split persistence**: Saved to `train_val_test_split.json` so inference uses the exact same held-out test set
- **Regression loss**: Huber for xT (robust to outliers) + weighted MSE for the other 4 targets
- **Classification loss**: Focal loss (γ=2.0) to focus on hard boundary cases
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Early stopping**: Patience 12, based on validation loss only
- **Gradient clipping**: Max norm 1.0 (prevents LSTM gradient explosion)


## Prediction Targets

### Regression (5 targets — next window)
| Target           | Description                              | Normalisation            |
|------------------|------------------------------------------|--------------------------|
| Cumulative xT    | Offensive threat generation              | log1p                    |
| Press Height     | Average y-coordinate of pressing actions | ÷ 120                   |
| Field Tilt       | Share of touches in opponent's third     | Stored as-is (0–1)      |
| Verticality      | Forward vs lateral passing tendency      | Shifted from [-1,1] to [0,1] |
| Tempo            | Passes per minute                        | ÷ 30                    |

### Classification (2 targets — next window)
| Target            | Classes                              |
|-------------------|--------------------------------------|
| Defensive Posture | Low Block / Mid Block / High Press   |
| Offensive Style   | Patient / Balanced / Counter         |


## Key Design Decisions

**Why next-window prediction instead of same-window?** Predicting the current window's labels from its own features is largely a compression task — the model just learns to reconstruct what it already sees. Predicting the *next* window forces the model to learn temporal dynamics: how tactical states evolve.

**Why match-level splits?** Windows from the same match are highly correlated. A random window-level split would leak information (the model could see window 10 in training and predict window 11 in testing). Match-level splits ensure entire matches are held out.

**Why 3-way instead of 2-way split?** Early stopping selects the best checkpoint based on validation performance. If that same set is reported as the test score, the metrics are optimistically biased. The held-out test set is never touched during training in any form.

**Why focal loss?** Mid Block and Balanced are boundary classes that are harder to distinguish. Standard cross-entropy wastes gradient budget on easy examples. Focal loss downweights confident correct predictions (factor (1−p)^γ) so the model focuses on the ambiguous cases.