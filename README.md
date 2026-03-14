# COMPLETE TACTICAL SUITE — FINAL
# ==================================
# GNN-Based Football Tactical Analysis System
#
# Predicts 8 simultaneous tactical outputs from 5-minute rolling windows
# using a Graph Attention Network (GAT) with 86K parameters.


## PREDICTIONS

### Regression (5 targets)
| Target          | R²    | Description                           |
|-----------------|-------|---------------------------------------|
| Cumulative xT   | 0.615 | Offensive threat generation           |
| Press Height     | 0.693 | Opponent's defensive line position    |
| Field Tilt       | 0.810 | Territorial dominance                 |
| Verticality      | 0.681 | Forward vs lateral passing tendency   |
| Tempo            | 0.936 | Passes per minute                     |

### Classification (3 targets)
| Target           | Accuracy | Description                        |
|------------------|----------|------------------------------------|
| Defensive Posture| 73.8%    | Low Block / Mid Block / High Press |
| Offensive Style  | 75.1%    | Patient / Balanced / Counter       |
| Match Outcome    | 74.9%    | Win / Draw / Loss                  |


## QUICK START

### If starting from scratch:
```bash
# 1. Download raw event data
python preprocessing/download_raw.py

# 2. Build graph dataset
python -m preprocessing.dataset

# 3. Train the model
python train.py

# 4. Evaluate
python inference.py

# 5. Run match timeline (presentation demo)
python match_timeline.py

# 6. Key player identification
python key_player.py
```

### If you already have best_model.pth:
```bash
# Just run the demos
python match_timeline.py                    # Full match dashboard
python match_timeline.py --match_id 3869685 # Different match
python match_timeline.py --team "France"    # Single team
python key_player.py                        # Key player analysis
python key_player.py --match_id 3869685     # Different match
```


## FILE INVENTORY

```
├── train.py                    # Training script (Huber xT loss + class-weighted outcome)
├── inference.py                # Evaluation on test set (all 8 predictions)
├── match_timeline.py           # Full match dashboard visualization
├── key_player.py               # Attention-based player importance analysis
├── models/
│   ├── __init__.py
│   └── model.py                # TacticalGAT (4-head: reg + def + off + outcome)
└── preprocessing/
    ├── __init__.py
    ├── download_raw.py         # Fetches StatsBomb open data
    ├── data_pipeline.py        # Extracts passes, shots, defense, carries, dribbles
    ├── utils.py                # Feature encoding (position→role, heights, patterns)
    ├── window_slicer.py        # Rolling windows + tactical metric computation
    ├── dataset.py              # PyG InMemoryDataset builder
    └── graph_builder.py        # Window→Graph conversion (nodes, edges, globals)
```


## ARCHITECTURE

```
Input: 5-minute window of match events
  │
  ├─ Passes + Carries + Dribbles + Defensive actions
  │
  ▼
Graph Construction
  ├─ 12 Nodes (tactical roles: GK, LB, CB_L, ..., ST)
  │   └─ 12 features each (position, action share, centrality, etc.)
  ├─ Multi-event Edges (passes + carries + dribbles + defense)
  │   └─ 9 features each (action type, distance, ΔxT, time gap, etc.)
  └─ 5 Global features (opp density, counterpress, half, score diff, progress)
  │
  ▼
GATv2Conv Layer 1 (64 × 4 heads = 256-dim)
  → ELU → Dropout(0.1)
  │
  ▼
GATv2Conv Layer 2 (128-dim, 1 head)
  → ELU
  │
  ▼
Global Mean Pool → Concatenate with Global Context
  │
  ▼
Shared Trunk (Linear 133→64, ReLU)
  │
  ├──→ Regression Head (64→5):  xT, Press Height, Tilt, Verticality, Tempo
  ├──→ Def Posture Head (64→3): Low Block / Mid Block / High Press
  ├──→ Off Style Head (64→3):   Patient / Balanced / Counter
  └──→ Outcome Head (64→3):     Loss / Draw / Win
```


## DATASET

- Source: StatsBomb Open Data (free)
- Competitions: WC 2018, WC 2022, Euro 2020, Euro 2024, Copa America 2024
- ~230 matches, ~80K+ window-graphs
- Dataset name: offline_mix_v7_suite
