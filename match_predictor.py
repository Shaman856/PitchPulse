"""
MATCH PREDICTOR — Rolling Next-Window Prediction
==================================================
Takes a match ID, processes it into windows, then rolls through the match
chronologically: at each position, feeds the last 5 windows to the GAT-LSTM
and predicts the next window's tactical metrics.

Compares predicted vs actual across the full 90 minutes.

Usage:
    python match_predictor.py                        # Default: WC 2018 Final
    python match_predictor.py --match_id 3869685     # Any match
    python match_predictor.py --team "France"         # Single team focus
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import warnings
from collections import defaultdict
from torch_geometric.data import Batch

from preprocessing.data_pipeline import fetch_match_data
from preprocessing.utils import encode_features, encode_action_features
from preprocessing.window_slicer import get_rolling_windows
from preprocessing.graph_builder import build_graph_from_window
from models.model import TacticalGATLSTM

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_PATH = "best_model.pth"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN    = 5

REG_NAMES   = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']
REG_UNITS   = ['xT',            'pitch units',  'ratio',      '[-1, +1]',    'passes/min']
DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
OFF_CLASSES = ['Patient',   'Balanced',  'Counter']


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def load_model(sample_graph):
    """Load the trained GAT-LSTM model with auto-detected dimensions."""
    model = TacticalGATLSTM(
        num_node_features=sample_graph.x.shape[1],
        num_reg_targets=5,
        num_def_classes=3,
        num_off_classes=3,
        edge_dim=sample_graph.edge_attr.shape[1],
        global_dim=sample_graph.u.shape[1],
        lstm_hidden=128,
        lstm_layers=2,
        seq_len=SEQ_LEN,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def denormalize_single(reg_tensor):
    """
    Denormalize a single [5] regression prediction to human-readable units.
    Matches the encoding in graph_builder.py.
    """
    vals = reg_tensor.cpu().numpy().copy()
    vals[0] = np.expm1(vals[0])         # Cumulative xT: log1p → expm1
    vals[1] = vals[1] * 120.0           # Press Height: /120 → *120
    # vals[2] unchanged                 # Field Tilt: stored as-is
    vals[3] = vals[3] * 2.0 - 1.0      # Verticality: (v+1)/2 → v*2-1
    vals[4] = vals[4] * 30.0           # Tempo: /30 → *30
    return vals


def compute_r2(actual, predicted):
    """R² per metric. Handles single-metric arrays."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def prepare_match(match_id):
    """
    Phase 1: Fetch match data, encode, slice windows, build graphs.
    Returns graphs grouped by team with metadata, plus goals timeline.
    """
    print(f"1. Fetching match {match_id}...")
    raw = fetch_match_data(match_id)

    if raw['passes'].empty:
        print("   ERROR: No pass data found for this match.")
        return None, None

    # Encode features (identical to train pipeline)
    raw['passes'] = encode_features(raw['passes'])
    if not raw['carries'].empty:
        raw['carries'] = encode_action_features(raw['carries'], 'carry')
    if not raw['dribbles'].empty:
        raw['dribbles'] = encode_action_features(raw['dribbles'], 'dribble')
    if not raw['defense'].empty:
        raw['defense'] = encode_action_features(raw['defense'], 'defense')

    print("2. Building windows...")
    windows = get_rolling_windows(raw, match_id, window_size=5, stride=1)

    # Build graphs and attach metadata
    print("3. Building graphs...")
    team_graphs = defaultdict(list)

    for w in windows:
        g = build_graph_from_window(w)
        g._start = w['start_time']
        g._end   = w['end_time']
        g._mid   = (w['start_time'] + w['end_time']) / 2.0
        team_graphs[w['team_name']].append(g)

    # Sort each team's graphs by window_id
    for team in team_graphs:
        team_graphs[team].sort(key=lambda g: int(g.window_id))

    # Extract goals timeline
    shots = raw['shots']
    goals = None
    if not shots.empty and 'is_goal' in shots.columns:
        goal_rows = shots[shots['is_goal'] == True]
        if not goal_rows.empty:
            goals = goal_rows[['minute', 'second', 'player', 'team']].copy()
            goals['time_min'] = goals['minute'] + goals['second'] / 60.0

    total_graphs = sum(len(v) for v in team_graphs.values())
    print(f"   Built {total_graphs} graphs across {len(team_graphs)} teams")

    if goals is not None:
        print(f"   Goals: {len(goals)}")
        for _, g in goals.iterrows():
            print(f"      {int(g['minute'])}' — {g['player']} ({g['team']})")

    return dict(team_graphs), goals


def run_rolling_prediction(model, graphs, team_name):
    """
    Phase 2: Roll through a team's windows and predict each next window.

    For windows [w0, w1, w2, ..., wN], at each position i >= SEQ_LEN:
        Input:  [w_{i-5}, w_{i-4}, w_{i-3}, w_{i-2}, w_{i-1}]
        Target: w_i (actual labels)
        Predict: model(input) → predicted labels for w_i

    Skips sequences that cross the halftime boundary.

    Returns list of dicts with time, predicted values, actual values.
    """
    results = []

    for i in range(SEQ_LEN, len(graphs)):
        target_graph = graphs[i]
        input_graphs = graphs[i - SEQ_LEN : i]

        # ── HALFTIME CHECK ────────────────────────────────────────────
        # u[0, 2] = half_indicator: 0.0 = first half, 1.0 = second half
        # If any input window is in a different half than the target,
        # this sequence crosses halftime — skip it.
        target_half = int(target_graph.u[0, 2].item())
        crosses_halftime = False
        for g in input_graphs:
            if int(g.u[0, 2].item()) != target_half:
                crosses_halftime = True
                break

        if crosses_halftime:
            continue

        # ── INFERENCE ─────────────────────────────────────────────────
        # Wrap each graph in a Batch (adds .batch attribute needed by
        # global_mean_pool). This simulates batch_size=1.
        sequence = [Batch.from_data_list([g]).to(DEVICE) for g in input_graphs]

        with torch.no_grad():
            reg_pred, cls_def_pred, cls_off_pred = model(sequence)

        # ── DENORMALIZE ───────────────────────────────────────────────
        pred_reg = denormalize_single(reg_pred[0])
        actual_reg = denormalize_single(target_graph.y[0])

        pred_def  = cls_def_pred.argmax(1).item()
        pred_off  = cls_off_pred.argmax(1).item()
        actual_def = int(target_graph.y_cls[0, 0].item())
        actual_off = int(target_graph.y_cls[0, 1].item())

        results.append({
            'time_mid':    target_graph._mid,
            'start':       target_graph._start,
            'end':         target_graph._end,
            # Regression
            'pred_reg':    pred_reg,
            'actual_reg':  actual_reg,
            # Classification
            'pred_def':    pred_def,
            'actual_def':  actual_def,
            'pred_off':    pred_off,
            'actual_off':  actual_off,
        })

    return results


def plot_team_dashboard(results, team_name, goals, teams, match_id):
    """
    Phase 3: Visualization — 7-panel dashboard for one team.
    5 regression panels (predicted vs actual line plots) +
    2 classification panels (predicted vs actual step plots).
    """
    if not results:
        print(f"   No predictions for {team_name} (not enough windows)")
        return

    times = [r['time_mid'] for r in results]

    # Goal times for vertical markers
    team_goals = goals[goals['team'] == team_name] if goals is not None else None
    opp_goals  = goals[goals['team'] != team_name] if goals is not None else None

    opponent = [t for t in teams if t != team_name]
    opponent = opponent[0] if opponent else "Opponent"

    # ── CREATE FIGURE ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(7, 1, figure=fig, hspace=0.38,
                           height_ratios=[1, 1, 1, 1, 1, 0.8, 0.8])

    colors = {
        'pred':   '#e74c3c',
        'actual': '#3498db',
        'goal':   '#2ecc71',
        'opp_goal': '#e74c3c',
    }

    # ── 5 REGRESSION PANELS ───────────────────────────────────────────
    for panel_idx in range(5):
        ax = fig.add_subplot(gs[panel_idx])

        actual_vals = [r['actual_reg'][panel_idx] for r in results]
        pred_vals   = [r['pred_reg'][panel_idx] for r in results]

        ax.plot(times, actual_vals, color=colors['actual'],
                linewidth=1.5, alpha=0.8, label='Actual')
        ax.plot(times, pred_vals, color=colors['pred'],
                linewidth=1.5, alpha=0.8, linestyle='--', label='Predicted')

        # Fill the gap between predicted and actual
        ax.fill_between(times, actual_vals, pred_vals,
                        alpha=0.1, color=colors['pred'])

        # Goal markers
        if team_goals is not None:
            for _, g in team_goals.iterrows():
                ax.axvline(x=g['time_min'], color=colors['goal'],
                          linestyle='--', alpha=0.6, linewidth=1.5)
        if opp_goals is not None:
            for _, g in opp_goals.iterrows():
                ax.axvline(x=g['time_min'], color=colors['opp_goal'],
                          linestyle=':', alpha=0.4, linewidth=1)

        # Halftime marker
        ax.axvline(x=45, color='gray', linestyle=':', alpha=0.4)

        # Compute per-metric R²
        r2 = compute_r2(actual_vals, pred_vals)
        mae = np.mean(np.abs(np.array(actual_vals) - np.array(pred_vals)))

        ax.set_ylabel(f'{REG_UNITS[panel_idx]}', fontsize=9)
        ax.set_title(f'{REG_NAMES[panel_idx]}  —  R²={r2:.3f}  |  MAE={mae:.3f}',
                     fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        if panel_idx < 4:
            ax.set_xticklabels([])

    # ── 2 CLASSIFICATION PANELS ───────────────────────────────────────
    cls_configs = [
        (5, 'pred_def', 'actual_def', DEF_CLASSES, 'Defensive posture'),
        (6, 'pred_off', 'actual_off', OFF_CLASSES, 'Offensive style'),
    ]

    for panel_idx, pred_key, actual_key, class_names, title in cls_configs:
        ax = fig.add_subplot(gs[panel_idx])

        actual_cls = [r[actual_key] for r in results]
        pred_cls   = [r[pred_key] for r in results]

        ax.step(times, actual_cls, color=colors['actual'],
                linewidth=1.5, alpha=0.8, where='post', label='Actual')
        ax.step(times, pred_cls, color=colors['pred'],
                linewidth=1.5, alpha=0.8, linestyle='--', where='post',
                label='Predicted')

        # Shade correct predictions in green
        for j in range(len(times) - 1):
            if actual_cls[j] == pred_cls[j]:
                ax.axvspan(times[j], times[j + 1], alpha=0.08,
                          color=colors['goal'], linewidth=0)

        # Goal markers
        if team_goals is not None:
            for _, g in team_goals.iterrows():
                ax.axvline(x=g['time_min'], color=colors['goal'],
                          linestyle='--', alpha=0.6, linewidth=1.5)
        if opp_goals is not None:
            for _, g in opp_goals.iterrows():
                ax.axvline(x=g['time_min'], color=colors['opp_goal'],
                          linestyle=':', alpha=0.4, linewidth=1)

        ax.axvline(x=45, color='gray', linestyle=':', alpha=0.4)

        # Y-axis: class names
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_ylim(-0.3, 2.3)

        # Accuracy
        acc = np.mean(np.array(actual_cls) == np.array(pred_cls))
        ax.set_title(f'{title}  —  Accuracy={acc:.1%}',
                     fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_xlabel('Match time (minutes)', fontsize=10)

    # ── TITLE AND SAVE ────────────────────────────────────────────────
    fig.suptitle(
        f'MATCH PREDICTOR — {team_name} vs {opponent}  |  Match {match_id}\n'
        f'Rolling next-window prediction (seq_len={SEQ_LEN})',
        fontsize=15, fontweight='bold', y=0.995
    )

    filename = f'match_prediction_{team_name.replace(" ", "_")}_{match_id}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"   Saved {filename}")
    plt.close()


def print_team_report(results, team_name):
    """Phase 4: Console report with per-metric R², MAE, and accuracy."""
    if not results:
        print(f"\n  {team_name}: No predictions (not enough windows)")
        return

    n = len(results)

    print(f"\n{'─'*60}")
    print(f"  {team_name}  ({n} predictions)")
    print(f"{'─'*60}")

    # Regression metrics
    print(f"\n  {'Metric':<18} | {'R²':>8} | {'MAE':>10} | {'Quality'}")
    print(f"  {'-'*55}")

    for i, name in enumerate(REG_NAMES):
        actual = [r['actual_reg'][i] for r in results]
        pred   = [r['pred_reg'][i] for r in results]

        r2  = compute_r2(actual, pred)
        mae = np.mean(np.abs(np.array(actual) - np.array(pred)))

        if r2 > 0.6:     quality = "Good"
        elif r2 > 0.35:  quality = "Moderate"
        elif r2 > 0.1:   quality = "Weak"
        else:             quality = "Poor"

        print(f"  {name:<18} | {r2:>8.3f} | {mae:>10.4f} | {quality}")

    # Classification metrics
    print(f"\n  {'Classifier':<18} | {'Accuracy':>8} | {'Per-class recall'}")
    print(f"  {'-'*55}")

    for pred_key, actual_key, class_names, title in [
        ('pred_def', 'actual_def', DEF_CLASSES, 'Def posture'),
        ('pred_off', 'actual_off', OFF_CLASSES, 'Off style'),
    ]:
        actual = np.array([r[actual_key] for r in results])
        pred   = np.array([r[pred_key] for r in results])
        acc    = np.mean(actual == pred)

        recalls = []
        for c, name in enumerate(class_names):
            mask = actual == c
            if mask.sum() > 0:
                recall = np.mean(pred[mask] == c)
                recalls.append(f"{name[:3]}:{recall:.0%}")
            else:
                recalls.append(f"{name[:3]}:n/a")

        print(f"  {title:<18} | {acc:>7.1%} | {', '.join(recalls)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Match Predictor — Rolling Prediction')
    parser.add_argument('--match_id', type=int, default=8658,
                        help='StatsBomb Match ID (default: 8658 = WC 2018 Final)')
    parser.add_argument('--team', type=str, default=None,
                        help='Single team to analyze (default: both teams)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"   MATCH PREDICTOR — Rolling Next-Window Prediction")
    print(f"   Match {args.match_id}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    # Phase 1: Prepare data
    team_graphs, goals = prepare_match(args.match_id)
    if team_graphs is None:
        return

    teams = sorted(team_graphs.keys())

    # Load model
    first_team = teams[0]
    first_graph = team_graphs[first_team][0]
    print(f"\n4. Loading model from {MODEL_PATH}...")
    model = load_model(first_graph)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Phase 2 + 3 + 4: Run prediction, visualize, report
    target_teams = teams
    if args.team:
        matching = [t for t in teams if args.team.lower() in t.lower()]
        if matching:
            target_teams = matching
        else:
            print(f"\n   Team '{args.team}' not found. Available: {teams}")
            return

    print(f"\n5. Running rolling predictions...")

    all_results = {}
    for team in target_teams:
        print(f"\n   Processing {team}...")
        graphs = team_graphs[team]
        print(f"   {len(graphs)} windows available, "
              f"seq_len={SEQ_LEN} → predicting from window {SEQ_LEN} onward")

        results = run_rolling_prediction(model, graphs, team)
        all_results[team] = results
        print(f"   {len(results)} valid predictions "
              f"(halftime-crossing sequences skipped)")

    # Visualization
    print(f"\n6. Generating dashboards...")
    for team in target_teams:
        plot_team_dashboard(all_results[team], team, goals, teams, args.match_id)

    # Console report
    print(f"\n{'='*60}")
    print(f"   MATCH REPORT — {' vs '.join(teams)}")
    print(f"{'='*60}")

    for team in target_teams:
        print_team_report(all_results[team], team)

    # Cross-team comparison
    if len(target_teams) == 2:
        print(f"\n{'='*60}")
        print(f"   COMPARISON")
        print(f"{'='*60}")

        for i, name in enumerate(REG_NAMES):
            r2s = []
            for team in target_teams:
                actual = [r['actual_reg'][i] for r in all_results[team]]
                pred   = [r['pred_reg'][i] for r in all_results[team]]
                r2s.append(compute_r2(actual, pred))

            print(f"  {name:<18} | "
                  f"{target_teams[0][:12]:<12} R²={r2s[0]:.3f} | "
                  f"{target_teams[1][:12]:<12} R²={r2s[1]:.3f}")

    print(f"\n{'='*60}")
    print(f"Done!")


if __name__ == "__main__":
    main()