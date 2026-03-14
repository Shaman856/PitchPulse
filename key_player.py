"""
KEY PLAYER IDENTIFICATION v2
==============================
Uses GAT attention weights instead of embedding magnitudes for sharper
player attribution. Attention directly measures which edges (interactions)
the model considers most important for its predictions.

Changes from v1:
- Attention-based importance (not embedding norms) 
- GK excluded from xT attribution (gets separate defensive importance score)
- Sharper differentiation between roles

Usage:
    python key_player.py
    python key_player.py --match_id 3869685
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings

from preprocessing.data_pipeline import fetch_match_data
from preprocessing.utils import encode_features, encode_action_features
from preprocessing.window_slicer import get_rolling_windows
from preprocessing.graph_builder import build_graph_from_window, NUM_NODES
from models.model import TacticalGAT

warnings.filterwarnings('ignore')

MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROLE_NAMES = {
    0: 'GK', 1: 'LB', 2: 'CB_L', 3: 'CB_R', 4: 'RB',
    5: 'DM', 6: 'CM_L', 7: 'CM_R', 8: 'AM',
    9: 'LW', 10: 'RW', 11: 'ST'
}

# Outfield roles for xT attribution (GK excluded)
OUTFIELD_ROLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def load_model(sample_graph):
    model = TacticalGAT(
        num_node_features=sample_graph.x.shape[1],
        num_reg_targets=5, num_def_classes=3, num_off_classes=3,
        num_outcome_classes=3,
        edge_dim=sample_graph.edge_attr.shape[1],
        global_dim=sample_graph.u.shape[1]
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def compute_attention_importance(model, graph):
    """
    SUITE v2: Attention-based player importance.
    
    For each node, computes:
    1. Incoming attention sum — how much other nodes attend TO this node
    2. Outgoing attention sum — how much this node's passes/actions are attended
    3. Combined importance = mean(incoming + outgoing)
    
    Uses layer 2 attention (final representation, most task-relevant).
    
    Returns:
        role_importance: dict {role_name: importance (0-1, sums to 1)}
        role_xt: dict {role_name: attributed xT}
        predicted_xt: float
    """
    graph = graph.to(DEVICE)
    
    with torch.no_grad():
        # Get attention weights
        attn1_idx, attn1_w, attn2_idx, attn2_w, node_emb = model.extract_attention_weights(graph)
        
        # Get predicted xT
        reg_out, _, _, _ = model(graph)
        predicted_xt = np.expm1(reg_out[0, 0].item())
    
    # Use layer 2 attention (most task-relevant, after full context)
    edge_index = attn2_idx.cpu().numpy()  # [2, E]
    attn_weights = attn2_w.cpu().numpy().flatten()  # [E]
    
    # Compute per-node attention scores
    incoming_attn = np.zeros(NUM_NODES)  # How much others attend TO this node
    outgoing_attn = np.zeros(NUM_NODES)  # How much this node's edges are weighted
    
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e]
        dst = edge_index[1, e]
        w = attn_weights[e]
        
        if src < NUM_NODES and dst < NUM_NODES:
            outgoing_attn[src] += w
            incoming_attn[dst] += w
    
    # Combined importance: average of incoming and outgoing
    combined = incoming_attn + outgoing_attn
    
    # Also factor in node activity (active nodes matter more than inactive)
    node_active = graph.x[:, 0].cpu().numpy()  # Active flag
    combined = combined * (0.5 + 0.5 * node_active)  # Inactive nodes get halved
    
    # Normalize to probability distribution
    total = combined.sum()
    if total > 0:
        importance = combined / total
    else:
        importance = np.ones(NUM_NODES) / NUM_NODES
    
    # xT attribution: only outfield players
    outfield_importance = np.zeros(NUM_NODES)
    for i in OUTFIELD_ROLES:
        outfield_importance[i] = importance[i]
    
    outfield_total = outfield_importance.sum()
    if outfield_total > 0:
        outfield_normed = outfield_importance / outfield_total
    else:
        outfield_normed = np.zeros(NUM_NODES)
    
    xt_per_role = outfield_normed * max(predicted_xt, 0)
    
    role_importance = {ROLE_NAMES[i]: float(importance[i]) for i in range(NUM_NODES)}
    role_xt = {ROLE_NAMES[i]: float(xt_per_role[i]) for i in range(NUM_NODES)}
    
    return role_importance, role_xt, predicted_xt


def analyze_match(match_id):
    print(f"{'='*60}")
    print(f"   KEY PLAYER IDENTIFICATION v2 — Match {match_id}")
    print(f"   (Attention-based, GK excluded from xT)")
    print(f"{'='*60}")
    
    # 1. Fetch and process
    print(f"\n1. Fetching match data...")
    raw = fetch_match_data(match_id)
    
    if raw['passes'].empty:
        print("ERROR: No pass data found.")
        return
    
    raw['passes'] = encode_features(raw['passes'])
    if not raw['carries'].empty:
        raw['carries'] = encode_action_features(raw['carries'], 'carry')
    if not raw['dribbles'].empty:
        raw['dribbles'] = encode_action_features(raw['dribbles'], 'dribble')
    if not raw['defense'].empty:
        raw['defense'] = encode_action_features(raw['defense'], 'defense')
    
    # 2. Build windows and graphs
    print("2. Building windows...")
    windows = get_rolling_windows(raw, match_id, window_size=5, stride=1)
    
    graphs = []
    for w in windows:
        g = build_graph_from_window(w)
        g._team_name_str = w['team_name']
        g._start_time = w['start_time']
        g._end_time = w['end_time']
        graphs.append(g)
    
    print(f"   Built {len(graphs)} graphs")
    
    # 3. Load model
    print("3. Loading model...")
    model = load_model(graphs[0])
    
    # 4. Compute attribution
    print("4. Computing attention-based attribution...")
    
    teams = sorted(set(g._team_name_str for g in graphs))
    results = []
    
    for g in graphs:
        importance, xt_attr, pred_xt = compute_attention_importance(model, g)
        results.append({
            'team': g._team_name_str,
            'start_time': g._start_time,
            'end_time': g._end_time,
            'predicted_xt': pred_xt,
            'importance': importance,
            'xt_attribution': xt_attr,
        })
    
    # 5. Match summary
    print(f"\n{'='*60}")
    print(f"   MATCH SUMMARY")
    print(f"{'='*60}")
    
    for team in teams:
        team_results = [r for r in results if r['team'] == team]
        
        total_xt_per_role = {}
        total_importance_per_role = {}
        for r in team_results:
            for role, xt in r['xt_attribution'].items():
                total_xt_per_role[role] = total_xt_per_role.get(role, 0) + xt
            for role, imp in r['importance'].items():
                total_importance_per_role[role] = total_importance_per_role.get(role, 0) + imp
        
        n_windows = len(team_results)
        avg_importance = {k: v/n_windows for k, v in total_importance_per_role.items()}
        
        # Sort by total xT (outfield only, GK will show 0 xT)
        sorted_roles = sorted(total_xt_per_role.items(), key=lambda x: x[1], reverse=True)
        
        total_team_xt = sum(v for k, v in total_xt_per_role.items() if k != 'GK')
        
        print(f"\n--- {team} ---")
        print(f"Total Predicted xT (outfield): {total_team_xt:.4f}")
        print(f"{'Role':<8} | {'Total xT':<10} | {'% of Team':<10} | {'Avg Attention':<15}")
        print("-" * 55)
        for role, xt in sorted_roles:
            if role == 'GK':
                pct = 0.0
                marker = " (excluded)"
            else:
                pct = (xt / total_team_xt * 100) if total_team_xt > 0 else 0
                marker = " ★" if pct > 11 else ""  # Above average = 100/11 ≈ 9.1%
            imp = avg_importance[role]
            print(f"{role:<8} | {xt:<10.4f} | {pct:<9.1f}% | {imp:<15.4f}{marker}")
    
    # 6. Timeline visualization
    print("\n5. Generating timeline visualization...")
    _plot_timeline(results, teams, match_id)
    
    # 7. Top moments
    print(f"\n{'='*60}")
    print(f"   TOP 5 ATTACKING MOMENTS")
    print(f"{'='*60}")
    
    sorted_results = sorted(results, key=lambda r: r['predicted_xt'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n#{i+1}: {r['team']} | {r['start_time']:.0f}-{r['end_time']:.0f} min | xT={r['predicted_xt']:.4f}")
        # Show top 3 outfield contributors
        outfield = {k: v for k, v in r['importance'].items() if k != 'GK'}
        top3 = sorted(outfield.items(), key=lambda x: x[1], reverse=True)[:3]
        for role, imp in top3:
            print(f"   {role}: {imp:.1%} attention ({r['xt_attribution'][role]:.4f} xT)")


def _plot_timeline(results, teams, match_id):
    """Heatmap of role importance over time, with attention-based values."""
    
    fig, axes = plt.subplots(len(teams), 1, figsize=(16, 5 * len(teams)))
    if len(teams) == 1:
        axes = [axes]
    
    role_order = ['GK', 'LB', 'CB_L', 'CB_R', 'RB', 'DM', 'CM_L', 'CM_R', 'AM', 'LW', 'RW', 'ST']
    
    for ax, team in zip(axes, teams):
        team_results = sorted(
            [r for r in results if r['team'] == team],
            key=lambda r: r['start_time']
        )
        
        if not team_results:
            continue
        
        times = [r['start_time'] for r in team_results]
        matrix = np.zeros((12, len(team_results)))
        
        for j, r in enumerate(team_results):
            for i, role in enumerate(role_order):
                matrix[i, j] = r['importance'].get(role, 0)
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', 
                       extent=[times[0], times[-1], 11.5, -0.5],
                       interpolation='nearest')
        
        ax.set_yticks(range(12))
        ax.set_yticklabels(role_order, fontsize=9)
        ax.set_xlabel('Match Time (minutes)')
        ax.set_title(f'{team} — Attention-Based Role Importance', fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Attention Score')
    
    plt.suptitle(f'Key Player Identification v2 (Attention) — Match {match_id}', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('key_player_analysis.png', bbox_inches='tight', dpi=150)
    print("   Saved key_player_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_id', type=int, default=8658)
    args = parser.parse_args()
    analyze_match(args.match_id)
