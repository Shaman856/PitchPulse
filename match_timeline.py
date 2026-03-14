"""
MATCH TIMELINE VISUALIZATION
==============================
Runs the complete Tactical Suite on every window of a match and produces
a presentation-ready dashboard showing all predictions evolving over time.

Key panels:
1. Win/Draw/Loss probability curves with goal markers
2. Regression metrics (xT, Press Height, Field Tilt, Verticality, Tempo)
3. Classification predictions (Def Posture, Off Style)
4. Key player attention heatmap
5. Score timeline

Usage:
    python match_timeline.py                      # Default: WC 2018 Final
    python match_timeline.py --match_id 3869685   # Any match
    python match_timeline.py --team "France"       # Single team focus
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
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

DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
OFF_CLASSES = ['Patient', 'Balanced', 'Counter']
OUT_CLASSES = ['Loss', 'Draw', 'Win']

# Colors
TEAM_COLORS = {}  # Assigned dynamically


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


def run_full_match(match_id):
    """Process every window and extract all predictions."""
    
    print(f"1. Fetching match {match_id}...")
    raw = fetch_match_data(match_id)
    
    if raw['passes'].empty:
        print("ERROR: No data found.")
        return None, None, None
    
    raw['passes'] = encode_features(raw['passes'])
    if not raw['carries'].empty:
        raw['carries'] = encode_action_features(raw['carries'], 'carry')
    if not raw['dribbles'].empty:
        raw['dribbles'] = encode_action_features(raw['dribbles'], 'dribble')
    if not raw['defense'].empty:
        raw['defense'] = encode_action_features(raw['defense'], 'defense')
    
    print("2. Building windows...")
    windows = get_rolling_windows(raw, match_id, window_size=5, stride=1)
    
    # Build graphs with metadata
    graphs = []
    for w in windows:
        g = build_graph_from_window(w)
        g._team = w['team_name']
        g._start = w['start_time']
        g._end = w['end_time']
        g._mid = (w['start_time'] + w['end_time']) / 2.0
        g._score_diff = w['score_diff']
        graphs.append(g)
    
    print(f"   Built {len(graphs)} graphs")
    
    # Load model
    print("3. Loading model...")
    model = load_model(graphs[0])
    
    # Extract goals timeline
    shots = raw['shots']
    goals = shots[shots['is_goal'] == True][['minute', 'second', 'player', 'team']].copy()
    goals['time_min'] = goals['minute'] + goals['second'] / 60.0
    
    print(f"   Goals in match: {len(goals)}")
    for _, g_row in goals.iterrows():
        print(f"      {g_row['minute']}' - {g_row['player']} ({g_row['team']})")
    
    # Run inference on every window
    print("4. Running inference on all windows...")
    results = []
    
    with torch.no_grad():
        for g in graphs:
            g_dev = g.to(DEVICE)
            reg_out, cls_def_out, cls_off_out, cls_out_out = model(g_dev)
            
            # Outcome probabilities via softmax
            outcome_probs = F.softmax(cls_out_out, dim=1)[0].cpu().numpy()
            
            # Denormalize regression
            reg = reg_out[0].cpu().numpy()
            cum_xt = np.expm1(reg[0])
            press_height = reg[1] * 120.0
            field_tilt = reg[2]
            verticality = reg[3] * 2.0 - 1.0
            tempo = reg[4] * 30.0
            
            # Classification predictions
            def_pred = cls_def_out.argmax(1).item()
            off_pred = cls_off_out.argmax(1).item()
            
            # Def/Off class probabilities
            def_probs = F.softmax(cls_def_out, dim=1)[0].cpu().numpy()
            off_probs = F.softmax(cls_off_out, dim=1)[0].cpu().numpy()
            
            # Attention for key player
            try:
                _, _, attn2_idx, attn2_w, _ = model.extract_attention_weights(g_dev)
                edge_index = attn2_idx.cpu().numpy()
                attn_weights = attn2_w.cpu().numpy().flatten()
                
                node_attn = np.zeros(NUM_NODES)
                for e in range(edge_index.shape[1]):
                    src, dst = edge_index[0, e], edge_index[1, e]
                    w = attn_weights[e]
                    if src < NUM_NODES and dst < NUM_NODES:
                        node_attn[src] += w
                        node_attn[dst] += w
                
                node_active = g.x[:, 0].numpy()
                node_attn *= (0.5 + 0.5 * node_active)
                total = node_attn.sum()
                if total > 0:
                    node_attn /= total
            except Exception:
                node_attn = np.ones(NUM_NODES) / NUM_NODES
            
            results.append({
                'team': g._team,
                'time_mid': g._mid,
                'start': g._start,
                'end': g._end,
                'score_diff': g._score_diff,
                # Regression
                'cum_xt': cum_xt,
                'press_height': press_height,
                'field_tilt': field_tilt,
                'verticality': verticality,
                'tempo': tempo,
                # Classification
                'def_pred': def_pred,
                'off_pred': off_pred,
                'def_probs': def_probs,
                'off_probs': off_probs,
                # Outcome
                'outcome_probs': outcome_probs,  # [Loss, Draw, Win]
                # Key player
                'node_attn': node_attn,
            })
    
    teams = sorted(set(r['team'] for r in results))
    
    return results, goals, teams


def plot_team_timeline(results, goals, team, teams, match_id, save_suffix=""):
    """Generate the full timeline dashboard for one team."""
    
    team_results = sorted(
        [r for r in results if r['team'] == team],
        key=lambda r: r['time_mid']
    )
    
    if not team_results:
        print(f"No data for {team}")
        return
    
    times = [r['time_mid'] for r in team_results]
    
    # Determine opponent
    opponent = [t for t in teams if t != team][0] if len(teams) > 1 else "Opponent"
    
    # Goal times for markers
    team_goals = goals[goals['team'] == team]
    opp_goals = goals[goals['team'] != team]
    
    # --- CREATE FIGURE ---
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25,
                           height_ratios=[1.2, 1, 1, 1, 1, 1.5])
    
    # Color scheme
    win_color = '#2ecc71'
    draw_color = '#f39c12'
    loss_color = '#e74c3c'
    team_color = '#3498db'
    
    # ===================================================================
    # PANEL 1: OUTCOME PROBABILITY (Full width, top)
    # ===================================================================
    ax_out = fig.add_subplot(gs[0, :])
    
    p_loss = [r['outcome_probs'][0] for r in team_results]
    p_draw = [r['outcome_probs'][1] for r in team_results]
    p_win = [r['outcome_probs'][2] for r in team_results]
    
    ax_out.fill_between(times, 0, p_win, alpha=0.3, color=win_color, label='Win')
    ax_out.fill_between(times, p_win, [w+d for w,d in zip(p_win, p_draw)], 
                        alpha=0.3, color=draw_color, label='Draw')
    ax_out.fill_between(times, [w+d for w,d in zip(p_win, p_draw)], 1, 
                        alpha=0.3, color=loss_color, label='Loss')
    
    ax_out.plot(times, p_win, color=win_color, linewidth=2)
    ax_out.plot(times, p_draw, color=draw_color, linewidth=2)
    ax_out.plot(times, p_loss, color=loss_color, linewidth=2)
    
    # Goal markers
    for _, g_row in team_goals.iterrows():
        ax_out.axvline(x=g_row['time_min'], color=win_color, linestyle='--', alpha=0.8, linewidth=2)
        ax_out.text(g_row['time_min'] + 0.5, 0.95, f"⚽ {g_row['player']}", 
                   fontsize=8, color=win_color, fontweight='bold', va='top')
    
    for _, g_row in opp_goals.iterrows():
        ax_out.axvline(x=g_row['time_min'], color=loss_color, linestyle='--', alpha=0.8, linewidth=2)
        ax_out.text(g_row['time_min'] + 0.5, 0.05, f"⚽ {g_row['player']}", 
                   fontsize=8, color=loss_color, va='bottom')
    
    # Half-time line
    ax_out.axvline(x=45, color='gray', linestyle=':', alpha=0.5)
    ax_out.text(45.5, 0.5, 'HT', fontsize=9, color='gray', va='center')
    
    ax_out.set_xlim(times[0], times[-1])
    ax_out.set_ylim(0, 1)
    ax_out.set_ylabel('Probability')
    ax_out.set_title(f'Match Outcome Probability — {team}', fontsize=13, fontweight='bold')
    ax_out.legend(loc='upper right', fontsize=9)
    ax_out.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 2: CUMULATIVE xT + FIELD TILT
    # ===================================================================
    ax_xt = fig.add_subplot(gs[1, 0])
    ax_xt.plot(times, [r['cum_xt'] for r in team_results], color=team_color, linewidth=1.5)
    ax_xt.fill_between(times, [r['cum_xt'] for r in team_results], alpha=0.2, color=team_color)
    for _, g_row in team_goals.iterrows():
        ax_xt.axvline(x=g_row['time_min'], color=win_color, linestyle='--', alpha=0.5)
    for _, g_row in opp_goals.iterrows():
        ax_xt.axvline(x=g_row['time_min'], color=loss_color, linestyle='--', alpha=0.5)
    ax_xt.set_ylabel('Cumulative xT')
    ax_xt.set_title('Offensive Threat', fontweight='bold')
    ax_xt.grid(True, alpha=0.2)
    
    ax_ft = fig.add_subplot(gs[1, 1])
    ax_ft.plot(times, [r['field_tilt'] for r in team_results], color='#9b59b6', linewidth=1.5)
    ax_ft.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax_ft.fill_between(times, 0.5, [r['field_tilt'] for r in team_results], 
                       where=[r['field_tilt'] > 0.5 for r in team_results],
                       alpha=0.2, color='#9b59b6', label='Dominant')
    ax_ft.fill_between(times, 0.5, [r['field_tilt'] for r in team_results],
                       where=[r['field_tilt'] <= 0.5 for r in team_results],
                       alpha=0.2, color='#e67e22', label='Opponent dominant')
    ax_ft.set_ylabel('Field Tilt')
    ax_ft.set_title('Territorial Dominance', fontweight='bold')
    ax_ft.legend(fontsize=8, loc='upper right')
    ax_ft.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 3: PRESS HEIGHT + TEMPO
    # ===================================================================
    ax_ph = fig.add_subplot(gs[2, 0])
    ax_ph.plot(times, [r['press_height'] for r in team_results], color='#e74c3c', linewidth=1.5)
    ax_ph.axhline(y=65, color='orange', linestyle=':', alpha=0.5, label='High Press line')
    ax_ph.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Mid Block line')
    ax_ph.set_ylabel('Opponent Press Height')
    ax_ph.set_title('Defensive Pressure Faced', fontweight='bold')
    ax_ph.legend(fontsize=8)
    ax_ph.grid(True, alpha=0.2)
    
    ax_tempo = fig.add_subplot(gs[2, 1])
    ax_tempo.plot(times, [r['tempo'] for r in team_results], color='#27ae60', linewidth=1.5)
    ax_tempo.fill_between(times, [r['tempo'] for r in team_results], alpha=0.15, color='#27ae60')
    ax_tempo.set_ylabel('Passes/min')
    ax_tempo.set_title('Tempo', fontweight='bold')
    ax_tempo.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 4: DEFENSIVE POSTURE (stacked probabilities)
    # ===================================================================
    ax_def = fig.add_subplot(gs[3, 0])
    p_lb = [r['def_probs'][0] for r in team_results]
    p_mb = [r['def_probs'][1] for r in team_results]
    p_hp = [r['def_probs'][2] for r in team_results]
    
    ax_def.fill_between(times, 0, p_lb, alpha=0.5, color='#3498db', label='Low Block')
    ax_def.fill_between(times, p_lb, [a+b for a,b in zip(p_lb, p_mb)], 
                       alpha=0.5, color='#f39c12', label='Mid Block')
    ax_def.fill_between(times, [a+b for a,b in zip(p_lb, p_mb)], 1, 
                       alpha=0.5, color='#e74c3c', label='High Press')
    ax_def.set_ylim(0, 1)
    ax_def.set_ylabel('Probability')
    ax_def.set_title('Opponent Defensive Posture', fontweight='bold')
    ax_def.legend(fontsize=8, loc='upper right')
    ax_def.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 5: OFFENSIVE STYLE (stacked probabilities)
    # ===================================================================
    ax_off = fig.add_subplot(gs[3, 1])
    p_pat = [r['off_probs'][0] for r in team_results]
    p_bal = [r['off_probs'][1] for r in team_results]
    p_cnt = [r['off_probs'][2] for r in team_results]
    
    ax_off.fill_between(times, 0, p_pat, alpha=0.5, color='#2ecc71', label='Patient')
    ax_off.fill_between(times, p_pat, [a+b for a,b in zip(p_pat, p_bal)], 
                       alpha=0.5, color='#f39c12', label='Balanced')
    ax_off.fill_between(times, [a+b for a,b in zip(p_pat, p_bal)], 1, 
                       alpha=0.5, color='#e74c3c', label='Counter')
    ax_off.set_ylim(0, 1)
    ax_off.set_ylabel('Probability')
    ax_off.set_title('Offensive Style', fontweight='bold')
    ax_off.legend(fontsize=8, loc='upper right')
    ax_off.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 6: VERTICALITY + SCORE DIFF
    # ===================================================================
    ax_vert = fig.add_subplot(gs[4, 0])
    ax_vert.plot(times, [r['verticality'] for r in team_results], color='#8e44ad', linewidth=1.5)
    ax_vert.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_vert.set_ylabel('Verticality')
    ax_vert.set_title('Passing Direction', fontweight='bold')
    ax_vert.grid(True, alpha=0.2)
    
    ax_score = fig.add_subplot(gs[4, 1])
    score_diffs = [r['score_diff'] for r in team_results]
    ax_score.step(times, score_diffs, color=team_color, linewidth=2, where='post')
    ax_score.fill_between(times, 0, score_diffs, step='post',
                         where=[s > 0 for s in score_diffs], alpha=0.2, color=win_color)
    ax_score.fill_between(times, 0, score_diffs, step='post',
                         where=[s < 0 for s in score_diffs], alpha=0.2, color=loss_color)
    ax_score.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax_score.set_ylabel('Goal Difference')
    ax_score.set_title('Score State (model input)', fontweight='bold')
    ax_score.grid(True, alpha=0.2)
    
    # ===================================================================
    # PANEL 7: KEY PLAYER HEATMAP (Full width, bottom)
    # ===================================================================
    ax_kp = fig.add_subplot(gs[5, :])
    
    role_order = ['GK', 'LB', 'CB_L', 'CB_R', 'RB', 'DM', 'CM_L', 'CM_R', 'AM', 'LW', 'RW', 'ST']
    matrix = np.zeros((12, len(team_results)))
    
    for j, r in enumerate(team_results):
        for i, role in enumerate(role_order):
            role_idx = list(ROLE_NAMES.values()).index(role)
            matrix[i, j] = r['node_attn'][role_idx]
    
    im = ax_kp.imshow(matrix, aspect='auto', cmap='YlOrRd',
                      extent=[times[0], times[-1], 11.5, -0.5],
                      interpolation='nearest')
    
    # Goal markers on heatmap
    for _, g_row in team_goals.iterrows():
        ax_kp.axvline(x=g_row['time_min'], color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    for _, g_row in opp_goals.iterrows():
        ax_kp.axvline(x=g_row['time_min'], color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax_kp.set_yticks(range(12))
    ax_kp.set_yticklabels(role_order, fontsize=9)
    ax_kp.set_xlabel('Match Time (minutes)', fontsize=11)
    ax_kp.set_title(f'Key Player Identification (Attention-Based)', fontweight='bold')
    fig.colorbar(im, ax=ax_kp, fraction=0.015, pad=0.01, label='Importance')
    
    # Set x-labels on bottom panels
    for ax in [ax_vert, ax_score, ax_kp]:
        ax.set_xlabel('Match Time (minutes)')
    
    # --- MAIN TITLE ---
    fig.suptitle(
        f'TACTICAL SUITE — {team} vs {opponent}  |  Match {match_id}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    filename = f'match_timeline_{team.replace(" ", "_")}{save_suffix}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"   Saved {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Match Timeline Visualization')
    parser.add_argument('--match_id', type=int, default=8658,
                       help='StatsBomb Match ID (default: 8658 = WC 2018 Final)')
    parser.add_argument('--team', type=str, default=None,
                       help='Specific team to visualize (default: both teams)')
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"   MATCH TIMELINE VISUALIZATION — Match {args.match_id}")
    print(f"{'='*60}")
    
    results, goals, teams = run_full_match(args.match_id)
    
    if results is None:
        return
    
    print(f"\n5. Generating visualizations...")
    
    if args.team:
        matching = [t for t in teams if args.team.lower() in t.lower()]
        if matching:
            plot_team_timeline(results, goals, matching[0], teams, args.match_id)
        else:
            print(f"Team '{args.team}' not found. Available: {teams}")
    else:
        for team in teams:
            plot_team_timeline(results, goals, team, teams, args.match_id)
    
    # --- COMPARATIVE SUMMARY ---
    print(f"\n{'='*60}")
    print(f"   MATCH NARRATIVE")
    print(f"{'='*60}")
    
    for team in teams:
        team_r = [r for r in results if r['team'] == team]
        avg_xt = np.mean([r['cum_xt'] for r in team_r])
        avg_tilt = np.mean([r['field_tilt'] for r in team_r])
        avg_tempo = np.mean([r['tempo'] for r in team_r])
        avg_vert = np.mean([r['verticality'] for r in team_r])
        avg_win_prob = np.mean([r['outcome_probs'][2] for r in team_r])
        
        # Dominant style
        off_preds = [r['off_pred'] for r in team_r]
        dominant_style = OFF_CLASSES[max(set(off_preds), key=off_preds.count)]
        
        def_preds = [r['def_pred'] for r in team_r]
        # This is opponent's posture against this team
        
        print(f"\n{team}:")
        print(f"  Avg xT: {avg_xt:.3f} | Avg Field Tilt: {avg_tilt:.1%} | Avg Tempo: {avg_tempo:.1f} p/min")
        print(f"  Avg Verticality: {avg_vert:+.3f} | Dominant Style: {dominant_style}")
        print(f"  Avg Win Probability: {avg_win_prob:.1%}")
    
    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
