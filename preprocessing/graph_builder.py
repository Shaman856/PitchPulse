import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data

# --- CONFIGURATION ---
NUM_NODES = 12 
DEFAULT_POSITIONS = {
    0: [0.05, 0.50], # GK
    1: [0.25, 0.10], # LB
    2: [0.20, 0.35], # CB_L
    3: [0.20, 0.65], # CB_R
    4: [0.25, 0.90], # RB
    5: [0.40, 0.50], # DM
    6: [0.55, 0.30], # CM_L
    7: [0.55, 0.70], # CM_R
    8: [0.70, 0.50], # AM
    9: [0.75, 0.15], # LW
    10:[0.75, 0.85], # RW
    11:[0.85, 0.50]  # ST
}

XT_GRID = np.array([
    [0.0030, 0.0036, 0.0042, 0.0045, 0.0048, 0.0052, 0.0064, 0.0082, 0.0117, 0.0191, 0.0370, 0.0790],
    [0.0023, 0.0030, 0.0036, 0.0041, 0.0046, 0.0053, 0.0070, 0.0098, 0.0153, 0.0268, 0.0530, 0.1060],
    [0.0020, 0.0028, 0.0035, 0.0041, 0.0049, 0.0060, 0.0082, 0.0117, 0.0189, 0.0345, 0.0710, 0.1630],
    [0.0018, 0.0026, 0.0034, 0.0042, 0.0052, 0.0064, 0.0088, 0.0126, 0.0209, 0.0390, 0.0810, 0.2100],
    [0.0018, 0.0026, 0.0034, 0.0042, 0.0052, 0.0064, 0.0088, 0.0126, 0.0209, 0.0390, 0.0810, 0.2100],
    [0.0020, 0.0028, 0.0035, 0.0041, 0.0049, 0.0060, 0.0082, 0.0117, 0.0189, 0.0345, 0.0710, 0.1630],
    [0.0023, 0.0030, 0.0036, 0.0041, 0.0046, 0.0053, 0.0070, 0.0098, 0.0153, 0.0268, 0.0530, 0.1060],
    [0.0030, 0.0036, 0.0042, 0.0045, 0.0048, 0.0052, 0.0064, 0.0082, 0.0117, 0.0191, 0.0370, 0.0790],
])

def _get_xt_value(x, y):
    col = int(np.clip(x / 10.0, 0, 11))
    row = int(np.clip(y / 10.0, 0, 7))
    return XT_GRID[row, col]

ACTION_TYPE_MAP = {
    'pass': 0,
    'carry': 1,
    'dribble': 2,
    'defense': 3,
}


def _build_unified_actions(window):
    """
    Merges passes, carries, dribbles, and defensive actions
    into a single sorted action stream for multi-event edge construction.
    """
    frames = []
    
    # --- Passes ---
    passes = window['passes']
    if not passes.empty and 'node_idx' in passes.columns:
        p = passes[['time_min', 'node_idx', 'x', 'y', 'end_x', 'end_y',
                     'pressure_code', 'height_code', 'pattern_code',
                     'pass_length', 'pass_angle']].copy()
        p['action_type'] = 'pass'
        if 'possession' in passes.columns:
            p['possession'] = passes['possession'].values
        else:
            p['possession'] = -1
        frames.append(p)
    
    # --- Carries ---
    carries = window.get('carries', pd.DataFrame())
    if not carries.empty and 'node_idx' in carries.columns:
        c = pd.DataFrame({
            'time_min': carries['time_min'],
            'node_idx': carries['node_idx'],
            'x': carries['x'], 'y': carries['y'],
            'end_x': carries['end_x'], 'end_y': carries['end_y'],
            'pressure_code': carries['pressure_code'] if 'pressure_code' in carries.columns else 0,
            'height_code': 0,
            'pattern_code': carries['pattern_code'] if 'pattern_code' in carries.columns else 0,
            'pass_length': carries['pass_length'] if 'pass_length' in carries.columns else 0.0,
            'pass_angle': carries['pass_angle'] if 'pass_angle' in carries.columns else 0.0,
            'action_type': 'carry',
            'possession': carries['possession'] if 'possession' in carries.columns else -1,
        })
        frames.append(c)
    
    # --- Dribbles ---
    dribbles = window.get('dribbles', pd.DataFrame())
    if not dribbles.empty and 'node_idx' in dribbles.columns:
        d = pd.DataFrame({
            'time_min': dribbles['time_min'],
            'node_idx': dribbles['node_idx'],
            'x': dribbles['x'], 'y': dribbles['y'],
            'end_x': dribbles['end_x'], 'end_y': dribbles['end_y'],
            'pressure_code': dribbles['pressure_code'] if 'pressure_code' in dribbles.columns else 0,
            'height_code': 0,
            'pattern_code': dribbles['pattern_code'] if 'pattern_code' in dribbles.columns else 0,
            'pass_length': dribbles['pass_length'] if 'pass_length' in dribbles.columns else 0.0,
            'pass_angle': dribbles['pass_angle'] if 'pass_angle' in dribbles.columns else 0.0,
            'action_type': 'dribble',
            'possession': dribbles['possession'] if 'possession' in dribbles.columns else -1,
        })
        frames.append(d)
    
    # --- Defensive Actions ---
    team_defense = window.get('team_defense', pd.DataFrame())
    if not team_defense.empty and 'node_idx' in team_defense.columns:
        td = pd.DataFrame({
            'time_min': team_defense['time_min'],
            'node_idx': team_defense['node_idx'],
            'x': team_defense['x'], 'y': team_defense['y'],
            'end_x': team_defense['x'], 'end_y': team_defense['y'],
            'pressure_code': team_defense['pressure_code'] if 'pressure_code' in team_defense.columns else 0,
            'height_code': 0, 'pattern_code': 0,
            'pass_length': 0.0, 'pass_angle': 0.0,
            'action_type': 'defense',
            'possession': -2,
        })
        frames.append(td)
    
    if not frames:
        return pd.DataFrame()
    
    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values('time_min').reset_index(drop=True)
    unified['node_idx'] = unified['node_idx'].fillna(7).astype(int).clip(0, NUM_NODES - 1)
    
    return unified


def _compute_centrality(edge_sources, edge_targets):
    """
    TIER 2 NEW: Computes degree centrality and betweenness centrality
    for each of the 12 role-nodes from the constructed edge list.
    
    Uses NetworkX on a small 12-node graph — computationally trivial.
    
    Returns:
        degree_centrality: np.array of shape [12]
        betweenness_centrality: np.array of shape [12]
    """
    degree_cent = np.zeros(NUM_NODES)
    between_cent = np.zeros(NUM_NODES)
    
    if len(edge_sources) == 0:
        return degree_cent, between_cent
    
    # Build a weighted directed graph (weight = edge count between node pairs)
    G = nx.DiGraph()
    G.add_nodes_from(range(NUM_NODES))
    
    # Count edges between each pair for weighting
    edge_weights = {}
    for s, t in zip(edge_sources, edge_targets):
        key = (s, t)
        edge_weights[key] = edge_weights.get(key, 0) + 1
    
    for (s, t), w in edge_weights.items():
        if s != t:  # Skip self-loops for centrality (they inflate degree artificially)
            G.add_edge(s, t, weight=w)
    
    # Degree centrality (normalized by max possible = NUM_NODES - 1)
    dc = nx.degree_centrality(G)
    for node, val in dc.items():
        degree_cent[node] = val
    
    # Betweenness centrality (normalized)
    # On a 12-node graph this is instant
    try:
        bc = nx.betweenness_centrality(G, weight='weight', normalized=True)
        for node, val in bc.items():
            between_cent[node] = val
    except Exception:
        pass  # If graph is disconnected or degenerate, leave as zeros
    
    return degree_cent, between_cent


def build_graph_from_window(window):
    """
    Converts a Window Bundle into a PyTorch Geometric Data object.
    
    =================================================================
    TIER 2 CHANGES (on top of Tier 1):
    =================================================================
    
    1. CENTRALITY NODE FEATURES (10→12 dim):
       - [10] Degree centrality     - How connected this role is in the action graph
       - [11] Betweenness centrality - How much this role bridges between other roles
       Computed from the window's constructed edge list using NetworkX.
       Provides pre-computed structural summary that complements GAT attention.
    
    2. CUMULATIVE CHAIN xT EDGE FEATURE (8→9 dim):
       - [8] Cumulative chain xT    - Running total of positive xT gained in the
                                       current possession chain up to this action.
       Gives the model a sense of "how threatening is this possession so far"
       at each edge, rather than just the instantaneous ΔxT.
    
    Dimensions: Nodes [12, 12], Edges [E, 9], Global [1, 5]
    =================================================================
    
    Node Features (12-dim):
    [0]  Active flag           
    [1]  Avg X position        
    [2]  Avg Y position        
    [3]  Action share          
    [4]  Directional tendency  
    [5]  Pressure ratio        
    [6]  Avg action distance   
    [7]  High pass ratio       
    [8]  Receive share         
    [9]  Set piece ratio       
    [10] Degree centrality     - TIER 2 NEW
    [11] Betweenness centrality- TIER 2 NEW
    
    Edge Features (9-dim):
    [0] Action type        
    [1] Action distance    
    [2] Action angle       
    [3] Under pressure     
    [4] Height code        
    [5] Pattern code       
    [6] ΔxT                
    [7] Time gap           
    [8] Cumulative chain xT - TIER 2 NEW
    
    Global Context (5-dim): Unchanged from Tier 1
    """
    
    # 1. Unpack
    passes = window['passes'].copy()
    opp_def = window['opp_defense']
    
    unified = _build_unified_actions(window)
    
    total_team_actions = len(unified) if not unified.empty else 1
    total_team_passes = len(passes) if not passes.empty else 1
    
    # --- 2. NODE FEATURES (x) [12, 12] ---
    if not passes.empty and 'node_idx' not in passes.columns:
        raise ValueError("Critical: 'node_idx' missing. Run utils.encode_features() before slicing.")

    node_features = np.zeros((NUM_NODES, 12))  # TIER 2: 10 → 12

    for i in range(NUM_NODES):
        def_x, def_y = DEFAULT_POSITIONS[i]
        node_features[i, 0] = 0.0
        node_features[i, 1] = def_x
        node_features[i, 2] = def_y
    
    # Receive count lookup
    receive_counts = np.zeros(NUM_NODES)
    if not passes.empty and 'pass_recipient' in passes.columns:
        player_role_map = passes.groupby('player')['node_idx'].first().to_dict()
        for _, row in passes.iterrows():
            recipient = row.get('pass_recipient', None)
            if pd.notna(recipient) and recipient in player_role_map:
                recv_idx = int(player_role_map[recipient])
                if 0 <= recv_idx < NUM_NODES:
                    receive_counts[recv_idx] += 1
    
    # Aggregate per Tactical Role
    if not unified.empty:
        grouped = unified.groupby('node_idx')
        for node_idx, data in grouped:
            if 0 <= node_idx < NUM_NODES:
                node_features[node_idx, 0] = 1.0
                node_features[node_idx, 1] = data['x'].mean() / 120.0
                node_features[node_idx, 2] = data['y'].mean() / 80.0
                node_features[node_idx, 3] = len(data) / total_team_actions
                
                movement_data = data[data['action_type'] != 'defense']
                if not movement_data.empty:
                    fwd_mean = (movement_data['end_x'] - movement_data['x']).mean()
                    if fwd_mean < -5:
                        node_features[node_idx, 4] = 0.0
                    elif fwd_mean > 5:
                        node_features[node_idx, 4] = 1.0
                    else:
                        node_features[node_idx, 4] = 0.5
                else:
                    node_features[node_idx, 4] = 0.5
                
                if 'pressure_code' in data.columns:
                    node_features[node_idx, 5] = data['pressure_code'].mean()
                
                if 'pass_length' in data.columns:
                    node_features[node_idx, 6] = data['pass_length'].mean() / 120.0
                
                pass_data = data[data['action_type'] == 'pass']
                if not pass_data.empty and 'height_code' in pass_data.columns:
                    node_features[node_idx, 7] = (pass_data['height_code'] == 2).mean()
                
                node_features[node_idx, 8] = receive_counts[node_idx] / total_team_passes
                
                if 'pattern_code' in data.columns:
                    node_features[node_idx, 9] = (data['pattern_code'] == 1).mean()
    
    # Centrality features [10, 11] are filled AFTER edge construction (need the edge list)

    # --- 3. EDGE CONSTRUCTION [E, 9] ---
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    
    # TIER 2: Track cumulative xT per possession chain
    chain_cum_xt = {}  # possession_id → running total of positive ΔxT
    
    if not unified.empty:
        sorted_actions = unified.sort_values('time_min').reset_index(drop=True)
        
        node_indices = sorted_actions['node_idx'].values
        action_types = sorted_actions['action_type'].values
        x_coords = sorted_actions['x'].values
        y_coords = sorted_actions['y'].values
        end_x_coords = sorted_actions['end_x'].values
        end_y_coords = sorted_actions['end_y'].values
        
        p_len = sorted_actions['pass_length'].values if 'pass_length' in sorted_actions.columns else np.zeros(len(sorted_actions))
        p_ang = sorted_actions['pass_angle'].values if 'pass_angle' in sorted_actions.columns else np.zeros(len(sorted_actions))
        p_pres = sorted_actions['pressure_code'].values if 'pressure_code' in sorted_actions.columns else np.zeros(len(sorted_actions))
        p_height = sorted_actions['height_code'].values if 'height_code' in sorted_actions.columns else np.zeros(len(sorted_actions))
        p_pattern = sorted_actions['pattern_code'].values if 'pattern_code' in sorted_actions.columns else np.zeros(len(sorted_actions))
        time_mins = sorted_actions['time_min'].values
        poss_ids = sorted_actions['possession'].values if 'possession' in sorted_actions.columns else np.full(len(sorted_actions), -1)
        
        # Pre-compute ΔxT and cumulative chain xT for every action
        delta_xts = np.zeros(len(sorted_actions))
        cum_chain_xts = np.zeros(len(sorted_actions))
        
        for idx in range(len(sorted_actions)):
            sx, sy = x_coords[idx], y_coords[idx]
            ex, ey = end_x_coords[idx], end_y_coords[idx]
            
            if pd.notna(sx) and pd.notna(ex):
                delta_xts[idx] = _get_xt_value(ex, ey) - _get_xt_value(sx, sy)
            
            # Track cumulative xT per possession chain
            poss_id = poss_ids[idx]
            if poss_id >= 0:  # Valid possession
                if poss_id not in chain_cum_xt:
                    chain_cum_xt[poss_id] = 0.0
                if delta_xts[idx] > 0:
                    chain_cum_xt[poss_id] += delta_xts[idx]
                cum_chain_xts[idx] = chain_cum_xt[poss_id]
        
        def make_edge_attr(idx, time_gap_seconds):
            delta_xt_norm = np.clip(delta_xts[idx] / 0.2, -1.0, 1.0)
            
            at = action_types[idx]
            action_code = ACTION_TYPE_MAP.get(at, 0) / 3.0
            
            time_gap_norm = min(time_gap_seconds / 60.0, 1.0)
            
            # TIER 2: Cumulative chain xT — how threatening is this possession so far
            # Normalize by 0.5 (a very threatening chain might accumulate ~0.3-0.5 xT)
            cum_xt_norm = min(cum_chain_xts[idx] / 0.5, 1.0)
            
            return [
                action_code,                     # [0] Action type
                p_len[idx] / 120.0,              # [1] Distance
                p_ang[idx] / 3.14,               # [2] Angle
                float(p_pres[idx]),              # [3] Under pressure
                float(p_height[idx]) / 2.0,      # [4] Height code
                float(p_pattern[idx]) / 2.0,     # [5] Pattern code
                delta_xt_norm,                    # [6] ΔxT
                time_gap_norm,                    # [7] Time gap
                cum_xt_norm,                      # [8] Cumulative chain xT (TIER 2 NEW)
            ]
        
        for i in range(len(sorted_actions) - 1):
            src = node_indices[i]
            dst = node_indices[i + 1]
            
            if src >= NUM_NODES or dst >= NUM_NODES:
                continue
            
            time_gap = (time_mins[i + 1] - time_mins[i]) * 60.0
            
            current_action = action_types[i]
            next_action = action_types[i + 1]
            
            if current_action == 'defense':
                edge_sources.append(src)
                edge_targets.append(src)
                edge_attrs.append(make_edge_attr(i, 0.0))
                continue
            
            if poss_ids[i] == poss_ids[i + 1] and poss_ids[i] != -1:
                edge_sources.append(src)
                edge_targets.append(dst)
                edge_attrs.append(make_edge_attr(i, time_gap))
                continue
            
            if time_gap < 5.0 and next_action != 'defense':
                edge_sources.append(src)
                edge_targets.append(dst)
                edge_attrs.append(make_edge_attr(i, time_gap))
    
    # --- TIER 2: Compute centrality from constructed edges ---
    degree_cent, between_cent = _compute_centrality(edge_sources, edge_targets)
    
    for i in range(NUM_NODES):
        node_features[i, 10] = degree_cent[i]
        node_features[i, 11] = between_cent[i]
    
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    
    # Finalize edges
    if len(edge_sources) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # --- 4. GLOBAL CONTEXT (u) [1, 5] — unchanged from Tier 1 ---
    duration = window['end_time'] - window['start_time']
    
    opp_density = len(opp_def) / duration if duration > 0 else 0.0
    
    cp_count = window.get('opp_counterpress_count', 0)
    counterpress_intensity = cp_count / duration if duration > 0 else 0.0
    
    period = window.get('period', 1)
    half_indicator = 0.0 if period <= 1 else 1.0
    
    score_diff = window.get('score_diff', 0)
    score_diff_norm = np.clip(score_diff / 3.0, -1.0, 1.0)
    
    match_progress = window.get('match_progress', 0.5)
    
    u = torch.tensor([[
        opp_density,
        counterpress_intensity,
        half_indicator,
        score_diff_norm,
        match_progress,
    ]], dtype=torch.float)

    # --- 5. REGRESSION TARGETS ---
    y = torch.tensor([[
        np.log1p(window['y_cum_xt']),
        window['y_press_height'] / 120.0,
        window['y_field_tilt'],
        (window['y_verticality'] + 1.0) / 2.0,
        window['y_tempo'] / 30.0,
    ]], dtype=torch.float)
    
    # --- 6. CLASSIFICATION TARGETS (y_cls) [1, 3] ---
    # [0] Defensive Posture: 0=LowBlock, 1=MidBlock, 2=HighPress
    # [1] Offensive Style:   0=Patient, 1=Balanced, 2=Counter
    # [2] Match Outcome:     0=Loss, 1=Draw, 2=Win  (SUITE NEW)
    y_cls = torch.tensor([[
        window['y_def_posture'],
        window['y_off_style'],
        window.get('y_outcome', 1),  # Default to Draw if missing
    ]], dtype=torch.long)

    # --- 7. ASSEMBLE ---
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
    data.y_cls = y_cls
    data.match_id = window.get('match_id', 0)
    data.window_id = window['window_id']
    data.team_name = window['team_name']
    
    return data

# --- TEST BLOCK ---
if __name__ == "__main__":
    from window_slicer import get_rolling_windows
    from data_pipeline import fetch_match_data
    from utils import encode_features, encode_action_features
    
    match_id = 8658
    print(f"1. Fetching Match {match_id}...")
    raw = fetch_match_data(match_id)
    
    if not raw['passes'].empty:
        raw['passes'] = encode_features(raw['passes'])
    if not raw['carries'].empty:
        raw['carries'] = encode_action_features(raw['carries'], 'carry')
    if not raw['dribbles'].empty:
        raw['dribbles'] = encode_action_features(raw['dribbles'], 'dribble')
    if not raw['defense'].empty:
        raw['defense'] = encode_action_features(raw['defense'], 'defense')
    
    print("2. Slicing windows...")
    windows = get_rolling_windows(raw, match_id)
    
    print("3. Building Graphs...")
    graphs = []
    for w in windows:
        g = build_graph_from_window(w)
        graphs.append(g)
        
    print(f"Built {len(graphs)} graphs.")
    
    if len(graphs) > 10:
        g = graphs[10]
        print(f"\n--- Graph Inspection (Window 10) ---")
        print(f"Team: {g.team_name} | Match: {g.match_id}")
        print(f"Nodes (x): {g.x.shape} (Should be [12, 12])")
        print(f"Edges: {g.edge_index.shape[1]}")
        print(f"Edge attr: {g.edge_attr.shape} (Should be [E, 9])")
        print(f"Global (u): {g.u.shape} (Should be [1, 5])")
        
        print(f"\n--- Tier 2 Feature Check ---")
        print(f"Node[10] (degree cent):     {g.x[:, 10].tolist()}")
        print(f"Node[11] (betweenness cent):{g.x[:, 11].tolist()}")
        print(f"Edge[8] (cum chain xT):     mean={g.edge_attr[:, 8].mean():.4f}, max={g.edge_attr[:, 8].max():.4f}")
