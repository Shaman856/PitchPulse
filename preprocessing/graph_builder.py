import torch
import pandas as pd
import numpy as np
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

def build_graph_from_window(window):
    """
    Converts a Window Bundle into a PyTorch Geometric Data object.
    
    Node Features (11-dim):
    [0]  Active flag           - 1.0 if the role touched the ball
    [1]  Avg X position        - Normalized mean X (0-1)
    [2]  Avg Y position        - Normalized mean Y (0-1)
    [3]  Pass volume           - Passes made by this role (normalized)
    [4]  Forward progression   - Avg forward distance per pass
    [5]  Pressure ratio        - Fraction of passes under pressure
    [6]  Avg pass length       - Mean pass distance (normalized)
    [7]  High pass ratio       - Fraction of passes that are aerial
    [8]  Progressive pass ratio- Fraction of passes moving >10m forward
    [9]  Receive count         - Passes RECEIVED by this role (normalized)
    [10] Set piece ratio       - Fraction of passes from set piece situations
    
    Edge Features (5-dim):
    [0] Pass length    (normalized)
    [1] Pass angle     (normalized)
    [2] Pressure       (binary)
    [3] Height code    (normalized: 0=Ground, 0.5=Low, 1.0=High)
    [4] Pattern code   (normalized: 0=Regular, 0.5=SetPiece, 1.0=Counter)
    
    Global Context (3-dim):
    [0] Opponent defensive density
    [1] Counterpress intensity
    [2] Half indicator
    
    Targets:
      y:        [1, 5] float - Regression targets
                  [0] Cumulative xT (log-scaled)
                  [1] Press Height (normalized /120)
                  [2] Field Tilt (0-1)
                  [3] Verticality (shifted to 0-1)
                  [4] Tempo (normalized /30)
      y_cls:    [1, 2] long  - Classification targets
                  [0] Defensive Posture (0=LowBlock, 1=MidBlock, 2=HighPress)
                  [1] Offensive Style   (0=Patient, 1=Balanced, 2=Counter)
    """
    
    # 1. Unpack the Bundle
    passes = window['passes'].copy()
    opp_def = window['opp_defense']
    
    # --- 2. NODE FEATURES (x) [12, 11] ---
    if 'node_idx' not in passes.columns:
        raise ValueError("Critical: 'node_idx' missing. Run utils.encode_features() before slicing.")

    node_features = np.zeros((NUM_NODES, 11)) 

    # Pre-fill defaults
    for i in range(NUM_NODES):
        def_x, def_y = DEFAULT_POSITIONS[i]
        node_features[i, 0] = 0.0   # Active = 0
        node_features[i, 1] = def_x # Default X
        node_features[i, 2] = def_y # Default Y
    
    # --- Build receive count lookup ---
    receive_counts = np.zeros(NUM_NODES)
    if 'pass_recipient' in passes.columns:
        player_role_map = passes.groupby('player')['node_idx'].first().to_dict()
        for _, row in passes.iterrows():
            recipient = row.get('pass_recipient', None)
            if pd.notna(recipient) and recipient in player_role_map:
                recv_idx = int(player_role_map[recipient])
                if 0 <= recv_idx < NUM_NODES:
                    receive_counts[recv_idx] += 1
    
    # Aggregate stats per Tactical Role
    grouped = passes.groupby('node_idx')
    for node_idx, data in grouped:
        if 0 <= node_idx < NUM_NODES:
            node_features[node_idx, 0] = 1.0
            node_features[node_idx, 1] = data['x'].mean() / 120.0
            node_features[node_idx, 2] = data['y'].mean() / 80.0
            node_features[node_idx, 3] = len(data) / 20.0
            
            fwd = (data['end_x'] - data['x']).mean()
            node_features[node_idx, 4] = fwd / 120.0
            
            if 'pressure_code' in data.columns:
                node_features[node_idx, 5] = data['pressure_code'].mean()
            
            if 'pass_length' in data.columns:
                node_features[node_idx, 6] = data['pass_length'].mean() / 120.0
            
            if 'height_code' in data.columns:
                node_features[node_idx, 7] = (data['height_code'] == 2).mean()
            
            fwd_distances = data['end_x'] - data['x']
            node_features[node_idx, 8] = (fwd_distances > 10).mean()
            
            node_features[node_idx, 9] = receive_counts[node_idx] / 20.0
            
            if 'pattern_code' in data.columns:
                node_features[node_idx, 10] = (data['pattern_code'] == 1).mean()

    x_tensor = torch.tensor(node_features, dtype=torch.float)

    # --- 3. EDGE CONSTRUCTION [E, 5] ---
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    
    sorted_passes = passes.sort_values('time_min')
    node_indices = sorted_passes['node_idx'].values
    
    p_len = sorted_passes['pass_length'].values if 'pass_length' in passes.columns else np.zeros(len(sorted_passes))
    p_ang = sorted_passes['pass_angle'].values if 'pass_angle' in passes.columns else np.zeros(len(sorted_passes))
    p_pres = sorted_passes['pressure_code'].values if 'pressure_code' in passes.columns else np.zeros(len(sorted_passes))
    p_height = sorted_passes['height_code'].values if 'height_code' in passes.columns else np.zeros(len(sorted_passes))
    p_pattern = sorted_passes['pattern_code'].values if 'pattern_code' in passes.columns else np.zeros(len(sorted_passes))
    poss_ids = sorted_passes['possession'].values if 'possession' in passes.columns else np.zeros(len(sorted_passes))
    
    def make_edge_attr(idx):
        return [
            p_len[idx] / 120.0,
            p_ang[idx] / 3.14,
            float(p_pres[idx]),
            float(p_height[idx]) / 2.0,
            float(p_pattern[idx]) / 2.0,
        ]
    
    for i in range(len(sorted_passes) - 1):
        src = node_indices[i]
        dst = node_indices[i+1]
        
        if (src < NUM_NODES and dst < NUM_NODES) and (poss_ids[i] == poss_ids[i+1]):
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_attrs.append(make_edge_attr(i))
            
    if len(edge_sources) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # --- 4. GLOBAL CONTEXT (u) [1, 3] ---
    # IMPORTANT: Only opponent-facing features that the graph CANNOT see.
    # REMOVED: possession_share (leaked Field Tilt / Tempo)
    # REMOVED: team_territory   (leaked Field Tilt / xT directly)
    # These were allowing the model to shortcut past the graph structure.
    duration = window['end_time'] - window['start_time']
    
    # [0] Opponent defensive density (actions per minute)
    opp_density = len(opp_def) / duration if duration > 0 else 0.0
    
    # [1] Counterpress intensity (opponent counterpresses per minute)
    cp_count = window.get('opp_counterpress_count', 0)
    counterpress_intensity = cp_count / duration if duration > 0 else 0.0
    
    # [2] Half indicator (0 = 1st half, 1 = 2nd half)
    period = window.get('period', 1)
    half_indicator = 0.0 if period <= 1 else 1.0
    
    u = torch.tensor([[
        opp_density,
        counterpress_intensity,
        half_indicator
    ]], dtype=torch.float)

    # --- 5. REGRESSION TARGETS (y) [1, 5] ---
    # [0] Cumulative xT: log(1 + xT) to compress range, keeps non-zero signal
    # [1] Press Height: normalized by pitch length
    # [2] Field Tilt: already [0, 1]
    # [3] Verticality: shifted from [-1, 1] to [0, 1]
    # [4] Tempo: passes per minute, normalized by /30 (typical max ~25-30)
    y = torch.tensor([[
        np.log1p(window['y_cum_xt']),             # log(1 + xT)
        window['y_press_height'] / 120.0,
        window['y_field_tilt'],
        (window['y_verticality'] + 1.0) / 2.0,
        window['y_tempo'] / 30.0,                  # Normalize tempo
    ]], dtype=torch.float)
    
    # --- 6. CLASSIFICATION TARGETS (y_cls) [1, 2] ---
    # [0] Defensive Posture: 0=LowBlock, 1=MidBlock, 2=HighPress
    # [1] Offensive Style:   0=Patient, 1=Balanced, 2=Counter
    y_cls = torch.tensor([[
        window['y_def_posture'],
        window['y_off_style'],
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
    from utils import encode_features
    
    match_id = 8658
    print(f"1. Fetching Match {match_id}...")
    raw = fetch_match_data(match_id)
    
    if not raw['passes'].empty:
        raw['passes'] = encode_features(raw['passes'])
    
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
        print(f"Nodes (x): {g.x.shape} (Should be [12, 11])")
        print(f"Edges: {g.edge_index.shape[1]}")
        print(f"Edge attr: {g.edge_attr.shape} (Should be [E, 5])")
        print(f"Global (u): {g.u.shape} (Should be [1, 3])")
        print(f"Reg targets (y): {g.y.shape} (Should be [1, 5]) -> {g.y.tolist()}")
        print(f"Cls targets (y_cls): {g.y_cls.shape} (Should be [1, 2]) -> {g.y_cls.tolist()}")