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

    passes = window['passes'].copy()
    opp_def = window['opp_defense']
    
    # --- NODE FEATURES ---
    if 'node_idx' not in passes.columns:
        raise ValueError("Critical: 'node_idx' missing. Run utils.encode_features() before slicing.")

    node_features = np.zeros((NUM_NODES, 11)) 

    for i in range(NUM_NODES):
        def_x, def_y = DEFAULT_POSITIONS[i]
        node_features[i, 0] = 0.0
        node_features[i, 1] = def_x
        node_features[i, 2] = def_y
    
    receive_counts = np.zeros(NUM_NODES)
    if 'pass_recipient' in passes.columns:
        player_role_map = passes.groupby('player')['node_idx'].first().to_dict()
        for _, row in passes.iterrows():
            recipient = row.get('pass_recipient', None)
            if pd.notna(recipient) and recipient in player_role_map:
                recv_idx = int(player_role_map[recipient])
                if 0 <= recv_idx < NUM_NODES:
                    receive_counts[recv_idx] += 1
    
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

    # --- EDGE CONSTRUCTION ---
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

    # --- GLOBAL CONTEXT ---
    duration = window['end_time'] - window['start_time']
    
    opp_density = len(opp_def) / duration if duration > 0 else 0.0
    
    cp_count = window.get('opp_counterpress_count', 0)
    counterpress_intensity = cp_count / duration if duration > 0 else 0.0
    
    period = window.get('period', 1)
    half_indicator = 0.0 if period <= 1 else 1.0
    
    u = torch.tensor([[
        opp_density,
        counterpress_intensity,
        half_indicator
    ]], dtype=torch.float)

    # --- REGRESSION TARGETS (y) ---
    y = torch.tensor([[
        np.log1p(window['y_cum_xt']),
        window['y_press_height'] / 120.0,
        window['y_field_tilt'],
        (window['y_verticality'] + 1.0) / 2.0,
        window['y_tempo'] / 30.0,
    ]], dtype=torch.float)
    
    # === CLASSIFICATION DISABLED ===
    # y_cls = torch.tensor([[
    #     window['y_def_posture'],
    #     window['y_off_style'],
    # ]], dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)

    # === CLASSIFICATION DISABLED ===
    # data.y_cls = y_cls

    data.match_id = window.get('match_id', 0)
    data.window_id = window['window_id']
    data.team_name = window['team_name']
    
    return data