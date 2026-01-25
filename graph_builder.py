import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

# --- CONFIGURATION ---
# Fixed 12-Node structure.
# (Node definitions are handled in utils.py, here we just respect the count)
NUM_NODES = 12 
DEFAULT_POSITIONS = {
    0: [0.05, 0.50], # GK (Goal line, center)
    1: [0.25, 0.10], # LB (Defensive third, left)
    2: [0.20, 0.35], # CB_L (Defensive third, left-center)
    3: [0.20, 0.65], # CB_R (Defensive third, right-center)
    4: [0.25, 0.90], # RB (Defensive third, right)
    5: [0.40, 0.50], # DM (Midfield circle, deep)
    6: [0.55, 0.30], # CM_L (Midfield, left)
    7: [0.55, 0.70], # CM_R (Midfield, right)
    8: [0.70, 0.50], # AM (Attacking third, center)
    9: [0.75, 0.15], # LW (Attacking third, wide left)
    10:[0.75, 0.85], # RW (Attacking third, wide right)
    11:[0.85, 0.50]  # ST (Opponent box, center)
}

def build_graph_from_window(window):
    """
    Converts a Window Bundle (Passes + Labels) into a PyTorch Geometric Data object.
    
    Features:
    1. Fixed 12-Node Topology (GK to Striker).
    2. Sequence-based Edges (Pass i -> Pass i+1).
    3. Ghost Pass Fix (Checks Possession ID chain).
    4. Context-Aware (Global Opponent Density).
    """
    
    # 1. Unpack the Bundle
    passes = window['passes'].copy()
    opp_def = window['opp_defense']
    
    # --- 2. NODE FEATURES (x) ---
    # Shape: [12, 3] -> [Active?, Avg_X, Avg_Y]
    # We rely on 'node_idx' created by utils.encode_features
    if 'node_idx' not in passes.columns:
        raise ValueError("Critical: 'node_idx' missing. Please run utils.encode_features() before slicing.")

    node_features = np.zeros((NUM_NODES, 3)) 

    # 1. Pre-fill with Defaults (The "Ghost" Structure)
    # This ensures that if a player is silent, they stay in position visually
    for i in range(NUM_NODES):
        def_x, def_y = DEFAULT_POSITIONS[i]
        node_features[i, 0] = 0.0   # Active = 0 (Silent)
        node_features[i, 1] = def_x # Default X
        node_features[i, 2] = def_y # Default Y
    
    # Aggregate stats per Tactical Role
    grouped = passes.groupby('node_idx')
    for node_idx, data in grouped:
        if 0 <= node_idx < NUM_NODES:
            node_features[node_idx, 0] = 1.0 # Feature 0: Active in this window
            node_features[node_idx, 1] = data['x'].mean() / 120.0 # Feature 1: Normalized X
            node_features[node_idx, 2] = data['y'].mean() / 80.0  # Feature 2: Normalized Y

    x_tensor = torch.tensor(node_features, dtype=torch.float)

    # --- 3. EDGE CONSTRUCTION (Sequence & Flow) ---
    # Logic: Connect Pass(i) -> Pass(i+1) if they belong to the same possession chain.
    
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    
    # Sort chronologically to establish flow
    sorted_passes = passes.sort_values('time_min')
    node_indices = sorted_passes['node_idx'].values
    
    # Extract Edge Features (Safe Access with defaults)
    p_len = sorted_passes['pass_length'].values if 'pass_length' in passes else np.zeros(len(passes))
    p_ang = sorted_passes['pass_angle'].values if 'pass_angle' in passes else np.zeros(len(passes))
    p_pres = sorted_passes['pressure_code'].values if 'pressure_code' in passes else np.zeros(len(passes))
    
    # Extract Possession ID (Critical for Ghost Pass Fix)
    # If missing (user didn't update pipeline), default to 0s (Logic degrades gracefully but loses fix)
    poss_ids = sorted_passes['possession'].values if 'possession' in passes else np.zeros(len(passes))
    
    for i in range(len(sorted_passes) - 1):
        src = node_indices[i]
        dst = node_indices[i+1]
        
        # EDGE VALIDATION LOGIC:
        # 1. Source and Target must be valid 0-11 Roles (No unknown/subs)
        # 2. Must be part of the SAME possession chain (Fixes "Teleporting" edges)
        if (src < NUM_NODES and dst < NUM_NODES) and (poss_ids[i] == poss_ids[i+1]):
            
            edge_sources.append(src)
            edge_targets.append(dst)
            
            # Edge Attributes: [Normalized Length, Normalized Angle, Pressure Binary]
            attr = [
                p_len[i] / 120.0,  
                p_ang[i] / 3.14,   
                float(p_pres[i])   
            ]
            edge_attrs.append(attr)
            
    # Handle rare case of 0 edges (e.g., 1 pass in whole window)
    if len(edge_sources) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.tensor([[0, 0, 0]], dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # --- 4. GLOBAL CONTEXT (u) ---
    # Feature: Opponent Defensive Density (Actions per Minute)
    duration = window['end_time'] - window['start_time']
    # Avoid division by zero
    opp_density = len(opp_def) / duration if duration > 0 else 0.0
    
    # Global feature vector [1, 1]
    u = torch.tensor([[opp_density]], dtype=torch.float)

    # --- 5. TARGET LABELS (y) ---
    # The Tactical Suite: [xG, PressHeight, Tilt, Verticality]
    y = torch.tensor([[
        window['y_xg'], 
        window['y_press_height'] / 120.0, # Normalize coordinate
        window['y_field_tilt'],
        window['y_verticality']
    ]], dtype=torch.float)

    # --- 6. ASSEMBLE OBJECT ---
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
    
    # Attach Metadata (Useful for debugging/splitting later)
    data.match_id = window.get('match_id', 0)
    data.window_id = window['window_id']
    data.team_name = window['team_name']
    
    return data

# --- TEST BLOCK ---
if __name__ == "__main__":
    from window_slicer import get_rolling_windows
    from data_pipeline import fetch_match_data
    from utils import encode_features
    
    match_id = 8658 # World Cup Final
    print(f"1. Fetching Match {match_id}...")
    raw = fetch_match_data(match_id)
    
    # CRITICAL: Encode features BEFORE slicing so 'node_idx' exists
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
        print("\n--- Graph Inspection (Window 10) ---")
        print(f"Team: {g.team_name} | Match: {g.match_id}")
        print(f"Nodes (x): {g.x.shape} (Should be [12, 3])")
        print(f"Edges: {g.edge_index.shape[1]} (Possession Chains)")
        print(f"Global (u): {g.u.item():.4f} (Opponent Density)")
        print(f"Targets (y): {g.y.tolist()} \n(xG, Press, Tilt, Vert)")