import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def build_graph(window_df):
    """
    Converts a 15-minute window DataFrame into a PyTorch Geometric Graph.
    Nodes: Any player involved (Sender OR Receiver).
    """
    # --- 1. IDENTIFY NODES (Sender AND Receiver) ---
    senders = window_df['player'].dropna().unique()
    receivers = window_df['pass_recipient'].dropna().unique()
    
    # Union of both sets = All active players
    unique_players = list(set(senders) | set(receivers))
    
    if len(unique_players) == 0:
        # Handle empty window (rare but possible)
        return Data(x=torch.zeros((0, 2)), edge_index=torch.zeros((2, 0), dtype=torch.long))

    # Map 'Player Name' -> 'Node Index'
    player_to_idx = {name: i for i, name in enumerate(unique_players)}
    
    # --- 2. NODE FEATURES (Average Position) ---
    node_features = []
    
    for player in unique_players:
        # Get all actions involving this player (as sender) to determine position
        # If they only received (never sent), we estimate pos from where they received
        p_sent = window_df[window_df['player'] == player]
        p_received = window_df[window_df['pass_recipient'] == player]
        
        if not p_sent.empty:
            avg_x = p_sent['x'].mean()
            avg_y = p_sent['y'].mean()
        elif not p_received.empty:
            # Use end location of passes sent TO them
            avg_x = p_received['end_x'].mean()
            avg_y = p_received['end_y'].mean()
        else:
            avg_x, avg_y = 60.0, 40.0 # Fallback (Midfield)

        # Normalize (120x80 pitch)
        node_features.append([avg_x / 120.0, avg_y / 80.0])

    x = torch.tensor(node_features, dtype=torch.float)

    # --- 3. EDGES (Passes) ---
    source_nodes = []
    target_nodes = []
    
    for _, row in window_df.iterrows():
        sender = row['player']
        receiver = row['pass_recipient']
        
        # Now this check will pass for Strikers too!
        if pd.notna(receiver) and sender in player_to_idx and receiver in player_to_idx:
            src_idx = player_to_idx[sender]
            tgt_idx = player_to_idx[receiver]
            
            source_nodes.append(src_idx)
            target_nodes.append(tgt_idx)
            
    # Safety: Handle windows with 0 edges
    if len(source_nodes) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # --- 4. DATA OBJECT ---
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(unique_players) # Explicitly set num_nodes for safety
    
    return data

# --- TEST BLOCK ---
if __name__ == "__main__":
    from data_pipeline import fetch_match_data
    from window_slicer import get_rolling_windows
    
    print("Fetching and Slicing Data...")
    # 1. Get Data Bundle
    data = fetch_match_data(8658) # France vs Croatia
    windows = get_rolling_windows(data)
    
    # 2. Select Window 5
    print(f"\n--- Building Graph for Window 5 ---")
    target_bundle = windows[5]
    
    # FIX IS HERE: We pass target_bundle['passes'], not just target_bundle
    graph = build_graph(target_bundle['passes'])
    
    print("Graph Created Successfully!")
    print(graph)
    print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")