import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def build_graph(window_df):
    """
    Converts a 15-minute window DataFrame into a PyTorch Geometric Graph.
    """
    # 1. IDENTIFY NODES (PLAYERS)
    # We define nodes as any player who made a pass in this window.
    # (Design choice: Players who don't touch the ball are excluded to keep the graph dense)
    unique_players = list(set(window_df['player'].dropna().unique()))
    
    # Map 'Player Name' -> 'Node Index' (0, 1, 2...)
    player_to_idx = {name: i for i, name in enumerate(unique_players)}
    # print(player_to_idx)
    
    # 2. CREATE NODE FEATURES (Average Position)
    # Shape: [Num_Nodes, 2] -> (x, y)
    node_features = []
    
    for player in unique_players:
        # Get all passes made by this player in this window
        p_stats = window_df[window_df['player'] == player]
        
        # Calculate average position
        avg_x = p_stats['x'].mean()
        avg_y = p_stats['y'].mean()
        
        # Normalize coordinates to 0-1 range (Pitch dims approx 120x80)
        # Normalization helps the Neural Network learn faster
        norm_x = avg_x / 120.0
        norm_y = avg_y / 80.0
        
        node_features.append([norm_x, norm_y])
        
    # Convert to PyTorch Tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 3. DEFINE EDGES (Passes)
    # Shape: [2, Num_Edges]
    source_nodes = []
    target_nodes = []
    
    for _, row in window_df.iterrows():
        sender = row['player']
        receiver = row['pass_recipient']
        
        # We only create an edge if the receiver is also a valid Node
        # (i.e., the receiver also made a pass in this window)
        if pd.notna(receiver) and receiver in player_to_idx:
            src_idx = player_to_idx[sender]
            tgt_idx = player_to_idx[receiver]
            
            source_nodes.append(src_idx)
            target_nodes.append(tgt_idx)
            
    # PyG requires LongTensor for indices
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # 4. CREATE DATA OBJECT
    data = Data(x=x, edge_index=edge_index)
    
    return data

# --- Test Block ---
if __name__ == "__main__":
    from data_pipeline import fetch_match_data
    from window_slicer import get_rolling_windows
    
    # 1. Get Data
    print("Fetching and Slicing Data...")
    df = fetch_match_data(8658)
    windows = get_rolling_windows(df)
    
    # 2. Build Graph for Window #5 (arbitrary choice)
    print("\n--- Building Graph for Window 5 ---")
    window_5 = windows[5]
    graph_data = build_graph(window_5)
    
    # 3. Verify Structure
    print("Success!")
    print(f"Num Nodes: {graph_data.num_nodes} (Active players)")
    print(f"Num Edges: {graph_data.num_edges} (Passes between them)")
    print(f"Feature Matrix (x): {graph_data.x.shape} -> Should be [Nodes, 2]")
    print(f"Edge Index: {graph_data.edge_index.shape} -> Should be [2, Edges]")
    
    # Check what the first node looks like
    print(f"\nExample Node Feature (Normalized X, Y): {graph_data.x[0]}")