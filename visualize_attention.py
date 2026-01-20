import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import numpy as np
import os

# Import your modules
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph
from model import PitchPulseGAT

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Point to your new brain
MATCH_ID_TO_TEST = 3749133  # A specific high-scoring match (or keep random)

def visualize_attention(match_id, window_idx=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Visualizing Attention for Match {match_id} ---")
    
    # 1. Load the "Super Brain"
    model = PitchPulseGAT(in_channels=2,hidden_channels=64, out_channels=1).to(device) # Must match training config!
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    else:
        print("Warning: No checkpoint found! Using random weights (bad).")
        
    model.eval()

    # 2. Get Data
    data = fetch_match_data(match_id)
    windows = get_rolling_windows(data)
    
    # 3. Pick a Window (Find a Goal)
    target_window = None
    if window_idx is None:
        print("Searching for a window with a Goal...")
        for i, w in enumerate(windows):
            goals = w['shots'][w['shots']['is_goal'] == True]
            if not goals.empty:
                scorer = goals.iloc[0]['player']
                print(f"Found Goal in Window {i}! Scored by: {scorer}")
                target_window = w
                break
    else:
        target_window = windows[window_idx]
        
    if target_window is None:
        print("No goal found, using Window 10.")
        target_window = windows[10]

    # 4. Build Graph
    graph_data = build_graph(target_window['passes'])
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data = graph_data.to(device)
    
    # 5. Extract Attention
    edge_index, attention_weights = model(graph_data, return_attention=True)
    
    # Average across heads
    att_scores = attention_weights.mean(dim=1).detach().cpu().numpy()
    
    # Normalize for Plotting (Make the differences pop)
    att_scores = (att_scores - att_scores.min()) / (att_scores.max() - att_scores.min())
    
    # 6. Plotting
    G = to_networkx(graph_data, to_undirected=False)
    pos = {}
    for i, (x, y) in enumerate(graph_data.x):
        # Flip Y for plotting so it looks like a TV broadcast
        pos[i] = (x.item(), 80 - y.item()) 
        
    plt.figure(figsize=(12, 8))
    plt.xlim(-5, 125)
    plt.ylim(-5, 85)
    
    # Draw Pitch Outline
    plt.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color='black', linewidth=2)
    plt.axvline(x=60, color='black', linestyle='--', alpha=0.3)
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=200, alpha=0.9)
    
    # Draw Edges (Red = High Threat, Gray = Low Threat)
    for i, (u, v) in enumerate(G.edges()):
        score = att_scores[i] if i < len(att_scores) else 0.0
        
        # Threshold: Only highlight the top 30% of passes
        if score > 0.6:
            color = 'red'
            width = 3.0
            alpha = 1.0
        elif score > 0.3:
            color = 'orange'
            width = 1.5
            alpha = 0.6
        else:
            color = 'lightgray'
            width = 0.5
            alpha = 0.1
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=width, alpha=alpha)

    plt.title(f"AI Vision: What leads to a goal? (Red = Critical Passes)")
    plt.show()

if __name__ == "__main__":
    # Test on the 2022 World Cup Final (Argentina vs France)
    # Match ID: 3869685
    visualize_attention(3869685)