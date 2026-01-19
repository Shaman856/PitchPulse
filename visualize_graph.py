import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Import your existing modules
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph

def visualize_pass_network(graph_data, title="Pass Network"):
    """
    Visualizes a PyG Graph on a 2D pitch.
    """
    # 1. Convert PyG Data to NetworkX Graph
    # to_networkx converts the edge_index into a standard graph object
    G = to_networkx(graph_data, to_undirected=False)
    
    # 2. Extract Positions from Node Features
    # Recall: data.x is [Num_Nodes, 2] containing (Normalized X, Normalized Y)
    # We need to create a dictionary {Node_ID: (x, y)} for the plotter
    pos = {}
    for i, (x, y) in enumerate(graph_data.x):
        # We plot exactly at the normalized coordinates (0.0 to 1.0)
        pos[i] = (x.item(), y.item())

    # 3. Setup the Plot
    plt.figure(figsize=(10, 7))
    
    # Draw the "Pitch" (Just a box for context)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5) # Half-way line
    
    # 4. Draw Nodes (Players)
    # Node color: Blue, Size: 300
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.9)
    
    # 5. Draw Edges (Passes)
    # We use arrows to show direction
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.5)
    
    # 6. Draw Labels (Node IDs)
    # (To see Names, we'd need to modify build_graph to return the name map, 
    # but IDs work for checking structure)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(title)
    plt.xlabel("Pitch Length (Normalized)")
    plt.ylabel("Pitch Width (Normalized)")
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box') # Keep pitch proportions real
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    # 1. Get Data
    print("Fetching data...")
    df = fetch_match_data(8658) # World Cup Final
    windows = get_rolling_windows(df)
    
    # 2. Pick a Window to Visualize
    # Let's look at Window 15 (Mid-game)
    window_idx = 15
    print(f"Visualizing Window {window_idx}...")
    
    graph = build_graph(windows[window_idx])
    
    # 3. Draw!
    visualize_pass_network(graph, title=f"PitchPulse Network - Window {window_idx}")