import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import warnings

# Import your modules
from window_slicer import get_rolling_windows
from data_pipeline import fetch_match_data
from utils import encode_features
from graph_builder import build_graph_from_window

# --- CONFIGURATION ---
MATCH_ID = 8658
WINDOW_SIZE = 5
STRIDE = 1

ROLE_MAP_INV = {
    0: 'GK', 
    1: 'LB', 2: 'CB_L', 3: 'CB_R', 4: 'RB',
    5: 'DM', 6: 'CM_L', 7: 'CM_R', 8: 'AM',
    9: 'LW', 10: 'RW', 11: 'ST'
}

warnings.filterwarnings('ignore')

def get_all_match_goals(raw_data):
    """
    Extracts a simple dataframe of all goals in the match from raw events.
    Returns: DataFrame with columns [minute, second, player, team]
    """
    shots = raw_data['shots']
    if shots.empty:
        return pd.DataFrame()
    
    # Filter for goals
    goals = shots[shots['is_goal'] == True][['minute', 'second', 'player', 'team']].copy()
    return goals

def visualize_gnn_object(data_object, goal_info, is_scoring_team):
    """
    Visualizes the graph. 
    is_scoring_team: Boolean. True if this graph belongs to the team that scored.
    """
    x = data_object.x.numpy()
    edge_index = data_object.edge_index.numpy()
    edge_attr = data_object.edge_attr.numpy()
    team_name = data_object.team_name 
    
    G = nx.DiGraph()
    
    for i in range(12):
        pos_x = x[i, 1] * 120.0
        pos_y = x[i, 2] * 80.0
        active = x[i, 0] > 0
        G.add_node(i, pos=(pos_x, pos_y), active=active)
        
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        pressure = edge_attr[i, 2]
        G.add_edge(src, dst, pressure=pressure)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Pitch
    plt.plot([0, 0], [0, 80], "k-", alpha=0.3)
    plt.plot([0, 120], [80, 80], "k-", alpha=0.3)
    plt.plot([120, 120], [80, 0], "k-", alpha=0.3)
    plt.plot([120, 0], [0, 0], "k-", alpha=0.3)
    plt.plot([60, 60], [0, 80], "k-", alpha=0.3)
    ax.add_patch(plt.Circle((60, 40), 9.15, color="black", fill=False, alpha=0.3))
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Colors
    if "France" in team_name:
        active_color = 'blue'
        inactive_color = 'lightblue'
    elif "Croatia" in team_name:
        active_color = 'red'
        inactive_color = 'salmon'
    else:
        active_color = 'skyblue'
        inactive_color = 'lightgray'

    # Draw Nodes
    inactive_nodes = [n for n, v in G.nodes(data=True) if not v['active']]
    nx.draw_networkx_nodes(G, pos, nodelist=inactive_nodes, node_color=inactive_color, node_size=300, alpha=0.3)
    
    active_nodes = [n for n, v in G.nodes(data=True) if v['active']]
    nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_color=active_color, node_size=700, edgecolors='black')
    
    labels = {i: ROLE_MAP_INV[i] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', font_color='white')
    
    # Draw Edges
    edges = G.edges(data=True)
    normal_edges = [(u, v) for u, v, d in edges if d['pressure'] == 0]
    pressure_edges = [(u, v) for u, v, d in edges if d['pressure'] == 1]
    
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color=active_color, width=1.5, alpha=0.6, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=pressure_edges, edge_color='gold', width=2, alpha=0.9, arrows=True)

    # --- DYNAMIC TITLE ---
    scorer = goal_info['player']
    time_str = f"{int(goal_info['minute'])}:{int(goal_info['second']):02d}"
    
    # Contextual Title
    if is_scoring_team:
        context_str = "ATTACKING PHASE (Goal Scorer)"
    else:
        context_str = "DEFENSIVE REACTION (Goal Conceded)"
        
    plt.title(f"TEAM: {team_name.upper()}\n{context_str}\nGOAL! {scorer} ({time_str}) | Window {data_object.window_id}", 
              fontsize=14, fontweight='bold', color=active_color)
    
    plt.xlim(-5, 125)
    plt.ylim(-5, 85)
    plt.show()

if __name__ == "__main__":
    print(f"1. Fetching Match {MATCH_ID}...")
    raw = fetch_match_data(MATCH_ID)
    if not raw['passes'].empty:
        raw['passes'] = encode_features(raw['passes'])
        
    # --- EXTRACT GLOBAL GOALS FIRST ---
    all_goals = get_all_match_goals(raw)
    print(f"   Found {len(all_goals)} goals in the match.")
        
    print(f"2. Slicing Windows ({WINDOW_SIZE}m / {STRIDE}m)...")
    windows = get_rolling_windows(raw, MATCH_ID, window_size=WINDOW_SIZE, stride=STRIDE)
    
    print("3. Generating Graphs for ALL involved teams...")

    for w in windows:
        w_start = w['start_time']
        w_end = w['end_time']
        
        # Check if ANY goal happened in this time window
        # We look at the global 'all_goals' list, not the window's specific shots
        goals_in_window = all_goals[
            (all_goals['minute'] >= w_start) & 
            (all_goals['minute'] < w_end)
        ]
        
        if not goals_in_window.empty:
            for _, goal in goals_in_window.iterrows():
                
                # Determine context: Did this team score or concede?
                is_scoring_team = (w['team_name'] == goal['team'])
                
                print(f"\n[GOAL DETECTED] Window {w['window_id']} | Team Graph: {w['team_name']} | Context: {'Scored' if is_scoring_team else 'Conceded'}")
                
                # Build the Graph Object
                gnn_data = build_graph_from_window(w)
                
                # Visualize
                visualize_gnn_object(gnn_data, goal, is_scoring_team)