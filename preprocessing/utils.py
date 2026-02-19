import pandas as pd
import numpy as np

def encode_features(df):
    """
    Applies categorical mapping to StatsBomb text features for GAT/LSTM.
    Safely handles missing columns by checking existence first.
    
    Returns dataframe with new encoded columns:
    - 'pos_group': 0-11 (The 12 fixed Tactical Roles for the Graph Nodes)
    - 'node_idx': Duplicate of 'pos_group' (Used explicitly by graph_builder)
    - 'height_code': 0=Ground, 1=Low, 2=High
    - 'pattern_code': 0=Regular, 1=SetPiece, 2=Counter
    - 'pressure_code': 0=No, 1=Yes
    """
    
    # 1. POSITION MAPPING (Detailed -> 12 Tactical Roles)
    # We map specific StatsBomb names to our 12 fixed graph nodes.
    # The Graph Builder expects these exact integer IDs (0 to 11).
    pos_map = {
        # 0: Goalkeeper
        'Goalkeeper': 0,
        
        # 1: Left Back (Includes Wing Backs)
        'Left Back': 1, 'Left Wing Back': 1,
        
        # 2: Left Center Back (Generic CBs default here)
        'Left Center Back': 2, 'Center Back': 2,
        
        # 3: Right Center Back
        'Right Center Back': 3,
        
        # 4: Right Back (Includes Wing Backs)
        'Right Back': 4, 'Right Wing Back': 4,
        
        # 5: Defensive Midfield (The "Holding" Role)
        'Center Defensive Midfield': 5, 'Right Defensive Midfield': 5, 'Left Defensive Midfield': 5,
        
        # 6: Left Center Midfield
        'Left Center Midfield': 6, 'Left Midfield': 6, 
        
        # 7: Right Center Midfield
        'Right Center Midfield': 7, 'Right Midfield': 7, 'Center Midfield': 7,
        
        # 8: Attacking Midfield (The "10" Role)
        'Center Attacking Midfield': 8, 'Right Attacking Midfield': 8, 'Left Attacking Midfield': 8,
        
        # 9: Left Wing
        'Left Wing': 9, 'Left Center Forward': 9,
        
        # 10: Right Wing
        'Right Wing': 10, 'Right Center Forward': 10,
        
        # 11: Striker
        'Center Forward': 11, 'Striker': 11, 'Second Striker': 11
    }
    
    # 2. PASS HEIGHT MAPPING
    height_map = {'Ground Pass': 0, 'Low Pass': 1, 'High Pass': 2}
    
    # 3. PLAY PATTERN MAPPING (Grouping sparse classes)
    def map_pattern(pat):
        if pd.isna(pat): return 0 # Default to Regular
        if 'Regular' in pat: return 0
        # Group all Set Pieces together
        if any(x in pat for x in ['Throw In', 'Free Kick', 'Corner', 'Goal Kick', 'Kick Off']):
            return 1 
        if 'Counter' in pat: return 2
        return 0 # Default other obscure types to Regular for simplicity

    # --- APPLY MAPPINGS ---
    
    # Position Encoding (Node Feature)
    if 'position' in df.columns:
        # Map positions using the 12-node map. 
        # Fallback to 'Center Midfield' (7) if the position is unknown/NaN.
        df['pos_group'] = df['position'].map(pos_map).fillna(7).astype(int)
        
        # Create 'node_idx' alias - This is what graph_builder.py looks for
        df['node_idx'] = df['pos_group']
        
    # Pass Height Encoding (Edge Feature)
    if 'pass_height' in df.columns:
        df['height_code'] = df['pass_height'].map(height_map).fillna(0).astype(int)
        
    # Play Pattern Encoding (Global/State Feature)
    if 'play_pattern' in df.columns:
        df['pattern_code'] = df['play_pattern'].apply(map_pattern).astype(int)
        
    # Pressure Encoding (Edge Weight / Attention)
    if 'under_pressure' in df.columns:
        df['pressure_code'] = df['under_pressure'].fillna(False).astype(int)

    return df