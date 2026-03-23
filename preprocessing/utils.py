# utils.py
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
    - 'body_part_code': 0=Right Foot, 1=Left Foot, 2=Head, 3=Other
    """
    
    # 1. POSITION MAPPING (Detailed -> 12 Tactical Roles)
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
    
    # 3. PLAY PATTERN MAPPING
    def map_pattern(pat):
        if pd.isna(pat): return 0
        if 'Regular' in pat: return 0
        if any(x in pat for x in ['Throw In', 'Free Kick', 'Corner', 'Goal Kick', 'Kick Off']):
            return 1 
        if 'Counter' in pat: return 2
        return 0

    # 4. BODY PART MAPPING
    body_part_map = {
        'Right Foot': 0, 'Left Foot': 1, 'Head': 2, 'Other': 3,
        'Drop Kick': 3, 'Keeper Arm': 3,
    }

    # --- APPLY MAPPINGS ---
    
    if 'position' in df.columns:
        df['pos_group'] = df['position'].map(pos_map).fillna(7).astype(int)
        df['node_idx'] = df['pos_group']
        
    if 'pass_height' in df.columns:
        df['height_code'] = df['pass_height'].map(height_map).fillna(0).astype(int)
        
    if 'play_pattern' in df.columns:
        df['pattern_code'] = df['play_pattern'].apply(map_pattern).astype(int)
        
    if 'under_pressure' in df.columns:
        df['pressure_code'] = df['under_pressure'].fillna(False).astype(int)

    if 'pass_body_part' in df.columns:
        df['body_part_code'] = df['pass_body_part'].map(body_part_map).fillna(0).astype(int)

    return df


def encode_action_features(df, action_type_str):
    """
    TIER 1 NEW: Encodes features for non-pass action types (carries, dribbles, defense).
    Applies position mapping and pressure encoding, plus sets action-specific defaults.
    
    Args:
        df: DataFrame of actions (carries, dribbles, or defensive actions)
        action_type_str: One of 'carry', 'dribble', 'defense'
    
    Returns:
        DataFrame with encoded columns matching what graph_builder expects.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Position mapping (same as passes)
    pos_map = {
        'Goalkeeper': 0,
        'Left Back': 1, 'Left Wing Back': 1,
        'Left Center Back': 2, 'Center Back': 2,
        'Right Center Back': 3,
        'Right Back': 4, 'Right Wing Back': 4,
        'Center Defensive Midfield': 5, 'Right Defensive Midfield': 5, 'Left Defensive Midfield': 5,
        'Left Center Midfield': 6, 'Left Midfield': 6,
        'Right Center Midfield': 7, 'Right Midfield': 7, 'Center Midfield': 7,
        'Center Attacking Midfield': 8, 'Right Attacking Midfield': 8, 'Left Attacking Midfield': 8,
        'Left Wing': 9, 'Left Center Forward': 9,
        'Right Wing': 10, 'Right Center Forward': 10,
        'Center Forward': 11, 'Striker': 11, 'Second Striker': 11
    }
    
    if 'position' in df.columns:
        df['pos_group'] = df['position'].map(pos_map).fillna(7).astype(int)
        df['node_idx'] = df['pos_group']
    
    if 'under_pressure' in df.columns:
        df['pressure_code'] = df['under_pressure'].fillna(False).astype(int)
    else:
        df['pressure_code'] = 0
    
    if 'play_pattern' in df.columns:
        def map_pattern(pat):
            if pd.isna(pat): return 0
            if 'Regular' in pat: return 0
            if any(x in pat for x in ['Throw In', 'Free Kick', 'Corner', 'Goal Kick', 'Kick Off']):
                return 1 
            if 'Counter' in pat: return 2
            return 0
        df['pattern_code'] = df['play_pattern'].apply(map_pattern).astype(int)
    else:
        df['pattern_code'] = 0
    
    # Set defaults for pass-specific features that non-pass actions don't have
    df['height_code'] = 0       # Ground level (not aerial)
    df['body_part_code'] = 0    # Default
    
    # Compute pass_length equivalent (action distance) if not present
    if 'pass_length' not in df.columns:
        if 'end_x' in df.columns and 'x' in df.columns:
            df['pass_length'] = np.sqrt(
                (df['end_x'] - df['x'])**2 + (df['end_y'] - df['y'])**2
            )
        else:
            df['pass_length'] = 0.0
    
    # Compute pass_angle equivalent if not present
    if 'pass_angle' not in df.columns:
        if 'end_x' in df.columns and 'x' in df.columns:
            df['pass_angle'] = np.arctan2(
                df['end_y'] - df['y'], 
                df['end_x'] - df['x']
            )
        else:
            df['pass_angle'] = 0.0
    
    return df
