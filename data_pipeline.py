from statsbombpy import sb
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def fetch_match_data(match_id):
    """
    Fetches events and splits them into:
    1. [cite_start]COMPLETED Passes (for the Graph Edges) [cite: 57]
    2. Shots & Own Goals (for the Target Label)
    """
    print(f"Fetching data for Match ID: {match_id}...")
    events = sb.events(match_id=match_id)
    
    # --- 1. PROCESS PASSES (For the Graph) ---
    # [cite_start]PDF Requirement: "Edges: Completed passes" [cite: 57]
    # In StatsBomb, successful passes have 'pass_outcome' as NaN.
    # If 'pass_outcome' is 'Incomplete', 'Out', etc., it failed.
    
    # Ensure 'pass_outcome' column exists (sometimes it's missing if all passes were good)
    if 'pass_outcome' not in events.columns:
        events['pass_outcome'] = None

    pass_mask = (
        (events['type'] == 'Pass') & 
        (events['location'].notna()) & 
        (events['pass_end_location'].notna()) &
        (events['pass_outcome'].isna())  # <--- CRITICAL FIX: Only successful passes
    )
    
    cols_pass = ['timestamp', 'minute', 'second', 'player', 'pass_recipient', 
                 'location', 'pass_end_location', 'team']
    passes = events.loc[pass_mask, cols_pass].copy()
    
    # Flatten Coordinates
    passes['x'] = passes['location'].apply(lambda loc: loc[0])
    passes['y'] = passes['location'].apply(lambda loc: loc[1])
    passes['end_x'] = passes['pass_end_location'].apply(lambda loc: loc[0])
    passes['end_y'] = passes['pass_end_location'].apply(lambda loc: loc[1])
    
    # Time Fix
    passes['time_min'] = passes['minute'] + passes['second'] / 60.0
    passes = passes.drop(columns=['location', 'pass_end_location'])
    
    # --- 2. PROCESS SHOTS & GOALS (For the Target Label) ---
    # We want: Normal Shots OR Own Goals
    shot_mask = (events['type'] == 'Shot') | (events['type'] == 'Own Goal Against')
    
    # Columns to keep (Handle missing 'shot_outcome' for Own Goals)
    cols_shot = ['timestamp', 'minute', 'second', 'player', 'type', 'location', 'team']
    if 'shot_outcome' in events.columns:
        cols_shot.append('shot_outcome')
    
    if shot_mask.sum() > 0:
        shots = events.loc[shot_mask].copy()
        
        # Ensure 'shot_outcome' exists
        if 'shot_outcome' not in shots.columns: 
            shots['shot_outcome'] = None
            
        shots = shots[cols_shot].copy()
        
        # Flatten Coords (Handle cases where location might be missing for some weird events)
        shots['x'] = shots['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else 120)
        shots['y'] = shots['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else 40)
        shots['time_min'] = shots['minute'] + shots['second'] / 60.0
        shots = shots.drop(columns=['location'])
        
        # --- LOGIC TO FIX OWN GOALS ---
        # 1. Identify Own Goals
        is_own_goal = shots['type'] == 'Own Goal Against'
        
        # 2. Mark them as goals
        shots['is_goal'] = (shots['shot_outcome'] == 'Goal') | is_own_goal
        
        # 3. Swap Team for Own Goals
        # (Attribute the goal to the opponent, not the scorer)
        match_teams = events['team'].unique()
        
        def swap_team(row):
            if row['type'] == 'Own Goal Against':
                for t in match_teams:
                    if t != row['team']:
                        return t
            return row['team']
            
        shots['team'] = shots.apply(swap_team, axis=1)
        
    else:
        shots = pd.DataFrame(columns=['time_min', 'x', 'y', 'is_goal', 'team'])

    return {
        "passes": passes.sort_values('time_min').reset_index(drop=True),
        "shots": shots.sort_values('time_min').reset_index(drop=True)
    }

# --- Test Block ---
if __name__ == "__main__":
    data = fetch_match_data(8658) 
    df_passes = data['passes']
    df_shots = data['shots']
    
    print(f"\nExtracted {len(df_passes)} SUCCESSFUL passes.") # Should be lower than 846
    print(f"Extracted {len(df_shots)} shots/goals.")