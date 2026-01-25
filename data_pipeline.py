from statsbombpy import sb
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def fetch_match_data(match_id,raw_events=None):
    """
    Fetches events and splits them into:
    1. Passes (Graph Edges) - Includes 'under_pressure', 'play_pattern', 'pass_height'
    2. Shots (Threat Targets) - Includes 'shot_statsbomb_xg'
    3. Defense (Intensity Labels) - Includes 'counterpress', 'duel_type'
    """
    if raw_events is not None:
        # Use the data loaded from disk
        events = raw_events
    else:
        # Fallback to API if no local data provided
        print(f"Fetching data from API for Match ID: {match_id}...")
        events = sb.events(match_id=match_id)

    # --- 1. PROCESS PASSES (Graph Edges) ---
    if 'pass_outcome' not in events.columns:
        events['pass_outcome'] = None

    # Mask: Successful passes only
    pass_mask = (
        (events['type'] == 'Pass') & 
        (events['location'].notna()) & 
        (events['pass_end_location'].notna()) &
        (events['pass_outcome'].isna()) 
    )
    
    # Select columns if they exist
    potential_pass_cols = [
        'timestamp', 'minute', 'second', 'player', 'pass_recipient', 
        'location', 'pass_end_location', 'team', 'position',
        'under_pressure', 'play_pattern', 'pass_height', 'pass_length', 'pass_angle','possession'
    ]
    cols_pass = [c for c in potential_pass_cols if c in events.columns]
    
    passes = events.loc[pass_mask, cols_pass].copy()
    
    # Handle Boolean Columns
    if 'under_pressure' in passes.columns:
        passes['under_pressure'] = passes['under_pressure'].fillna(False)

    # Vectorized Coordinate Flattening
    if not passes.empty:
        passes[['x', 'y']] = pd.DataFrame(passes['location'].tolist(), index=passes.index)
        passes[['end_x', 'end_y']] = pd.DataFrame(passes['pass_end_location'].tolist(), index=passes.index)
    else:
        passes[['x', 'y', 'end_x', 'end_y']] = pd.DataFrame(columns=['x', 'y', 'end_x', 'end_y'])

    passes['time_min'] = passes['minute'] + passes['second'] / 60.0
    passes = passes.drop(columns=['location', 'pass_end_location'], errors='ignore')

    # --- 2. PROCESS SHOTS (Threat Targets) ---
    shot_mask = (events['type'] == 'Shot') | (events['type'] == 'Own Goal Against')
    
    potential_shot_cols = [
        'timestamp', 'minute', 'second', 'player', 'type', 'location', 'team',
        'shot_statsbomb_xg', 'shot_outcome', 'play_pattern', 'under_pressure'
    ]
    cols_shot = [c for c in potential_shot_cols if c in events.columns]

    if shot_mask.sum() > 0:
        shots = events.loc[shot_mask].copy()
        
        # Ensure Critical Columns Exist (Fill with 0/None if missing)
        if 'shot_statsbomb_xg' not in shots.columns: shots['shot_statsbomb_xg'] = 0.0
        if 'shot_outcome' not in shots.columns: shots['shot_outcome'] = None
        
        shots = shots[cols_shot].copy()
        shots[['x', 'y']] = pd.DataFrame(shots['location'].tolist(), index=shots.index)
        shots['time_min'] = shots['minute'] + shots['second'] / 60.0
        shots = shots.drop(columns=['location'], errors='ignore')
        
        # Own Goal Logic
        is_own_goal = shots['type'] == 'Own Goal Against'
        shots['is_goal'] = (shots['shot_outcome'] == 'Goal') | is_own_goal
        
        # Swap team for Own Goals (attribute to attacking team)
        match_teams = events['team'].unique()
        def swap_team(row):
            if row['type'] == 'Own Goal Against':
                for t in match_teams:
                    if t != row['team']: return t
            return row['team']
        shots['team'] = shots.apply(swap_team, axis=1)
    else:
        shots = pd.DataFrame(columns=['time_min', 'x', 'y', 'is_goal', 'team', 'shot_statsbomb_xg'])

    # --- 3. PROCESS DEFENSIVE ACTIONS (Intensity Labels) ---
    # Events indicating defensive pressure/intensity
    def_types = ['Pressure', 'Duel', 'Interception', 'Block', 'Ball Recovery', 'Foul Committed']
    def_mask = events['type'].isin(def_types)
    
    potential_def_cols = [
        'timestamp', 'minute', 'second', 'player', 'type', 'location', 'team',
        'counterpress', 'under_pressure', 'duel_type'
    ]
    cols_def = [c for c in potential_def_cols if c in events.columns]
    
    if def_mask.sum() > 0:
        defense = events.loc[def_mask, cols_def].copy()
        
        # Fill N/A for counterpress (NaN usually means False)
        if 'counterpress' in defense.columns:
            defense['counterpress'] = defense['counterpress'].fillna(False)
            
        defense[['x', 'y']] = pd.DataFrame(defense['location'].tolist(), index=defense.index)
        defense['time_min'] = defense['minute'] + defense['second'] / 60.0
        defense = defense.drop(columns=['location'], errors='ignore')
    else:
        defense = pd.DataFrame()

    return {
        "passes": passes.sort_values('time_min').reset_index(drop=True),
        "shots": shots.sort_values('time_min').reset_index(drop=True),
        "defense": defense.sort_values('time_min').reset_index(drop=True)
    }

# --- DIAGNOSTIC TEST FUNCTION ---
if __name__ == "__main__":
    # Use the 2018 World Cup Final or the user's specific match ID
    TEST_MATCH_ID = 8658  # 2018 World Cup Final (France vs Croatia)
    
    print(f"Running Diagnostic on Match: {TEST_MATCH_ID}")
    data = fetch_match_data(TEST_MATCH_ID)
    
    if data:
        # 1. Check PASSES
        passes = data['passes']
        print(f"\n[1] PASSES: {passes.shape[0]} events")
        print(f"    - Columns: {list(passes.columns)}")
        if 'play_pattern' in passes.columns:
            print("    - [OK] 'play_pattern' found.")
        else:
            print("    - [FAIL] 'play_pattern' MISSING!")
            
        if 'under_pressure' in passes.columns:
             print(f"    - [OK] 'under_pressure' found. (True Count: {passes['under_pressure'].sum()})")

        # 2. Check SHOTS
        shots = data['shots']
        print(f"\n[2] SHOTS: {shots.shape[0]} events")
        print(f"    - Columns: {list(shots.columns)}")
        if 'shot_statsbomb_xg' in shots.columns:
            xg_sum = shots['shot_statsbomb_xg'].sum()
            print(f"    - [OK] 'shot_statsbomb_xg' found. Total xG in match: {xg_sum:.2f}")
        else:
            print("    - [FAIL] 'shot_statsbomb_xg' MISSING! (Critical for xT Prediction)")

        # 3. Check DEFENSE
        defense = data['defense']
        print(f"\n[3] DEFENSE: {defense.shape[0]} events")
        print(f"    - Columns: {list(defense.columns)}")

        if 'counterpress' in defense.columns:
            cp_count = defense['counterpress'].sum()
            print(f"    - [OK] 'counterpress' found. Count: {cp_count}")
        else:
            print("    - [FAIL] 'counterpress' MISSING! (Critical for Intensity Prediction)")

        print("\n--- TEST COMPLETE ---")
        
        # Optional: Print first row of each to inspect visually
        # print("\nSample Pass:\n", passes.head(1))
        # print("\nSample Shot:\n", shots.head(1))