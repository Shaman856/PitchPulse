from statsbombpy import sb
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def fetch_match_data(match_id, raw_events=None):
    """
    Fetches events and splits them into:
    1. Passes (Graph Edges) - Includes 'under_pressure', 'play_pattern', 'pass_height',
                               'pass_body_part', 'period'
    2. Shots (Threat Targets) - Includes 'shot_statsbomb_xg'
    3. Defense (Intensity Labels) - Includes 'counterpress', 'duel_type'
    4. Carries (Ball Progression) - NEW: Ball carries between events
    5. Dribbles (Ball Progression) - NEW: Successful dribbles/takes-on
    
    TIER 1 CHANGE: Added carries and dribbles extraction for multi-event edge construction.
    These action types contribute to territorial control, threat generation, and tempo
    but were previously invisible to the graph.
    """
    if raw_events is not None:
        events = raw_events
    else:
        print(f"Fetching data from API for Match ID: {match_id}...")
        events = sb.events(match_id=match_id)

    # --- 1. PROCESS PASSES (Graph Edges) ---
    if 'pass_outcome' not in events.columns:
        events['pass_outcome'] = None

    pass_mask = (
        (events['type'] == 'Pass') & 
        (events['location'].notna()) & 
        (events['pass_end_location'].notna()) &
        (events['pass_outcome'].isna()) 
    )
    
    potential_pass_cols = [
        'timestamp', 'minute', 'second', 'player', 'pass_recipient', 
        'location', 'pass_end_location', 'team', 'position',
        'under_pressure', 'play_pattern', 'pass_height', 'pass_length', 
        'pass_angle', 'possession', 'pass_body_part', 'period'
    ]
    cols_pass = [c for c in potential_pass_cols if c in events.columns]
    
    passes = events.loc[pass_mask, cols_pass].copy()
    
    if 'under_pressure' in passes.columns:
        passes['under_pressure'] = passes['under_pressure'].fillna(False)

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
        
        if 'shot_statsbomb_xg' not in shots.columns: shots['shot_statsbomb_xg'] = 0.0
        if 'shot_outcome' not in shots.columns: shots['shot_outcome'] = None
        
        shots = shots[cols_shot].copy()
        shots[['x', 'y']] = pd.DataFrame(shots['location'].tolist(), index=shots.index)
        shots['time_min'] = shots['minute'] + shots['second'] / 60.0
        shots = shots.drop(columns=['location'], errors='ignore')
        
        is_own_goal = shots['type'] == 'Own Goal Against'
        shots['is_goal'] = (shots['shot_outcome'] == 'Goal') | is_own_goal
        
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
    def_types = ['Pressure', 'Duel', 'Interception', 'Block', 'Ball Recovery', 'Foul Committed']
    def_mask = events['type'].isin(def_types)
    
    potential_def_cols = [
        'timestamp', 'minute', 'second', 'player', 'type', 'location', 'team',
        'counterpress', 'under_pressure', 'duel_type', 'position', 'period',
        'possession', 'play_pattern'
    ]
    cols_def = [c for c in potential_def_cols if c in events.columns]
    
    if def_mask.sum() > 0:
        defense = events.loc[def_mask, cols_def].copy()
        
        if 'counterpress' in defense.columns:
            defense['counterpress'] = defense['counterpress'].fillna(False)
            
        if not defense.empty and 'location' in defense.columns:
            defense[['x', 'y']] = pd.DataFrame(defense['location'].tolist(), index=defense.index)
            defense = defense.drop(columns=['location'], errors='ignore')
        else:
            defense[['x', 'y']] = np.nan
            
        defense['time_min'] = defense['minute'] + defense['second'] / 60.0
    else:
        defense = pd.DataFrame()

    # --- 4. PROCESS CARRIES (NEW - Tier 1) ---
    # Carries represent ball movement between events (player running with the ball).
    # StatsBomb records carry start/end locations, which capture territorial progression
    # that passes alone cannot see (e.g., a defender carrying out of the back).
    carry_mask = (events['type'] == 'Carry') & (events['location'].notna())
    
    potential_carry_cols = [
        'timestamp', 'minute', 'second', 'player', 'team', 'position',
        'location', 'carry_end_location', 'under_pressure', 'play_pattern',
        'possession', 'period'
    ]
    cols_carry = [c for c in potential_carry_cols if c in events.columns]
    
    if carry_mask.sum() > 0:
        carries = events.loc[carry_mask, cols_carry].copy()
        
        if 'under_pressure' in carries.columns:
            carries['under_pressure'] = carries['under_pressure'].fillna(False)
        
        if not carries.empty:
            carries[['x', 'y']] = pd.DataFrame(carries['location'].tolist(), index=carries.index)
            
            if 'carry_end_location' in carries.columns:
                carries[['end_x', 'end_y']] = pd.DataFrame(
                    carries['carry_end_location'].tolist(), index=carries.index
                )
            else:
                # Fallback: carry end = carry start (no movement)
                carries['end_x'] = carries['x']
                carries['end_y'] = carries['y']
            
            carries = carries.drop(columns=['location', 'carry_end_location'], errors='ignore')
        
        carries['time_min'] = carries['minute'] + carries['second'] / 60.0
        
        # Filter out trivial carries (< 3m movement) to reduce noise
        if not carries.empty:
            carry_dist = np.sqrt(
                (carries['end_x'] - carries['x'])**2 + 
                (carries['end_y'] - carries['y'])**2
            )
            carries = carries[carry_dist >= 3.0].copy()
    else:
        carries = pd.DataFrame()

    # --- 5. PROCESS DRIBBLES (NEW - Tier 1) ---
    # Successful dribbles (take-ons) represent 1v1 situations won.
    # These break defensive lines and are key to counter-attacking play.
    if 'dribble_outcome' not in events.columns:
        events['dribble_outcome'] = None
        
    dribble_mask = (
        (events['type'] == 'Dribble') & 
        (events['location'].notna()) &
        (events['dribble_outcome'] == 'Complete')  # Only successful dribbles
    )
    
    potential_dribble_cols = [
        'timestamp', 'minute', 'second', 'player', 'team', 'position',
        'location', 'under_pressure', 'play_pattern', 'possession', 'period'
    ]
    cols_dribble = [c for c in potential_dribble_cols if c in events.columns]
    
    if dribble_mask.sum() > 0:
        dribbles = events.loc[dribble_mask, cols_dribble].copy()
        
        if 'under_pressure' in dribbles.columns:
            dribbles['under_pressure'] = dribbles['under_pressure'].fillna(False)
        
        if not dribbles.empty:
            dribbles[['x', 'y']] = pd.DataFrame(dribbles['location'].tolist(), index=dribbles.index)
            # Dribbles don't have an end_location in StatsBomb — use same location
            # The spatial movement is captured by the subsequent carry event
            dribbles['end_x'] = dribbles['x']
            dribbles['end_y'] = dribbles['y']
            dribbles = dribbles.drop(columns=['location'], errors='ignore')
        
        dribbles['time_min'] = dribbles['minute'] + dribbles['second'] / 60.0
    else:
        dribbles = pd.DataFrame()

    return {
        "passes": passes.sort_values('time_min').reset_index(drop=True),
        "shots": shots.sort_values('time_min').reset_index(drop=True),
        "defense": defense.sort_values('time_min').reset_index(drop=True) if not defense.empty else defense,
        "carries": carries.sort_values('time_min').reset_index(drop=True) if not carries.empty else carries,
        "dribbles": dribbles.sort_values('time_min').reset_index(drop=True) if not dribbles.empty else dribbles,
    }

# --- DIAGNOSTIC TEST ---
if __name__ == "__main__":
    TEST_MATCH_ID = 8658  # 2018 World Cup Final
    
    print(f"Running Diagnostic on Match: {TEST_MATCH_ID}")
    data = fetch_match_data(TEST_MATCH_ID)
    
    if data:
        passes = data['passes']
        print(f"\n[1] PASSES: {passes.shape[0]} events")
        print(f"    Columns: {list(passes.columns)}")

        shots = data['shots']
        print(f"\n[2] SHOTS: {shots.shape[0]} events")
        
        defense = data['defense']
        print(f"\n[3] DEFENSE: {defense.shape[0]} events")
        
        carries = data['carries']
        print(f"\n[4] CARRIES: {carries.shape[0]} events (after filtering trivial <3m)")
        if not carries.empty:
            print(f"    Columns: {list(carries.columns)}")
            carry_dists = np.sqrt((carries['end_x']-carries['x'])**2 + (carries['end_y']-carries['y'])**2)
            print(f"    Avg carry distance: {carry_dists.mean():.1f}m")
            print(f"    Teams: {carries['team'].value_counts().to_dict()}")
        
        dribbles = data['dribbles']
        print(f"\n[5] DRIBBLES: {dribbles.shape[0]} events (successful only)")
        if not dribbles.empty:
            print(f"    Teams: {dribbles['team'].value_counts().to_dict()}")

        print("\n--- TEST COMPLETE ---")
