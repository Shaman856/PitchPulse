import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_window_metrics(passes, shots, defense, team_name):
    # 1. Split Data into Team vs Opponent
    t_passes = passes[passes['team'] == team_name]
    opp_passes = passes[passes['team'] != team_name]
    
    t_shots = shots[shots['team'] == team_name]
    
    # CRITICAL CHANGE: For "Opponent Strategy", we need OPPONENT defense data
    opp_def = defense[defense['team'] != team_name]
    
    # --- METRIC 1: OFFENSIVE THREAT (xT / xG) ---
    xg_sum = t_shots['shot_statsbomb_xg'].sum()
    
    # --- METRIC 2: DEFENSIVE INTENSITY (Opponent High Press) ---
    # We measure where the OPPONENT is performing defensive actions.
    # In StatsBomb, X=120 is always the attacking end. 
    # If Opponent is pressing high, their X will be high (near our goal? No, wait).
    # StatsBomb coordinates are relative to the active team's attacking direction.
    # When Team B defends, their actions are logged relative to Team B's attacking direction.
    # So X > 80 for Team B = High Press (Defending near Team A's goal).
    if not opp_def.empty:
        avg_press_height = opp_def['x'].mean()
    else:
        avg_press_height = 50.0 
        
    # --- METRIC 3: TRUE FIELD TILT ---
    t_f3 = t_passes[t_passes['x'] > 80].shape[0]
    opp_f3 = opp_passes[opp_passes['x'] > 80].shape[0]
    total_f3 = t_f3 + opp_f3
    field_tilt = t_f3 / total_f3 if total_f3 > 0 else 0.5
        
    # --- METRIC 4: VERTICALITY ---
    if not t_passes.empty:
        # Use pass_length if available, else euclidean
        if 'pass_length' in t_passes.columns:
            dist = t_passes['pass_length'].sum()
        else:
            dist = np.sqrt((t_passes['end_x']-t_passes['x'])**2 + (t_passes['end_y']-t_passes['y'])**2).sum()
            
        forward_dist = (t_passes['end_x'] - t_passes['x']).sum()
        verticality = forward_dist / dist if dist > 0 else 0.0
    else:
        verticality = 0.0

    return {
        'y_xg': xg_sum,
        'y_press_height': avg_press_height, # Now reflects OPPONENT pressure
        'y_field_tilt': field_tilt,   
        'y_verticality': verticality,
        '_opp_def': opp_def 
    }

def get_rolling_windows(data_dict,match_id, window_size=5, stride=1):
    """
    Slices match into overlapping windows and computes tactical labels 
    for each team separately.
    """
    passes_df = data_dict['passes']
    shots_df = data_dict['shots']
    defense_df = data_dict['defense']
    
    windows = []
    
    # 1. Determine Match Duration
    if not passes_df.empty:
        match_duration = passes_df['time_min'].max()
    else:
        match_duration = 90.0
    
    # 2. Identify Teams
    teams = passes_df['team'].unique()
    if len(teams) < 2:
        print("Warning: Less than 2 teams found. Skipping.")
        return []
    
    # Sort to ensure consistent processing order
    team_list = sorted(teams) 
    
    print(f"Processing Match Duration: {match_duration:.1f} min")
    
    start_time = 0
    window_id = 0
    
    while start_time < match_duration:
        end_time = start_time + window_size
        
        # --- Slice Data (Temporal Slice) ---
        pass_win = passes_df[(passes_df['time_min'] >= start_time) & (passes_df['time_min'] < end_time)]
        shot_win = shots_df[(shots_df['time_min'] >= start_time) & (shots_df['time_min'] < end_time)]
        def_win = defense_df[(defense_df['time_min'] >= start_time) & (defense_df['time_min'] < end_time)]
        
        # We only process windows that have some activity
        if not pass_win.empty:
            
            # --- Generate Training Sample for EACH Team ---
            # We treat Team A and Team B as separate training examples
            for team in team_list:
                
                # Get metrics for this specific team (Contextualized by opponent data)
                metrics = calculate_window_metrics(
                    pass_win, shot_win, def_win, team
                )
                
                # Extract the graph structure (Passes made by THIS team)
                # This isolates the team's passing network for the GAT
                team_passes = pass_win[pass_win['team'] == team].copy()
                team_shots = shot_win[shot_win['team'] == team].copy()
                
                # Only save if the team actually played in this window
                if not team_passes.empty:
                    window_bundle = {
                        'match_id': match_id,
                        'window_id': window_id,
                        'team_name': team,
                        'start_time': start_time,
                        'end_time': end_time,
                        
                        # RAW DATA (Added Opponent Defense for Graph Context)
                        'passes': team_passes, 
                        'shots': team_shots,
                        'opp_defense': metrics['_opp_def'], # <--- NEW: Store this for Graph Builder
                        
                        # LABELS
                        'y_xg': metrics['y_xg'],
                        'y_press_height': metrics['y_press_height'],
                        'y_field_tilt': metrics['y_field_tilt'],
                        'y_verticality': metrics['y_verticality']
                    }
                    windows.append(window_bundle)
            
        start_time += stride
        window_id += 1
        
    return windows

# --- Test Block ---
if __name__ == "__main__":
    from data_pipeline import fetch_match_data
    from utils import encode_features
    
    # 1. Fetch & Encode
    match_id = 8658 # World Cup Final
    print(f"Fetching Match {match_id}...")
    data = fetch_match_data(match_id)
    
    # IMPORTANT: Ensure features are encoded before slicing
    # (Because slicer uses 'pass_length' and other raw cols, but graph builder will need encoded cols)
    if not data['passes'].empty:
        data['passes'] = encode_features(data['passes'])
    
    # 2. Slice
    print("Slicing windows...")
    windows = get_rolling_windows(data,match_id)
    
    print(f"\nGenerated {len(windows)} training samples (Windows x Teams).")
    
    # 3. Inspect a sample to verify "Directness" logic
    if len(windows) > 10:
        w = windows[10] 
        print(f"\n--- Sample Window ID {w['window_id']} ({w['team_name']}) ---")
        print(f"Time: {w['start_time']} - {w['end_time']} min")
        print(f"Graph Edges (Passes): {len(w['passes'])}")
        print("--- LABELS ---")
        print(f"Offensive Threat (xG):  {w['y_xg']:.3f}")
        print(f"Pressing Intensity:     {w['y_press_height']:.1f} (Avg X)")
        print(f"Field Tilt (Dominance): {w['y_field_tilt']:.1%}")
        print(f"Verticality (Directness): {w['y_verticality']:.2f} (Ratio)")