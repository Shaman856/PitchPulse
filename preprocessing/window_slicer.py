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
    
    # --- METRIC 1: OFFENSIVE THREAT (Threat Zone Entry Rate) ---
    # Instead of xG (which is sparse/binary), we measure what fraction of the
    # team's passes enter the penalty box area (x > 102, 18 < y < 62).
    # This gives every window a continuous, non-zero value reflecting how 
    # much the team is threatening the goal through passing.
    if not t_passes.empty:
        box_entries = t_passes[
            (t_passes['end_x'] > 102) & 
            (t_passes['end_y'] > 18) & 
            (t_passes['end_y'] < 62)
        ].shape[0]
        threat_rate = box_entries / len(t_passes)
    else:
        threat_rate = 0.0
    
    # --- METRIC 2: DEFENSIVE INTENSITY (Opponent High Press) ---
    if not opp_def.empty:
        avg_press_height = opp_def['x'].mean()
    else:
        avg_press_height = 50.0 
        
    # --- METRIC 3: TERRITORIAL DOMINANCE (Field Tilt) ---
    # Changed from final third (x > 80) to opponent half (x > 60).
    # This roughly triples the pass count in the denominator, producing
    # smoother ratios (0.42, 0.55, 0.61) instead of discrete jumps (0, 0.5, 1.0).
    t_opp_half = t_passes[t_passes['x'] > 60].shape[0]
    opp_opp_half = opp_passes[opp_passes['x'] > 60].shape[0]
    total_opp_half = t_opp_half + opp_opp_half
    field_tilt = t_opp_half / total_opp_half if total_opp_half > 0 else 0.5
        
    # --- METRIC 4: VERTICALITY ---
    if not t_passes.empty:
        if 'pass_length' in t_passes.columns:
            dist = t_passes['pass_length'].sum()
        else:
            dist = np.sqrt((t_passes['end_x']-t_passes['x'])**2 + (t_passes['end_y']-t_passes['y'])**2).sum()
            
        forward_dist = (t_passes['end_x'] - t_passes['x']).sum()
        verticality = forward_dist / dist if dist > 0 else 0.0
    else:
        verticality = 0.0

    return {
        'y_threat_rate': threat_rate,
        'y_press_height': avg_press_height,
        'y_field_tilt': field_tilt,   
        'y_verticality': verticality,
        '_opp_def': opp_def 
    }

def get_rolling_windows(data_dict, match_id, window_size=5, stride=1):
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
        
        if not pass_win.empty:
            
            # --- GLOBAL STATS (Computed once per window for both teams) ---
            total_passes_in_window = len(pass_win)
            
            # Counterpress count from opponent defense
            opp_counterpress = {}
            for team in team_list:
                opp_def_win = def_win[def_win['team'] != team]
                if 'counterpress' in opp_def_win.columns:
                    opp_counterpress[team] = opp_def_win['counterpress'].sum()
                else:
                    opp_counterpress[team] = 0
            
            # Determine period (1st half vs 2nd half)
            # Use the midpoint of the window to determine which half
            window_midpoint = (start_time + end_time) / 2.0
            if 'period' in pass_win.columns:
                # Use actual period from data (handles extra time correctly)
                period = pass_win['period'].mode().iloc[0] if not pass_win['period'].mode().empty else 1
            else:
                period = 1 if window_midpoint < 45 else 2
            
            for team in team_list:
                
                metrics = calculate_window_metrics(
                    pass_win, shot_win, def_win, team
                )
                
                team_passes = pass_win[pass_win['team'] == team].copy()
                team_shots = shot_win[shot_win['team'] == team].copy()
                
                if not team_passes.empty:
                    window_bundle = {
                        'match_id': match_id,
                        'window_id': window_id,
                        'team_name': team,
                        'start_time': start_time,
                        'end_time': end_time,
                        
                        # RAW DATA
                        'passes': team_passes, 
                        'shots': team_shots,
                        'opp_defense': metrics['_opp_def'],
                        
                        # NEW: Global context data for graph_builder
                        'total_passes_in_window': total_passes_in_window,
                        'opp_counterpress_count': opp_counterpress[team],
                        'period': period,
                        
                        # LABELS
                        'y_threat_rate': metrics['y_threat_rate'],
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
    
    match_id = 8658
    print(f"Fetching Match {match_id}...")
    data = fetch_match_data(match_id)
    
    if not data['passes'].empty:
        data['passes'] = encode_features(data['passes'])
    
    print("Slicing windows...")
    windows = get_rolling_windows(data, match_id)
    
    print(f"\nGenerated {len(windows)} training samples.")
    
    if len(windows) > 10:
        w = windows[10] 
        print(f"\n--- Sample Window ID {w['window_id']} ({w['team_name']}) ---")
        print(f"Time: {w['start_time']} - {w['end_time']} min")
        print(f"Total passes in window: {w['total_passes_in_window']}")
        print(f"Opp counterpress count: {w['opp_counterpress_count']}")
        print(f"Period: {w['period']}")
        print(f"--- LABELS ---")
        print(f"Threat Rate: {w['y_threat_rate']:.3f}")
        print(f"Press Height: {w['y_press_height']:.1f}")
        print(f"Field Tilt: {w['y_field_tilt']:.1%}")
        print(f"Verticality: {w['y_verticality']:.2f}")
