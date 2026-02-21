# ================================
# CLASSIFICATION DISABLED VERSION
# ================================

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# EXPECTED THREAT (xT) LOOKUP GRID
# =============================================================================

XT_GRID = np.array([
    [0.0030, 0.0036, 0.0042, 0.0045, 0.0048, 0.0052, 0.0064, 0.0082, 0.0117, 0.0191, 0.0370, 0.0790],
    [0.0023, 0.0030, 0.0036, 0.0041, 0.0046, 0.0053, 0.0070, 0.0098, 0.0153, 0.0268, 0.0530, 0.1060],
    [0.0020, 0.0028, 0.0035, 0.0041, 0.0049, 0.0060, 0.0082, 0.0117, 0.0189, 0.0345, 0.0710, 0.1630],
    [0.0018, 0.0026, 0.0034, 0.0042, 0.0052, 0.0064, 0.0088, 0.0126, 0.0209, 0.0390, 0.0810, 0.2100],
    [0.0018, 0.0026, 0.0034, 0.0042, 0.0052, 0.0064, 0.0088, 0.0126, 0.0209, 0.0390, 0.0810, 0.2100],
    [0.0020, 0.0028, 0.0035, 0.0041, 0.0049, 0.0060, 0.0082, 0.0117, 0.0189, 0.0345, 0.0710, 0.1630],
    [0.0023, 0.0030, 0.0036, 0.0041, 0.0046, 0.0053, 0.0070, 0.0098, 0.0153, 0.0268, 0.0530, 0.1060],
    [0.0030, 0.0036, 0.0042, 0.0045, 0.0048, 0.0052, 0.0064, 0.0082, 0.0117, 0.0191, 0.0370, 0.0790],
])

def _get_xt_value(x, y):
    col = int(np.clip(x / 10.0, 0, 11))
    row = int(np.clip(y / 10.0, 0, 7))
    return XT_GRID[row, col]

def compute_cumulative_xt(passes_df):
    if passes_df.empty:
        return 0.0
    
    xt_gained = 0.0
    for _, row in passes_df.iterrows():
        start_xt = _get_xt_value(row['x'], row['y'])
        end_xt = _get_xt_value(row['end_x'], row['end_y'])
        delta = end_xt - start_xt
        if delta > 0:
            xt_gained += delta
    
    return xt_gained


def calculate_window_metrics(passes, shots, defense, team_name, window_duration):

    t_passes = passes[passes['team'] == team_name]
    opp_passes = passes[passes['team'] != team_name]
    opp_def = defense[defense['team'] != team_name]
    
    # --- METRIC 1: Cumulative xT ---
    cum_xt = compute_cumulative_xt(t_passes)
    
    # --- METRIC 2: Press Height ---
    avg_press_height = opp_def['x'].mean() if not opp_def.empty else 50.0
        
    # --- METRIC 3: Field Tilt ---
    t_opp_half = t_passes[t_passes['x'] > 60].shape[0]
    opp_opp_half = opp_passes[opp_passes['x'] > 60].shape[0]
    total_opp_half = t_opp_half + opp_opp_half
    field_tilt = t_opp_half / total_opp_half if total_opp_half > 0 else 0.5
        
    # --- METRIC 4: Verticality ---
    if not t_passes.empty:
        if 'pass_length' in t_passes.columns:
            dist = t_passes['pass_length'].sum()
        else:
            dist = np.sqrt(
                (t_passes['end_x'] - t_passes['x'])**2 +
                (t_passes['end_y'] - t_passes['y'])**2
            ).sum()
            
        forward_dist = (t_passes['end_x'] - t_passes['x']).sum()
        verticality = forward_dist / dist if dist > 0 else 0.0
    else:
        verticality = 0.0

    # --- METRIC 5: Tempo ---
    team_pass_count = len(t_passes)
    tempo = team_pass_count / window_duration if window_duration > 0 else 0.0

    # === CLASSIFICATION DISABLED ===
    # def_posture logic removed
    
    return {
        'y_cum_xt': cum_xt,
        'y_press_height': avg_press_height,
        'y_field_tilt': field_tilt,
        'y_verticality': verticality,
        'y_tempo': tempo,
        '_team_pass_count': team_pass_count,
        '_opp_def': opp_def 
    }


# === CLASSIFICATION DISABLED ===
# Entire offensive style assignment removed
# def assign_offensive_style_labels(windows):
#     return windows


def get_rolling_windows(data_dict, match_id, window_size=5, stride=1):

    passes_df = data_dict['passes']
    shots_df = data_dict['shots']
    defense_df = data_dict['defense']
    
    windows = []
    
    match_duration = passes_df['time_min'].max() if not passes_df.empty else 90.0
    teams = passes_df['team'].unique()
    
    if len(teams) < 2:
        print("Warning: Less than 2 teams found. Skipping.")
        return []
    
    team_list = sorted(teams) 
    start_time = 0
    window_id = 0
    
    while start_time < match_duration:
        end_time = start_time + window_size
        
        pass_win = passes_df[(passes_df['time_min'] >= start_time) & (passes_df['time_min'] < end_time)]
        shot_win = shots_df[(shots_df['time_min'] >= start_time) & (shots_df['time_min'] < end_time)]
        def_win = defense_df[(defense_df['time_min'] >= start_time) & (defense_df['time_min'] < end_time)]
        
        if not pass_win.empty:
            
            duration = window_size
            
            for team in team_list:
                
                metrics = calculate_window_metrics(
                    pass_win, shot_win, def_win, team, duration
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
                        'passes': team_passes,
                        'shots': team_shots,
                        'opp_defense': metrics['_opp_def'],
                        'y_cum_xt': metrics['y_cum_xt'],
                        'y_press_height': metrics['y_press_height'],
                        'y_field_tilt': metrics['y_field_tilt'],
                        'y_verticality': metrics['y_verticality'],
                        'y_tempo': metrics['y_tempo'],
                        # === CLASSIFICATION DISABLED ===
                        # 'y_def_posture': removed
                        # 'y_off_style': removed
                    }
                    windows.append(window_bundle)
            
        start_time += stride
        window_id += 1
    
    # === CLASSIFICATION DISABLED ===
    # windows = assign_offensive_style_labels(windows)
        
    return windows