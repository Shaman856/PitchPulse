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
    """Look up the xT value for a given (x, y) on the StatsBomb pitch."""
    col = int(np.clip(x / 10.0, 0, 11))
    row = int(np.clip(y / 10.0, 0, 7))
    return XT_GRID[row, col]

def compute_cumulative_xt(passes_df):
    """Computes cumulative xT gained from all passes (positive gains only)."""
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


def compute_cumulative_xt_all_actions(passes_df, carries_df):
    """
    TIER 1 CHANGE: Computes cumulative xT from passes AND carries.
    Carries move the ball into higher-threat zones and should count
    toward total threat generated.
    """
    xt_from_passes = compute_cumulative_xt(passes_df)
    
    xt_from_carries = 0.0
    if not carries_df.empty:
        for _, row in carries_df.iterrows():
            if pd.notna(row.get('x')) and pd.notna(row.get('end_x')):
                start_xt = _get_xt_value(row['x'], row['y'])
                end_xt = _get_xt_value(row['end_x'], row['end_y'])
                delta = end_xt - start_xt
                if delta > 0:
                    xt_from_carries += delta
    
    return xt_from_passes + xt_from_carries


def compute_running_score(shots_df, team_name, window_start):
    """
    Computes the score differential at the start of a window.
    Returns: (team_goals - opponent_goals) as of window_start time.
    """
    if shots_df.empty:
        return 0
    
    goals_before = shots_df[
        (shots_df['is_goal'] == True) & 
        (shots_df['time_min'] < window_start)
    ]
    
    if goals_before.empty:
        return 0
    
    team_goals = (goals_before['team'] == team_name).sum()
    opp_goals = (goals_before['team'] != team_name).sum()
    
    return int(team_goals - opp_goals)


def compute_match_outcome(shots_df, team_name):
    """
    SUITE NEW: Computes the final match outcome for a team.
    
    Returns:
        0 = Loss
        1 = Draw  
        2 = Win
    
    Used as a classification target for the Outcome Probability head.
    Every window in a match gets the SAME outcome label — the model
    learns which tactical states are associated with winning/losing.
    """
    if shots_df.empty:
        return 1  # Default to draw if no shot data
    
    all_goals = shots_df[shots_df['is_goal'] == True]
    
    team_goals = (all_goals['team'] == team_name).sum()
    opp_goals = (all_goals['team'] != team_name).sum()
    
    if team_goals > opp_goals:
        return 2  # Win
    elif team_goals < opp_goals:
        return 0  # Loss
    else:
        return 1  # Draw


def calculate_window_metrics(passes, shots, defense, carries, team_name, window_duration):
    """
    Computes all tactical metrics for a single team in a single window.
    
    TIER 1 CHANGE: Now includes carries in xT computation.
    
    Returns 5 regression targets + 1 classification target (defensive posture).
    """
    # Split into team vs opponent
    t_passes = passes[passes['team'] == team_name]
    opp_passes = passes[passes['team'] != team_name]
    t_carries = carries[carries['team'] == team_name] if not carries.empty else pd.DataFrame()
    
    t_shots = shots[shots['team'] == team_name]
    opp_def = defense[defense['team'] != team_name]
    
    # --- METRIC 1: CUMULATIVE xT (Now includes carries) ---
    cum_xt = compute_cumulative_xt_all_actions(t_passes, t_carries)
    
    # --- METRIC 2: PRESS HEIGHT ---
    if not opp_def.empty and 'x' in opp_def.columns:
        avg_press_height = opp_def['x'].mean()
    else:
        avg_press_height = 50.0 
        
    # --- METRIC 3: FIELD TILT ---
    t_opp_half = t_passes[t_passes['x'] > 60].shape[0]
    opp_opp_half = opp_passes[opp_passes['x'] > 60].shape[0]
    
    # Also count carries in opponent half for territorial dominance
    if not t_carries.empty:
        t_opp_half += t_carries[t_carries['x'] > 60].shape[0]
    opp_carries = carries[carries['team'] != team_name] if not carries.empty else pd.DataFrame()
    if not opp_carries.empty:
        opp_opp_half += opp_carries[opp_carries['x'] > 60].shape[0]
    
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

    # --- METRIC 5: TEMPO (Passes per Minute) ---
    team_pass_count = len(t_passes)
    tempo = team_pass_count / window_duration if window_duration > 0 else 0.0
    
    # --- CLASSIFICATION 1: DEFENSIVE POSTURE ---
    if avg_press_height > 65:
        def_posture = 2  # High Press
    elif avg_press_height > 50:
        def_posture = 1  # Mid Block
    else:
        def_posture = 0  # Low Block
    
    return {
        'y_cum_xt': cum_xt,
        'y_press_height': avg_press_height,
        'y_field_tilt': field_tilt,   
        'y_verticality': verticality,
        'y_tempo': tempo,
        'y_def_posture': def_posture,
        '_team_pass_count': team_pass_count,
        '_opp_def': opp_def 
    }


def assign_offensive_style_labels(windows):
    """
    TWO-PASS CLASSIFICATION: Assigns offensive style labels using
    percentile-based thresholds for balanced classes.
    """
    if not windows:
        return windows
    
    verticalities = np.array([w['y_verticality'] for w in windows])
    tempos = np.array([w['y_tempo'] for w in windows])
    
    vert_min, vert_max = verticalities.min(), verticalities.max()
    tempo_min, tempo_max = tempos.min(), tempos.max()
    
    vert_range = vert_max - vert_min if vert_max != vert_min else 1.0
    tempo_range = tempo_max - tempo_min if tempo_max != tempo_min else 1.0
    
    norm_vert = (verticalities - vert_min) / vert_range
    norm_tempo = (tempos - tempo_min) / tempo_range
    
    directness = 0.5 * norm_vert + 0.5 * norm_tempo
    
    p33 = np.percentile(directness, 33.33)
    p67 = np.percentile(directness, 66.67)
    
    for i, w in enumerate(windows):
        if directness[i] <= p33:
            w['y_off_style'] = 0   # Patient Build-Up
        elif directness[i] <= p67:
            w['y_off_style'] = 1   # Balanced
        else:
            w['y_off_style'] = 2   # Counter / Direct
    
    return windows


def get_rolling_windows(data_dict, match_id, window_size=5, stride=1):
    """
    Slices match into overlapping windows and computes tactical labels.
    
    TIER 1 CHANGES:
    - Now slices carries and dribbles into each window
    - Computes running score differential at each window start
    - Computes match progress (0→1) for each window
    - Passes carries/dribbles through to graph_builder via window bundle
    """
    passes_df = data_dict['passes']
    shots_df = data_dict['shots']
    defense_df = data_dict['defense']
    carries_df = data_dict.get('carries', pd.DataFrame())
    dribbles_df = data_dict.get('dribbles', pd.DataFrame())
    
    windows = []
    
    # Match duration
    all_times = []
    if not passes_df.empty:
        all_times.append(passes_df['time_min'].max())
    if not defense_df.empty and 'time_min' in defense_df.columns:
        all_times.append(defense_df['time_min'].max())
    if not carries_df.empty and 'time_min' in carries_df.columns:
        all_times.append(carries_df['time_min'].max())
    
    match_duration = max(all_times) if all_times else 90.0
    
    # Teams
    teams = passes_df['team'].unique()
    if len(teams) < 2:
        print("Warning: Less than 2 teams found. Skipping.")
        return []
    team_list = sorted(teams)
    
    print(f"Processing Match Duration: {match_duration:.1f} min")
    
    # SUITE NEW: Compute match outcome for each team (same for all windows in match)
    team_outcomes = {}
    for team in team_list:
        team_outcomes[team] = compute_match_outcome(shots_df, team)
    print(f"Match Outcomes: {team_outcomes}")
    
    start_time = 0
    window_id = 0
    
    # =====================================================================
    # PASS 1: Compute all regression metrics + defensive posture
    # =====================================================================
    while start_time < match_duration:
        end_time = start_time + window_size
        
        # --- Temporal Slice ---
        pass_win = passes_df[(passes_df['time_min'] >= start_time) & (passes_df['time_min'] < end_time)]
        shot_win = shots_df[(shots_df['time_min'] >= start_time) & (shots_df['time_min'] < end_time)]
        def_win = defense_df[(defense_df['time_min'] >= start_time) & (defense_df['time_min'] < end_time)] if not defense_df.empty else pd.DataFrame()
        
        # NEW: Slice carries and dribbles
        carry_win = pd.DataFrame()
        if not carries_df.empty and 'time_min' in carries_df.columns:
            carry_win = carries_df[(carries_df['time_min'] >= start_time) & (carries_df['time_min'] < end_time)]
        
        dribble_win = pd.DataFrame()
        if not dribbles_df.empty and 'time_min' in dribbles_df.columns:
            dribble_win = dribbles_df[(dribbles_df['time_min'] >= start_time) & (dribbles_df['time_min'] < end_time)]
        
        if not pass_win.empty:
            
            total_passes_in_window = len(pass_win)
            duration = window_size
            
            # Counterpress count
            opp_counterpress = {}
            for team in team_list:
                opp_def_win = def_win[def_win['team'] != team] if not def_win.empty else pd.DataFrame()
                if 'counterpress' in opp_def_win.columns:
                    opp_counterpress[team] = opp_def_win['counterpress'].sum()
                else:
                    opp_counterpress[team] = 0
            
            # Period
            window_midpoint = (start_time + end_time) / 2.0
            if 'period' in pass_win.columns:
                period = pass_win['period'].mode().iloc[0] if not pass_win['period'].mode().empty else 1
            else:
                period = 1 if window_midpoint < 45 else 2
            
            # NEW: Match progress (0 → 1)
            match_progress = min(window_midpoint / match_duration, 1.0) if match_duration > 0 else 0.5
            
            for team in team_list:
                
                metrics = calculate_window_metrics(
                    pass_win, shot_win, def_win, carry_win, team, duration
                )
                
                team_passes = pass_win[pass_win['team'] == team].copy()
                team_shots = shot_win[shot_win['team'] == team].copy()
                team_carries = carry_win[carry_win['team'] == team].copy() if not carry_win.empty else pd.DataFrame()
                team_dribbles = dribble_win[dribble_win['team'] == team].copy() if not dribble_win.empty else pd.DataFrame()
                
                # NEW: Team's defensive actions in this window
                team_defense = def_win[def_win['team'] == team].copy() if not def_win.empty else pd.DataFrame()
                
                # NEW: Running score at window start
                score_diff = compute_running_score(shots_df, team, start_time)
                
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
                        
                        # NEW: Additional action types for multi-event edges
                        'carries': team_carries,
                        'dribbles': team_dribbles,
                        'team_defense': team_defense,
                        
                        # Global context
                        'total_passes_in_window': total_passes_in_window,
                        'opp_counterpress_count': opp_counterpress[team],
                        'period': period,
                        
                        # NEW: Game state context
                        'score_diff': score_diff,
                        'match_progress': match_progress,
                        'match_duration': match_duration,
                        
                        # --- REGRESSION LABELS (5) ---
                        'y_cum_xt': metrics['y_cum_xt'],
                        'y_press_height': metrics['y_press_height'],
                        'y_field_tilt': metrics['y_field_tilt'],
                        'y_verticality': metrics['y_verticality'],
                        'y_tempo': metrics['y_tempo'],
                        
                        # --- CLASSIFICATION LABELS ---
                        'y_def_posture': metrics['y_def_posture'],
                        # y_off_style assigned in Pass 2
                        'y_outcome': team_outcomes[team],  # SUITE NEW: 0=Loss, 1=Draw, 2=Win
                    }
                    windows.append(window_bundle)
            
        start_time += stride
        window_id += 1
    
    # =====================================================================
    # PASS 2: Assign offensive style from full distribution
    # =====================================================================
    windows = assign_offensive_style_labels(windows)
        
    return windows

# --- Test Block ---
if __name__ == "__main__":
    from data_pipeline import fetch_match_data
    from utils import encode_features, encode_action_features
    
    match_id = 8658
    print(f"Fetching Match {match_id}...")
    data = fetch_match_data(match_id)
    
    if not data['passes'].empty:
        data['passes'] = encode_features(data['passes'])
    if not data['carries'].empty:
        data['carries'] = encode_action_features(data['carries'], 'carry')
    if not data['dribbles'].empty:
        data['dribbles'] = encode_action_features(data['dribbles'], 'dribble')
    if not data['defense'].empty:
        data['defense'] = encode_action_features(data['defense'], 'defense')
    
    print("Slicing windows...")
    windows = get_rolling_windows(data, match_id)
    
    print(f"\nGenerated {len(windows)} training samples.")
    
    # Check new fields
    if windows:
        w = windows[10]
        print(f"\n--- Sample Window ID {w['window_id']} ({w['team_name']}) ---")
        print(f"Time: {w['start_time']} - {w['end_time']} min")
        print(f"Score Diff: {w['score_diff']}")
        print(f"Match Progress: {w['match_progress']:.3f}")
        print(f"Carries: {len(w['carries'])} | Dribbles: {len(w['dribbles'])}")
        print(f"Team Defense Actions: {len(w['team_defense'])}")
        print(f"Cum xT (incl carries): {w['y_cum_xt']:.4f}")
