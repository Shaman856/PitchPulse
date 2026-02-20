import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# EXPECTED THREAT (xT) LOOKUP GRID
# =============================================================================
# Published xT values on a 12x8 grid (pitch divided into 12 columns x 8 rows).
# Source: Karun Singh's xT model (widely used in football analytics).
# Each cell represents the probability that a possession in that zone
# will eventually lead to a goal. Values increase toward the opponent's goal.
# Columns: 0 (own goal line) -> 11 (opponent goal line)
# Rows: 0 (left touchline) -> 7 (right touchline)
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
    """
    Look up the xT value for a given (x, y) coordinate on the StatsBomb pitch.
    StatsBomb pitch: x in [0, 120], y in [0, 80].
    Grid: 12 columns (x-axis), 8 rows (y-axis).
    """
    col = int(np.clip(x / 10.0, 0, 11))    # 120 / 12 = 10m per column
    row = int(np.clip(y / 10.0, 0, 7))      # 80 / 8 = 10m per row
    return XT_GRID[row, col]

def compute_cumulative_xt(passes_df):
    """
    Computes cumulative xT gained from all passes.
    xT_gained = xT(end_zone) - xT(start_zone), only counted if positive.
    This gives every window a continuous, non-zero value.
    """
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
    """
    Computes all tactical metrics for a single team in a single window.
    
    Returns 5 regression targets + 1 classification target (defensive posture).
    Offensive style classification is deferred to a second pass (see get_rolling_windows).
    """
    # 1. Split Data into Team vs Opponent
    t_passes = passes[passes['team'] == team_name]
    opp_passes = passes[passes['team'] != team_name]
    
    t_shots = shots[shots['team'] == team_name]
    
    opp_def = defense[defense['team'] != team_name]
    
    # --- METRIC 1: CUMULATIVE xT (Offensive Threat) ---
    cum_xt = compute_cumulative_xt(t_passes)
    
    # --- METRIC 2: DEFENSIVE INTENSITY (Opponent Press Height) ---
    if not opp_def.empty:
        avg_press_height = opp_def['x'].mean()
    else:
        avg_press_height = 50.0 
        
    # --- METRIC 3: TERRITORIAL DOMINANCE (Field Tilt) ---
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

    # --- METRIC 5: TEMPO (Passes per Minute) ---
    team_pass_count = len(t_passes)
    tempo = team_pass_count / window_duration if window_duration > 0 else 0.0
    
    # --- CLASSIFICATION 1: DEFENSIVE POSTURE ---
    # Adjusted: Lowered High Press boundary from 70 -> 65 to improve 
    # class balance (more High Press samples for the model to learn from).
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
    TWO-PASS CLASSIFICATION: Assigns offensive style labels based on the
    actual data distribution using percentile-based thresholds.
    
    This guarantees roughly balanced classes (~33% each) regardless of
    what the raw verticality/tempo distributions look like.
    
    Steps:
      1. Normalize verticality and tempo to [0, 1] across all windows
      2. Compute composite 'directness' = 0.5 * norm_vert + 0.5 * norm_tempo
      3. Find 33rd and 67th percentile of this score
      4. Classify: bottom third = Patient, middle = Balanced, top = Counter
    
    Labels:
      0 = Patient Build-Up  (slow, lateral/backward passing)
      1 = Balanced           (moderate directness and tempo)
      2 = Counter / Direct   (fast, forward passing)
    """
    if not windows:
        return windows
    
    # 1. Extract raw values
    verticalities = np.array([w['y_verticality'] for w in windows])
    tempos = np.array([w['y_tempo'] for w in windows])
    
    # 2. Normalize each to [0, 1] range across all windows
    vert_min, vert_max = verticalities.min(), verticalities.max()
    tempo_min, tempo_max = tempos.min(), tempos.max()
    
    vert_range = vert_max - vert_min if vert_max != vert_min else 1.0
    tempo_range = tempo_max - tempo_min if tempo_max != tempo_min else 1.0
    
    norm_vert = (verticalities - vert_min) / vert_range
    norm_tempo = (tempos - tempo_min) / tempo_range
    
    # 3. Composite directness score (equal weight to both dimensions)
    directness = 0.5 * norm_vert + 0.5 * norm_tempo
    
    # 4. Percentile-based thresholds (terciles → ~33% each class)
    p33 = np.percentile(directness, 33.33)
    p67 = np.percentile(directness, 66.67)
    
    # 5. Assign labels
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
    Slices match into overlapping windows and computes tactical labels 
    for each team separately.
    
    TWO-PASS APPROACH:
      Pass 1: Compute all regression metrics and defensive posture labels.
      Pass 2: Derive offensive style labels from the full distribution
              using percentile-based thresholds (guarantees balanced classes).
    
    Each window bundle contains:
      - 5 regression targets (cum_xt, press_height, field_tilt, verticality, tempo)
      - 2 classification targets (def_posture, off_style)
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
    
    # =====================================================================
    # PASS 1: Compute all regression metrics + defensive posture
    # =====================================================================
    while start_time < match_duration:
        end_time = start_time + window_size
        
        # --- Slice Data (Temporal Slice) ---
        pass_win = passes_df[(passes_df['time_min'] >= start_time) & (passes_df['time_min'] < end_time)]
        shot_win = shots_df[(shots_df['time_min'] >= start_time) & (shots_df['time_min'] < end_time)]
        def_win = defense_df[(defense_df['time_min'] >= start_time) & (defense_df['time_min'] < end_time)]
        
        if not pass_win.empty:
            
            # --- GLOBAL STATS ---
            total_passes_in_window = len(pass_win)
            duration = window_size
            
            # Counterpress count from opponent defense
            opp_counterpress = {}
            for team in team_list:
                opp_def_win = def_win[def_win['team'] != team]
                if 'counterpress' in opp_def_win.columns:
                    opp_counterpress[team] = opp_def_win['counterpress'].sum()
                else:
                    opp_counterpress[team] = 0
            
            # Determine period
            window_midpoint = (start_time + end_time) / 2.0
            if 'period' in pass_win.columns:
                period = pass_win['period'].mode().iloc[0] if not pass_win['period'].mode().empty else 1
            else:
                period = 1 if window_midpoint < 45 else 2
            
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
                        
                        # RAW DATA
                        'passes': team_passes, 
                        'shots': team_shots,
                        'opp_defense': metrics['_opp_def'],
                        
                        # Global context data for graph_builder
                        'total_passes_in_window': total_passes_in_window,
                        'opp_counterpress_count': opp_counterpress[team],
                        'period': period,
                        
                        # --- REGRESSION LABELS (5) ---
                        'y_cum_xt': metrics['y_cum_xt'],
                        'y_press_height': metrics['y_press_height'],
                        'y_field_tilt': metrics['y_field_tilt'],
                        'y_verticality': metrics['y_verticality'],
                        'y_tempo': metrics['y_tempo'],
                        
                        # --- CLASSIFICATION LABELS ---
                        'y_def_posture': metrics['y_def_posture'],  # 0/1/2
                        # y_off_style assigned in Pass 2
                    }
                    windows.append(window_bundle)
            
        start_time += stride
        window_id += 1
    
    # =====================================================================
    # PASS 2: Assign offensive style from full distribution (balanced)
    # =====================================================================
    windows = assign_offensive_style_labels(windows)
        
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
    
    # --- Distribution Check ---
    xt_vals = [w['y_cum_xt'] for w in windows]
    tempo_vals = [w['y_tempo'] for w in windows]
    vert_vals = [w['y_verticality'] for w in windows]
    def_labels = [w['y_def_posture'] for w in windows]
    off_labels = [w['y_off_style'] for w in windows]
    
    print(f"\n--- Target Distribution Check ---")
    print(f"Cumulative xT: min={min(xt_vals):.4f}, max={max(xt_vals):.4f}, mean={np.mean(xt_vals):.4f}")
    print(f"Tempo (p/min):  min={min(tempo_vals):.1f}, max={max(tempo_vals):.1f}, mean={np.mean(tempo_vals):.1f}")
    print(f"Verticality:    min={min(vert_vals):.3f}, max={max(vert_vals):.3f}, mean={np.mean(vert_vals):.3f}")
    
    print(f"\n--- Classification Distribution ---")
    print(f"Def Posture:    {dict(pd.Series(def_labels).value_counts().sort_index())}")
    print(f"                (0=LowBlock, 1=MidBlock, 2=HighPress)")
    print(f"Off Style:      {dict(pd.Series(off_labels).value_counts().sort_index())}")
    print(f"                (0=Patient, 1=Balanced, 2=Counter)")
    
    # Verify balance
    total = len(off_labels)
    for cls_val, cls_name in [(0, 'Patient'), (1, 'Balanced'), (2, 'Counter')]:
        count = off_labels.count(cls_val)
        print(f"  {cls_name}: {count} ({count/total:.1%})")
    
    if len(windows) > 10:
        w = windows[10] 
        print(f"\n--- Sample Window ID {w['window_id']} ({w['team_name']}) ---")
        print(f"Time: {w['start_time']} - {w['end_time']} min")
        print(f"--- REGRESSION LABELS ---")
        print(f"Cumulative xT:  {w['y_cum_xt']:.4f}")
        print(f"Press Height:   {w['y_press_height']:.1f}")
        print(f"Field Tilt:     {w['y_field_tilt']:.1%}")
        print(f"Verticality:    {w['y_verticality']:.3f}")
        print(f"Tempo:          {w['y_tempo']:.1f} passes/min")
        print(f"--- CLASSIFICATION LABELS ---")
        print(f"Def Posture:    {w['y_def_posture']} ({'LowBlock' if w['y_def_posture']==0 else 'MidBlock' if w['y_def_posture']==1 else 'HighPress'})")
        print(f"Off Style:      {w['y_off_style']} ({'Patient' if w['y_off_style']==0 else 'Balanced' if w['y_off_style']==1 else 'Counter'})")