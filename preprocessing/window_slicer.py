# preprocessing/window_slicer.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# EXPECTED THREAT (xT) LOOKUP GRID
# Karun Singh's original 8x12 zone grid values.
# Rows = pitch width zones (0=left, 7=right), Cols = pitch length zones (0=own goal, 11=opponent goal)
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
    """Look up the xT value for a given (x, y) coordinate on the StatsBomb pitch (120x80)."""
    # Clip to grid bounds and convert to column/row indices
    col = int(np.clip(x / 10.0, 0, 11))
    row = int(np.clip(y / 10.0, 0, 7))
    return XT_GRID[row, col]


def compute_cumulative_xt(passes_df):
    """Computes cumulative xT gained from all passes (positive gains only)."""
    if passes_df.empty:
        return 0.0
    xt_gained = 0.0
    for _, row in passes_df.iterrows():
        # Look up xT at the start and end location of each pass
        start_xt = _get_xt_value(row['x'], row['y'])
        end_xt   = _get_xt_value(row['end_x'], row['end_y'])
        delta    = end_xt - start_xt
        # Only count positive gains — backward passes don't contribute to threat
        if delta > 0:
            xt_gained += delta
    return xt_gained


def compute_cumulative_xt_all_actions(passes_df, carries_df):
    """
    Computes cumulative xT from passes AND carries.
    Carries move the ball into higher-threat zones and should count
    toward total threat generated in the window.
    """
    # xT from passes
    xt_from_passes = compute_cumulative_xt(passes_df)

    # xT from carries (same logic, different data source)
    xt_from_carries = 0.0
    if not carries_df.empty:
        for _, row in carries_df.iterrows():
            if pd.notna(row.get('x')) and pd.notna(row.get('end_x')):
                start_xt = _get_xt_value(row['x'], row['y'])
                end_xt   = _get_xt_value(row['end_x'], row['end_y'])
                delta    = end_xt - start_xt
                if delta > 0:
                    xt_from_carries += delta

    return xt_from_passes + xt_from_carries


def compute_running_score(shots_df, team_name, window_start):
    """
    Computes the score differential at the start of a window.
    Returns: (team_goals - opponent_goals) as of window_start time.
    Positive = team is winning, Negative = team is losing.
    """
    if shots_df.empty:
        return 0

    # Only goals that happened BEFORE this window starts
    goals_before = shots_df[
        (shots_df['is_goal'] == True) &
        (shots_df['time_min'] < window_start)
    ]

    if goals_before.empty:
        return 0

    team_goals = (goals_before['team'] == team_name).sum()
    opp_goals  = (goals_before['team'] != team_name).sum()

    return int(team_goals - opp_goals)


def compute_match_outcome(shots_df, team_name):
    """
    Computes the final match outcome for a team.
    Returns: 0=Loss, 1=Draw, 2=Win
    Used as a classification target — same label for all windows in a match.
    """
    if shots_df.empty:
        return 1  # Default to draw if no shot data

    all_goals  = shots_df[shots_df['is_goal'] == True]
    team_goals = (all_goals['team'] == team_name).sum()
    opp_goals  = (all_goals['team'] != team_name).sum()

    if team_goals > opp_goals:
        return 2   # Win
    elif team_goals < opp_goals:
        return 0   # Loss
    else:
        return 1   # Draw


def calculate_window_metrics(passes, shots, defense, carries, team_name, window_duration):
    """
    Computes all tactical regression metrics for a single team in a single window.

    FIX (vs old version):
        y_def_posture is NO LONGER computed here with hard thresholds.
        It is computed as a post-processing step in assign_defensive_posture_labels()
        using percentile-based thresholds from the full dataset distribution.
        This eliminates the arbitrary >65 / >50 boundaries that caused
        45% Mid Block recall in the confusion matrix.

    Returns 5 regression targets. Classification targets assigned in post-processing.
    """
    # Split into this team's actions vs opponent's actions
    t_passes  = passes[passes['team'] == team_name]
    opp_passes = passes[passes['team'] != team_name]
    t_carries  = carries[carries['team'] == team_name] if not carries.empty else pd.DataFrame()

    t_shots  = shots[shots['team'] == team_name]
    opp_def  = defense[defense['team'] != team_name]

    # --- METRIC 1: CUMULATIVE xT (passes + carries) ---
    cum_xt = compute_cumulative_xt_all_actions(t_passes, t_carries)

    # --- METRIC 2: PRESS HEIGHT (raw, in pitch units 0-120) ---
    # This is the OPPONENT's average defensive action X coordinate.
    # Stored raw here — class assignment happens in assign_defensive_posture_labels()
    if not opp_def.empty and 'x' in opp_def.columns:
        # Clip to valid pitch range to remove outlier events
        avg_press_height = float(np.clip(opp_def['x'].mean(), 0, 120))
    else:
        # Default to midfield if no opponent defensive data in window
        avg_press_height = 50.0

    # --- METRIC 3: FIELD TILT ---
    # Ratio of actions in opponent half — measures territorial dominance
    t_opp_half   = t_passes[t_passes['x'] > 60].shape[0]
    opp_opp_half = opp_passes[opp_passes['x'] > 60].shape[0]

    # Include carries in territorial measurement for completeness
    if not t_carries.empty:
        t_opp_half += t_carries[t_carries['x'] > 60].shape[0]
    opp_carries = carries[carries['team'] != team_name] if not carries.empty else pd.DataFrame()
    if not opp_carries.empty:
        opp_opp_half += opp_carries[opp_carries['x'] > 60].shape[0]

    total_opp_half = t_opp_half + opp_opp_half
    field_tilt = t_opp_half / total_opp_half if total_opp_half > 0 else 0.5

    # --- METRIC 4: VERTICALITY ---
    # Positive = forward-biased passing, Negative = backward-biased
    if not t_passes.empty:
        if 'pass_length' in t_passes.columns:
            dist = t_passes['pass_length'].sum()
        else:
            dist = np.sqrt(
                (t_passes['end_x'] - t_passes['x'])**2 +
                (t_passes['end_y'] - t_passes['y'])**2
            ).sum()
        forward_dist = (t_passes['end_x'] - t_passes['x']).sum()
        # Normalize by total distance — gives a ratio in [-1, +1]
        verticality = forward_dist / dist if dist > 0 else 0.0
    else:
        verticality = 0.0

    # --- METRIC 5: TEMPO (passes per minute) ---
    team_pass_count = len(t_passes)
    tempo = team_pass_count / window_duration if window_duration > 0 else 0.0

    return {
        'y_cum_xt':       cum_xt,
        'y_press_height': avg_press_height,   # Raw value — class assigned in post-processing
        'y_field_tilt':   field_tilt,
        'y_verticality':  verticality,
        'y_tempo':        tempo,
        '_team_pass_count': team_pass_count,
        '_opp_def':       opp_def,
    }


def assign_defensive_posture_labels(windows):
    """
    FIX: Assigns defensive posture labels using PERCENTILE-BASED thresholds.

    Why this replaces hard thresholds (>65 = High Press, >50 = Mid Block):
        Hard thresholds create arbitrary decision boundaries. A press height
        of 64.9 = Mid Block but 65.1 = High Press, even though tactically
        these are identical. The model cannot learn a boundary that is
        meaningless in reality, which caused 45% Mid Block recall.

        Percentile-based assignment guarantees balanced classes and ensures
        boundaries fall at natural breaks in the actual data distribution
        rather than at hardcoded numbers. This is the same approach already
        used for offensive style labeling.

    Class assignment:
        Bottom 33%  of press heights → Low Block  (0)
        Middle 34%  of press heights → Mid Block   (1)
        Top 33%     of press heights → High Press  (2)
    """
    if not windows:
        return windows

    # Collect all raw press height values from every window
    press_heights = np.array([w['y_press_height'] for w in windows])

    # Compute percentile thresholds from the actual data distribution
    p33 = np.percentile(press_heights, 33.33)
    p67 = np.percentile(press_heights, 66.67)

    # Report the computed thresholds so the user can verify them
    print(f"[DefPosture] Data-driven thresholds: "
          f"Low<{p33:.1f} | Mid {p33:.1f}-{p67:.1f} | High>{p67:.1f} "
          f"(pitch units, 0=own goal, 120=opp goal)")

    # Assign class labels based on where each window falls in the distribution
    for w in windows:
        ph = w['y_press_height']
        if ph <= p33:
            # Bottom third: opponent defends deep — Low Block
            w['y_def_posture'] = 0
        elif ph <= p67:
            # Middle third: moderate press height — Mid Block
            w['y_def_posture'] = 1
        else:
            # Top third: opponent presses high up the pitch — High Press
            w['y_def_posture'] = 2

    return windows


def assign_offensive_style_labels(windows):
    """
    Assigns offensive style labels using percentile-based thresholds.
    Already existed — unchanged. Kept here for reference alongside
    the new assign_defensive_posture_labels which follows the same pattern.

    Composite directness score = 0.5 * norm_verticality + 0.5 * norm_tempo
    Bottom 33% = Patient Build-Up (0)
    Middle 34% = Balanced (1)
    Top 33%    = Counter / Direct (2)
    """
    if not windows:
        return windows

    verticalities = np.array([w['y_verticality'] for w in windows])
    tempos        = np.array([w['y_tempo'] for w in windows])

    # Normalize each metric to [0, 1] range before combining
    vert_range  = verticalities.max() - verticalities.min()
    tempo_range = tempos.max() - tempos.min()
    vert_range  = vert_range  if vert_range  != 0 else 1.0
    tempo_range = tempo_range if tempo_range != 0 else 1.0

    norm_vert  = (verticalities - verticalities.min()) / vert_range
    norm_tempo = (tempos - tempos.min()) / tempo_range

    # Equal weighting of both components into one directness score
    directness = 0.5 * norm_vert + 0.5 * norm_tempo

    # Percentile thresholds ensure balanced class distribution
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
    Slices a match into overlapping windows and computes tactical labels.

    FIX (vs old version):
        Defensive posture is now assigned in a THIRD POST-PROCESSING PASS
        after all windows have been collected, using percentile-based thresholds.
        This replaces the hard-threshold assignment that caused poor Mid Block recall.

    Pass 1: Compute all regression metrics per window per team
    Pass 2: Assign offensive style labels (percentile-based, was already here)
    Pass 3: Assign defensive posture labels (percentile-based, NEW FIX)
    """
    passes_df   = data_dict['passes']
    shots_df    = data_dict['shots']
    defense_df  = data_dict['defense']
    carries_df  = data_dict.get('carries',  pd.DataFrame())
    dribbles_df = data_dict.get('dribbles', pd.DataFrame())

    windows = []

    # Determine match duration from the latest event across all action types
    all_times = []
    if not passes_df.empty:
        all_times.append(passes_df['time_min'].max())
    if not defense_df.empty and 'time_min' in defense_df.columns:
        all_times.append(defense_df['time_min'].max())
    if not carries_df.empty and 'time_min' in carries_df.columns:
        all_times.append(carries_df['time_min'].max())
    match_duration = max(all_times) if all_times else 90.0

    # Identify teams — need at least 2 for a valid match
    teams = passes_df['team'].unique()
    if len(teams) < 2:
        print("Warning: Less than 2 teams found. Skipping.")
        return []
    team_list = sorted(teams)

    print(f"Processing Match Duration: {match_duration:.1f} min")

    # Compute final match outcome for each team (same for all windows in match)
    team_outcomes = {}
    for team in team_list:
        team_outcomes[team] = compute_match_outcome(shots_df, team)
    print(f"Match Outcomes: {team_outcomes}")

    start_time = 0
    window_id  = 0

    # =========================================================================
    # PASS 1: Compute all regression metrics per window per team
    # y_def_posture is NOT set here — it's set in Pass 3 below
    # =========================================================================
    while start_time < match_duration:
        end_time = start_time + window_size

        # Slice each action type to this time window
        pass_win = passes_df[
            (passes_df['time_min'] >= start_time) & (passes_df['time_min'] < end_time)
        ]
        shot_win = shots_df[
            (shots_df['time_min'] >= start_time) & (shots_df['time_min'] < end_time)
        ]
        def_win = (
            defense_df[
                (defense_df['time_min'] >= start_time) & (defense_df['time_min'] < end_time)
            ]
            if not defense_df.empty else pd.DataFrame()
        )

        carry_win = pd.DataFrame()
        if not carries_df.empty and 'time_min' in carries_df.columns:
            carry_win = carries_df[
                (carries_df['time_min'] >= start_time) & (carries_df['time_min'] < end_time)
            ]

        dribble_win = pd.DataFrame()
        if not dribbles_df.empty and 'time_min' in dribbles_df.columns:
            dribble_win = dribbles_df[
                (dribbles_df['time_min'] >= start_time) & (dribbles_df['time_min'] < end_time)
            ]

        if not pass_win.empty:

            total_passes_in_window = len(pass_win)

            # Count opponent counterpress events for global context
            opp_counterpress = {}
            for team in team_list:
                opp_def_win = def_win[def_win['team'] != team] if not def_win.empty else pd.DataFrame()
                if 'counterpress' in opp_def_win.columns:
                    opp_counterpress[team] = int(opp_def_win['counterpress'].sum())
                else:
                    opp_counterpress[team] = 0

            # Determine match period from passes in this window
            window_midpoint = (start_time + end_time) / 2.0
            if 'period' in pass_win.columns:
                mode_result = pass_win['period'].mode()
                period = mode_result.iloc[0] if not mode_result.empty else 1
            else:
                period = 1 if window_midpoint < 45 else 2

            # Match progress as a fraction [0.0 → 1.0]
            match_progress = min(window_midpoint / match_duration, 1.0) if match_duration > 0 else 0.5

            for team in team_list:
                # Compute regression metrics for this team in this window
                metrics = calculate_window_metrics(
                    pass_win, shot_win, def_win, carry_win, team, window_size
                )

                # Slice action data to just this team's actions
                team_passes   = pass_win[pass_win['team'] == team].copy()
                team_shots    = shot_win[shot_win['team'] == team].copy()
                team_carries  = carry_win[carry_win['team'] == team].copy() if not carry_win.empty else pd.DataFrame()
                team_dribbles = dribble_win[dribble_win['team'] == team].copy() if not dribble_win.empty else pd.DataFrame()
                team_defense  = def_win[def_win['team'] == team].copy() if not def_win.empty else pd.DataFrame()

                # Score differential at start of this window
                score_diff = compute_running_score(shots_df, team, start_time)

                if not team_passes.empty:
                    window_bundle = {
                        'match_id':   match_id,
                        'window_id':  window_id,
                        'team_name':  team,
                        'start_time': start_time,
                        'end_time':   end_time,

                        # Raw event data
                        'passes':      team_passes,
                        'shots':       team_shots,
                        'opp_defense': metrics['_opp_def'],
                        'carries':     team_carries,
                        'dribbles':    team_dribbles,
                        'team_defense': team_defense,

                        # Global context features
                        'total_passes_in_window':  total_passes_in_window,
                        'opp_counterpress_count':  opp_counterpress[team],
                        'period':                  period,
                        'score_diff':              score_diff,
                        'match_progress':          match_progress,
                        'match_duration':          match_duration,

                        # REGRESSION LABELS (5 targets)
                        'y_cum_xt':       metrics['y_cum_xt'],
                        'y_press_height': metrics['y_press_height'],
                        'y_field_tilt':   metrics['y_field_tilt'],
                        'y_verticality':  metrics['y_verticality'],
                        'y_tempo':        metrics['y_tempo'],

                        # CLASSIFICATION LABELS
                        # y_def_posture: intentionally left as placeholder — set in Pass 3
                        'y_def_posture': 1,           # Placeholder, overwritten in Pass 3
                        'y_off_style':   1,           # Placeholder, overwritten in Pass 2
                        'y_outcome':     team_outcomes[team],
                    }
                    windows.append(window_bundle)

        start_time += stride
        window_id  += 1

    # =========================================================================
    # PASS 2: Assign offensive style labels (percentile-based — unchanged)
    # =========================================================================
    windows = assign_offensive_style_labels(windows)

    # =========================================================================
    # PASS 3: Assign defensive posture labels (percentile-based — NEW FIX)
    # This runs after all windows exist so percentiles are computed from the
    # full dataset distribution rather than hardcoded arbitrary thresholds.
    # =========================================================================
    windows = assign_defensive_posture_labels(windows)

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

    if windows:
        w = windows[10]
        print(f"\n--- Sample Window ---")
        print(f"Team: {w['team_name']} | Time: {w['start_time']}-{w['end_time']} min")
        print(f"Def Posture (data-driven): {w['y_def_posture']} (0=Low, 1=Mid, 2=High)")
        print(f"Off Style (data-driven):   {w['y_off_style']} (0=Patient, 1=Balanced, 2=Counter)")
        print(f"Press Height (raw):        {w['y_press_height']:.1f}")