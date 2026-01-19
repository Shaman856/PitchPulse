import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_rolling_windows(data_dict, window_size=15, stride=3):
    """
    Slices both Passes and Shots into overlapping windows.
    
    Args:
        data_dict: Dictionary containing {'passes': df, 'shots': df}
        window_size: Duration of each window in minutes (Default: 15)
        stride: How much to move the window forward each step (Default: 3)
    """
    passes_df = data_dict['passes']
    shots_df = data_dict['shots']
    
    windows = []
    
    # Get the end of the match based on passes (usually the most reliable clock)
    if not passes_df.empty:
        match_duration = passes_df['time_min'].max()
    else:
        match_duration = 90.0
    
    start_time = 0
    window_id = 0
    
    while start_time < match_duration:
        end_time = start_time + window_size
        
        # 1. Slice Passes (The Input X)
        pass_window = passes_df[
            (passes_df['time_min'] >= start_time) & 
            (passes_df['time_min'] < end_time)
        ].copy()
        
        # 2. Slice Shots/Goals (The Target y)
        # We need to know if a goal happened in this specific window
        shot_window = shots_df[
            (shots_df['time_min'] >= start_time) & 
            (shots_df['time_min'] < end_time)
        ].copy()
        
        # Only keep windows that actually contain activity (Passes)
        # (We don't care if there are no shots, but we need passes to build a graph)
        if not pass_window.empty:
            pass_window['window_id'] = window_id
            
            # Bundle them together
            window_bundle = {
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'passes': pass_window,
                'shots': shot_window
            }
            windows.append(window_bundle)
            
        # Move forward
        start_time += stride
        window_id += 1
        
    return windows

# --- Test Block ---
if __name__ == "__main__":
    from data_pipeline import fetch_match_data
    
    print("Fetching data...")
    # Fetch both passes and shots
    data = fetch_match_data(8658)
    
    print(f"\nSlicing match into {15}-minute windows...")
    windows = get_rolling_windows(data)
    
    print(f"Total Windows Generated: {len(windows)}")
    
    # Verification: Find a window with a goal
    print("\n--- Searching for a window with a Goal ---")
    found_goal = False
    for w in windows:
        goals = w['shots'][w['shots']['is_goal'] == True]
        if not goals.empty:
            print(f"Window {w['window_id']} ({w['start_time']}-{w['end_time']} min):")
            print(f"   - Passes: {len(w['passes'])}")
            print(f"   - Goals: {len(goals)} (Scored by: {goals['player'].values})")
            found_goal = True
            break
            
    if not found_goal:
        print("No goals found in any window (Check data pipeline).")