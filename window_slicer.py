import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_rolling_windows(match_df, window_size=15, stride=3):
    """
    Slices the match into overlapping windows.
    Returns a list of DataFrames.
    
    Args:
        match_df: DataFrame containing all passes from the match
        window_size: Duration of each window in minutes (Default: 15)
        stride: How much to move the window forward each step (Default: 3)
    """
    windows = []
    
    # Get the end of the match (e.g., 95.5 minutes)
    match_duration = match_df['time_min'].max()
    
    # Start slicing
    start_time = 0
    window_id = 0
    
    while start_time < match_duration:
        end_time = start_time + window_size
        
        # Filter data: strictly greater than start, less than end
        # e.g., Window 1: 0-15', Window 2: 3-18'
        window_data = match_df[
            (match_df['time_min'] >= start_time) & 
            (match_df['time_min'] < end_time)
        ].copy()
        
        # Only keep windows that actually contain passes
        if not window_data.empty:
            window_data['window_id'] = window_id
            windows.append(window_data)
            
        # Move the window forward by the stride (3 minutes)
        start_time += stride
        window_id += 1
        
    return windows

# --- Test Block ---
if __name__ == "__main__":
    # Import the data fetcher we just fixed
    from data_pipeline import fetch_match_data
    
    print("Fetching data...")
    # Using 2018 World Cup Final (Match ID: 8658)
    df = fetch_match_data(8658)
    
    print(f"\nSlicing match into {15}-minute windows with {3}-minute stride...")
    windows = get_rolling_windows(df)
    
    print(f"Total Windows Generated: {len(windows)}")
    
    # verification
    if len(windows) > 0:
        first_window = windows[0]
        print(f"\n--- Window 0 (First 15 mins) ---")
        print(f"Passes: {len(first_window)}")
        print(f"Time Range: {first_window['time_min'].min():.2f} - {first_window['time_min'].max():.2f} min")
        
        last_window = windows[-1]
        print(f"\n--- Last Window ---")
        print(f"Passes: {len(last_window)}")
        print(f"Time Range: {last_window['time_min'].min():.2f} - {last_window['time_min'].max():.2f} min")