from statsbombpy import sb
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def fetch_match_data(match_id):
    """
    Fetches events for a single match and filters for passes.
    """
    print(f"Fetching data for Match ID: {match_id}...")
    events = sb.events(match_id=match_id)
    
    # Filter only 'Pass' events
    # We also need 'player' (source) and 'pass_recipient' (target)
    # 'location' contains [x, y]
    mask = (events['type'] == 'Pass') & (events['location'].notna())
    passes = events.loc[mask, ['timestamp', 'minute', 'second', 'player', 'pass_recipient', 'location', 'team']].copy()
    
    # 1. Flatten Coordinates (Crucial for GNN Features)
    # StatsBomb gives location as [x, y]. We split them.
    passes['x'] = passes['location'].apply(lambda loc: loc[0])
    passes['y'] = passes['location'].apply(lambda loc: loc[1])
    
    # Drop the original list column to save memory
    passes = passes.drop(columns=['location'])
    
    # --- BUG FIX START ---
    # OLD INCORRECT LOGIC: Parsing timestamp string (resets at half-time)
    # def time_to_min(t):
    #     h, m, s = str(t).split(':')
    #     return int(h) * 60 + int(m) + float(s)/60
    # passes['time_min'] = passes['timestamp'].apply(time_to_min)

    # NEW CORRECT LOGIC: Use StatsBomb's explicit minute/second columns
    # This correctly handles the second half (e.g., Minute 46 is actually 46.0)
    passes['time_min'] = passes['minute'] + passes['second'] / 60.0
    # --- BUG FIX END ---
    
    return passes.sort_values('time_min').reset_index(drop=True)

# --- Test Block ---
if __name__ == "__main__":
    # Using the 2018 World Cup Final as a test case (Match ID: 8658)
    df_passes = fetch_match_data(match_id=8658) 
    print(f"\nExtracted {len(df_passes)} passes.")
    
    # Check the head to ensure time_min looks correct (should start near 0)
    print("First 5 passes:")
    print(df_passes[['player', 'minute', 'time_min']].head())
    
    # Check the middle to ensure the fix worked (should show 45+ values, not resetting to 0)
    print("\nPasses around halftime (Verify continuity):")
    halftime_idx = len(df_passes) // 2
    print(df_passes[['player', 'minute', 'time_min']].iloc[halftime_idx:halftime_idx+5])