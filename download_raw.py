import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_DIR = "./data/raw_events"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Define the tournaments you want to download
# Format: (Competition ID, Season ID)
COMPETITIONS = [
    (43, 3),   # World Cup 2018
    (55, 43),  # Euro 2020
    # Add more here...
]

def download_raw_events():
    print(f"--- STARTING RAW DATA DOWNLOAD ---")
    print(f"Saving to: {RAW_DATA_DIR}")
    
    # 1. Get List of All Matches
    all_match_ids = []
    for comp_id, season_id in COMPETITIONS:
        try:
            matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
            ids = matches_df['match_id'].tolist()
            all_match_ids.extend(ids)
            print(f" -> Found {len(ids)} matches for Comp {comp_id}/Season {season_id}")
        except Exception as e:
            print(f" [!] Failed to fetch match list for {comp_id}/{season_id}: {e}")

    # Remove duplicates
    all_match_ids = list(set(all_match_ids))
    print(f"Total Unique Matches: {len(all_match_ids)}")

    # 2. Download and Save
    for match_id in tqdm(all_match_ids, desc="Downloading Events"):
        file_path = os.path.join(RAW_DATA_DIR, f"{match_id}.pkl")
        
        # Skip if already exists (Resume capability)
        if os.path.exists(file_path):
            continue
            
        try:
            # Fetch from API
            events = sb.events(match_id=match_id)
            
            # Save Raw DataFrame to Disk
            events.to_pickle(file_path)
            
        except Exception as e:
            print(f"\n[!] Error downloading {match_id}: {e}")
            continue

    print("\n--- DOWNLOAD COMPLETE ---")
    print(f"Raw files are stored in {RAW_DATA_DIR}")

if __name__ == "__main__":
    download_raw_events()