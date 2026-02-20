import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_DIR = "./data/raw_events"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Define the tournaments you want to download
# Format: (Competition ID, Season ID)
#
# IMPORTANT: Run 'python -c "from statsbombpy import sb; print(sb.competitions().to_string())"'
# to verify these IDs are still valid and see all available competitions.
#
# Current setup: ~115 matches (WC 2018 + Euro 2020)
# Expanded setup: ~800+ matches (adds La Liga seasons + WC 2022 + Euro 2024)
#
# La Liga seasons have ~380 matches each and are the richest source of 
# tactical variety (different formations, styles, levels of quality).

COMPETITIONS = [
    # --- INTERNATIONAL TOURNAMENTS ---
    (43, 3),     # FIFA World Cup 2018
    (43, 106),   # FIFA World Cup 2022
    (55, 43),    # UEFA Euro 2020
    (55, 282),   # UEFA Euro 2024
    (223, 282),  # Copa America 2024
]

def download_raw_events():
    print(f"--- STARTING RAW DATA DOWNLOAD ---")
    print(f"Saving to: {RAW_DATA_DIR}")
    print(f"Competitions configured: {len(COMPETITIONS)}")
    
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
            print(f"     This competition may not be available. Run:")
            print(f"     python -c \"from statsbombpy import sb; print(sb.competitions().to_string())\"")
            print(f"     to see all valid competition/season pairs.")

    # Remove duplicates
    all_match_ids = list(set(all_match_ids))
    print(f"\nTotal Unique Matches: {len(all_match_ids)}")
    
    # Count how many already exist (for resume info)
    existing = sum(1 for mid in all_match_ids 
                   if os.path.exists(os.path.join(RAW_DATA_DIR, f"{mid}.pkl")))
    print(f"Already downloaded: {existing}")
    print(f"Remaining: {len(all_match_ids) - existing}")

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

    # 3. Final Summary
    final_count = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pkl')])
    print(f"\n--- DOWNLOAD COMPLETE ---")
    print(f"Total raw files in {RAW_DATA_DIR}: {final_count}")

if __name__ == "__main__":
    download_raw_events()
