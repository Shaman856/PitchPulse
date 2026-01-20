import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import os
from statsbombpy import sb

# Import your modules
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph
from model2 import PitchPulseGAT

# --- CONFIGURATION ---
# Add more competitions here if you want.
# (Comp ID, Season ID)
# 43=World Cup, 55=Euro, 11=La Liga
TARGET_COMPETITIONS = [
    # --- MEN'S INTERNATIONAL ---
    (43, 106), # World Cup 2022 (64 matches)
    (43, 3),   # World Cup 2018 (64 matches)
    (43, 51),  # World Cup 2014 (64 matches)
    (55, 43),  # Euro 2020 (51 matches)

    # --- MEN'S CLUB (Messi Data) ---
    (11, 90),  # La Liga 20/21 (35 matches)
    (11, 42),  # La Liga 19/20 (33 matches)
    (11, 4),   # La Liga 18/19 (34 matches)
    (11, 1),   # La Liga 17/18 (36 matches)

    # --- WOMEN'S CLUB (FULL SEASONS - The Data Goldmine) ---
    (37, 90),  # FA Women's Super League 2020/2021 (~132 matches)
    (37, 42),  # FA Women's Super League 2019/2020 (~87 matches)
    (37, 4),   # FA Women's Super League 2018/2019 (~108 matches)
    
    # --- CHAMPIONS LEAGUE (High Quality) ---
    (16, 4),   # Champions League 18/19 (13 matches - Finals/Semis)
    (16, 1),   # Champions League 17/18 (13 matches)
]

EPOCHS = 100  # Deep training
SAVE_DIR = "checkpoints"
LOG_FILE = "training_log2.txt"

# --- 1. Helper Functions (Same as before) ---
def log(message):
    """Writes to console AND file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def calculate_threat_label(window_bundle):
    passes = window_bundle['passes']
    shots = window_bundle['shots']
    GOAL_X, GOAL_Y = 120.0, 40.0
    score = 0.0
    
    # Physics
    for _, row in passes.iterrows():
        dist_start = np.sqrt((GOAL_X - row['x'])**2 + (GOAL_Y - row['y'])**2)
        dist_end = np.sqrt((GOAL_X - row['end_x'])**2 + (GOAL_Y - row['end_y'])**2)
        progression = dist_start - dist_end
        if progression > 0:
            score += progression
            
    # Outcomes
    num_shots = len(shots)
    num_goals = len(shots[shots['is_goal'] == True])
    score += (num_shots * 20.0) + (num_goals * 50.0)
    
    return torch.tensor([score / 100.0], dtype=torch.float)

def process_match(match_id, device):
    dataset = []
    try:
        data = fetch_match_data(match_id)
        windows = get_rolling_windows(data)
        for window in windows:
            try:
                graph_data = build_graph(window['passes'])
                if graph_data.num_nodes == 0: continue
                
                graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
                graph_data = graph_data.to(device)
                label = calculate_threat_label(window).to(device)
                dataset.append((graph_data, label))
            except: continue
        return dataset
    except: return []

# --- 2. Main Training Loop ---
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # Clear log file
    with open(LOG_FILE, "w") as f: f.write("Starting Massive Training Run...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # --- PHASE 1: GATHER MATCH IDs ---
    all_match_ids = []
    log("\n--- Phase 1: Fetching Match Lists ---")
    
    for comp_id, season_id in TARGET_COMPETITIONS:
        try:
            matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
            ids = matches_df['match_id'].tolist()
            log(f"Found {len(ids)} matches for Comp {comp_id} / Season {season_id}")
            all_match_ids.extend(ids)
        except Exception as e:
            log(f"Failed to fetch Comp {comp_id}: {e}")
            
    # Remove duplicates
    all_match_ids = list(set(all_match_ids))
    log(f"\nTotal Unique Matches to Process: {len(all_match_ids)}")
    
    # --- PHASE 2: PROCESSING ---
    log("\n--- Phase 2: Bulk Processing (This will take a while) ---")
    full_dataset = []
    start_time = time.time()
    
    for i, mid in enumerate(all_match_ids):
        print(f"Processing {i+1}/{len(all_match_ids)}...", end="\r")
        match_data = process_match(mid, device)
        full_dataset.extend(match_data)
        
        # Periodic update
        if (i+1) % 50 == 0:
            log(f"Processed {i+1} matches... Current Dataset Size: {len(full_dataset)}")

    duration = time.time() - start_time
    log(f"\nProcessing Complete! Total Windows: {len(full_dataset)}")
    log(f"Time Taken: {duration/60:.2f} minutes")
    
    # --- PHASE 3: TRAIN/TEST SPLIT ---
    split_idx = int(len(full_dataset) * 0.85) # 85% Train
    random.shuffle(full_dataset)
    
    train_set = full_dataset[:split_idx]
    test_set = full_dataset[split_idx:]
    
    log(f"Training Samples: {len(train_set)}")
    log(f"Testing Samples:  {len(test_set)}")
    
    # --- PHASE 4: DEEP TRAINING ---
    log(f"\n--- Phase 4: Training for {EPOCHS} Epochs ---")
    model = PitchPulseGAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_test_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        random.shuffle(train_set)
        
        for graph, label in train_set:
            optimizer.zero_grad()
            pred = model(graph)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_set)
        
        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for graph, label in test_set:
                pred = model(graph)
                loss = criterion(pred, label)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_set)
        
        # Logging
        log(f"Epoch {epoch:02d} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")
        
        # Checkpoint: Save Best Model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            log(f"   >>> New Best Model Saved! (Test Loss: {avg_test_loss:.4f})")
            
        # Checkpoint: Periodic Save (Safety)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/checkpoint_epoch_{epoch}.pth")

    log("\n--- MARATHON COMPLETE ---")