import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from statsbombpy import sb

# Import your existing modules
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph
from model import PitchPulseGAT

# --- 1. Helper: The Judge (Threat Score) ---
def calculate_threat_label(window_bundle):
    """
    Calculates the composite threat score:
    Progression Distance + (Shots * 20) + (Goals * 50)
    """
    passes = window_bundle['passes']
    shots = window_bundle['shots']
    GOAL_X, GOAL_Y = 120.0, 40.0
    score = 0.0
    
    # Physics: Progression
    for _, row in passes.iterrows():
        dist_start = np.sqrt((GOAL_X - row['x'])**2 + (GOAL_Y - row['y'])**2)
        dist_end = np.sqrt((GOAL_X - row['end_x'])**2 + (GOAL_Y - row['end_y'])**2)
        progression = dist_start - dist_end
        
        # Only reward positive progression
        if progression > 0:
            score += progression
            
    # Outcome: Shots/Goals
    num_shots = len(shots)
    num_goals = len(shots[shots['is_goal'] == True])
    score += (num_shots * 20.0) + (num_goals * 50.0)
    
    # Normalize (Divide by 100 to keep loss numbers manageable)
    return torch.tensor([score / 100.0], dtype=torch.float)

# --- 2. Bulk Data Processor ---
def process_match(match_id, device):
    """
    Downloads and converts ONE match into a list of (Graph, Label) tuples.
    Includes error handling to skip broken matches.
    """
    dataset = []
    try:
        # A. Fetch
        data = fetch_match_data(match_id)
        # B. Slice
        windows = get_rolling_windows(data)
        
        # C. Convert to Graphs
        for window in windows:
            try:
                # Build Graph
                graph_data = build_graph(window['passes'])
                
                # Check for valid graph (needs nodes)
                if graph_data.num_nodes == 0:
                    continue
                    
                graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
                graph_data = graph_data.to(device)
                
                # Build Label
                label = calculate_threat_label(window).to(device)
                
                dataset.append((graph_data, label))
            except Exception:
                continue
                
        return dataset
        
    except Exception as e:
        print(f"    [!] Error processing Match {match_id}: {e}")
        return []

# --- 3. Training Loop Helpers ---
def train_one_epoch(dataset, model, optimizer, criterion):
    model.train()
    total_loss = 0
    random.shuffle(dataset) # Shuffle to prevent order bias
    
    for graph_data, label in dataset:
        optimizer.zero_grad()
        prediction = model(graph_data)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataset)

def evaluate(dataset, model, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph_data, label in dataset:
            prediction = model(graph_data)
            loss = criterion(prediction, label)
            total_loss += loss.item()
    return total_loss / len(dataset)

# --- Main Execution ---
if __name__ == "__main__":
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # --- PHASE 1: GET MATCH LIST ---
    print("\n--- Phase 1: Fetching World Cup Match List ---")
    # Competition 43 = World Cup, Season 3 = 2018
    matches_df = sb.matches(competition_id=43, season_id=3)
    
    # Get all IDs
    all_match_ids = matches_df['match_id'].tolist()
    
    # LIMIT TO 15 MATCHES for the pilot run
    match_ids = all_match_ids[:]
    
    print(f"Found {len(all_match_ids)} matches total.")
    print(f"Selecting the first {len(match_ids)} for this training run.")
    
    # --- PHASE 2: BULK PROCESSING ---
    print("\n--- Phase 2: Processing Matches (This may take 1-2 mins) ---")
    full_dataset = []
    start_time = time.time()
    
    count = 0
    for mid in match_ids:
        count += 1
        print(f"Processing {count}/{len(match_ids)}: ID {mid}...", end="\r")
        match_data = process_match(mid, device)
        full_dataset.extend(match_data)
        
    print(f"\n\nProcessing Complete! Total Windows: {len(full_dataset)}")
    print(f"Time Taken: {time.time() - start_time:.2f} seconds")
    
    # --- PHASE 3: TRAIN/TEST SPLIT ---
    # 80% Train, 20% Test
    if len(full_dataset) > 0:
        split_idx = int(len(full_dataset) * 0.8)
        random.shuffle(full_dataset) # Critical: Shuffle so we don't test on just one team
        
        train_set = full_dataset[:split_idx]
        test_set = full_dataset[split_idx:]
        
        print(f"Training Samples: {len(train_set)}")
        print(f"Testing Samples:  {len(test_set)}")
        
        # --- PHASE 4: TRAINING ---
        print("\n--- Phase 4: Training GAT ---")
        model = PitchPulseGAT().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for larger dataset
        criterion = nn.MSELoss()
        
        # Run 15 Epochs
        for epoch in range(1, 16): 
            train_loss = train_one_epoch(train_set, model, optimizer, criterion)
            test_loss = evaluate(test_set, model, criterion)
            
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            
        print("\n--- Training Complete! ---")
        
        # --- PHASE 5: SANITY CHECK ---
        # Find the highest predicted threat in the TEST set
        model.eval()
        best_pred = -100
        best_actual = -100
        
        for graph, label in test_set:
            pred = model(graph).item()
            if pred > best_pred:
                best_pred = pred
                best_actual = label.item()
                
        print(f"\nTop Threat in Test Set -> Predicted: {best_pred:.4f} / Actual: {best_actual:.4f}")
        
    else:
        print("\n[!] No data was processed. Check your internet connection or match IDs.")