import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Import your modules
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph
from model import PitchPulseGAT

# --- 1. The "Judge": Composite Threat Score ---
# (Same logic as before, just kept here for completeness)
def calculate_threat_label(window_bundle):
    passes = window_bundle['passes']
    shots = window_bundle['shots']
    
    GOAL_X, GOAL_Y = 120.0, 40.0
    score = 0.0
    
    # A. Progression
    for _, row in passes.iterrows():
        dist_start = np.sqrt((GOAL_X - row['x'])**2 + (GOAL_Y - row['y'])**2)
        dist_end = np.sqrt((GOAL_X - row['end_x'])**2 + (GOAL_Y - row['end_y'])**2)
        progression = dist_start - dist_end
        if progression > 0:
            score += progression
            
    # B. Outcomes
    num_shots = len(shots)
    num_goals = len(shots[shots['is_goal'] == True])
    score += (num_shots * 20.0) + (num_goals * 50.0)
    
    return torch.tensor([score / 100.0], dtype=torch.float)

# --- 2. NEW Helper: Prepare the Dataset in RAM ---
def prepare_dataset(windows, device):
    """
    Converts raw DataFrame windows into a list of (Graph, Label) tuples.
    This runs ONCE before training starts.
    """
    dataset = []
    print(f"   > Pre-processing {len(windows)} windows into Graphs...")
    
    for window in windows:
        try:
            # 1. Build Graph (Input X)
            graph_data = build_graph(window['passes'])
            
            # Prepare Batch Vector (Needed for single graph processing)
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
            
            # Move to GPU immediately (If your VRAM allows - 1 match is tiny, so yes)
            graph_data = graph_data.to(device)
            
            # 2. Build Label (Target y)
            label = calculate_threat_label(window).to(device)
            
            # Store tuple
            dataset.append((graph_data, label))
            
        except ValueError:
            # Skip empty/invalid windows
            continue
            
    print(f"   > Dataset Ready: {len(dataset)} valid training samples.")
    return dataset

# --- 3. Optimized Training Loop ---
def train_one_epoch(dataset, model, optimizer, criterion):
    """
    Iterates over the pre-loaded dataset. No fetching, no building. Just math.
    """
    model.train()
    total_loss = 0
    
    for graph_data, label in dataset:
        optimizer.zero_grad()
        
        # Forward Pass
        prediction = model(graph_data)
        
        # Loss & Backprop
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataset)

# --- Main Execution ---
if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = PitchPulseGAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    match_id = 8658
    
    # --- STEP 1: LOAD & PROCESS DATA (ONCE) ---
    print(f"\n--- Phase 1: Data Preparation ---")
    start_time = time.time()
    
    # A. Fetch
    data = fetch_match_data(match_id)
    
    # B. Slice
    windows = get_rolling_windows(data)
    
    # C. Convert to Tensors (The Level 2 Optimization)
    train_dataset = prepare_dataset(windows, device)
    
    print(f"Data Prep Time: {time.time() - start_time:.2f} seconds")
    
    # --- STEP 2: TRAINING LOOP (FAST) ---
    print(f"\n--- Phase 2: Training Loop ---")
    
    for epoch in range(1, 11):
        # We pass the pre-built 'train_dataset' instead of match_id
        avg_loss = train_one_epoch(train_dataset, model, optimizer, criterion)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")
        
    print("\n--- Training Complete! ---")
    
    # --- STEP 3: TESTING ---
    # We can just grab the goal window from our RAM dataset
    # (Assuming we know the index, usually the high scoring one)
    print("Testing on a High-Threat Sample (from RAM)...")
    model.eval()
    
    # Let's find the sample with the highest label in our dataset
    best_sample = max(train_dataset, key=lambda x: x[1].item())
    
    input_graph, true_label = best_sample
    pred = model(input_graph)
    
    print(f"Actual Score:    {true_label.item():.4f}")
    print(f"Predicted Score: {pred.item():.4f}")