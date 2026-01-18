import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_pipeline import fetch_match_data
from window_slicer import get_rolling_windows
from graph_builder import build_graph
from model import PitchPulseGAT

# --- 1. Helper: Calculate the Target (y) ---
def calculate_progress_label(window_df):
    """
    Calculates a proxy for xT: 'Cumulative Pass Progression'.
    How much closer did the team get to the goal (120, 40) in this window?
    """
    goal_x, goal_y = 120.0, 40.0
    total_progress = 0.0
    
    for _, row in window_df.iterrows():
        # Distance from start of pass to goal
        dist_start = np.sqrt((goal_x - row['x'])**2 + (goal_y - row['y'])**2)
        
        # Distance from end of pass (recipient) to goal
        # Note: We need end_x/end_y. StatsBomb stores this in 'pass_end_location' usually,
        # but for this simple proxy, let's assume successful passes reduce distance.
        # We'll rely on the fact that we filtered for successful passes or use a simplified metric.
        
        # SIMPLIFIED PROXY:
        # Just summing up "Forward Movement" (End X - Start X) for now
        # Ideally, you'd extract 'pass_end_location' in data_pipeline.py
        # But let's use the graph structure: 
        # The model predicts the POTENTIAL of the shape.
        
        # Let's use a dummy random target for the FIRST run to test the loop,
        # OR better: Use the number of passes into the final third (x > 80).
        if row['x'] > 80: 
            total_progress += 1.0
            
    # Normalize: A score of 10.0 is "Very High Threat"
    return torch.tensor([total_progress], dtype=torch.float)

# --- 2. The Training Function ---
def train_one_match(match_id, model, optimizer, criterion, device):
    # A. Get Data
    df = fetch_match_data(match_id)
    windows = get_rolling_windows(df)
    
    total_loss = 0
    model.train() # Set model to training mode
    
    # B. Loop through windows
    for window in windows:
        # 1. Build Graph (Input X)
        data = build_graph(window)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long) # Batch vector
        data = data.to(device)
        
        # 2. Calculate Label (Target y)
        # We want the model to predict how dangerous this window WAS.
        label = calculate_progress_label(window).to(device)
        
        # 3. Forward Pass
        optimizer.zero_grad() # Reset gradients
        prediction = model(data)
        
        # 4. Calculate Loss (Error)
        loss = criterion(prediction, label)
        
        # 5. Backpropagation (Learn)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(windows)

# --- Main Execution ---
if __name__ == "__main__":
    # Setup Device (RTX 3060)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Initialize Model
    model = PitchPulseGAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Learning Rate
    criterion = nn.MSELoss() # Mean Squared Error (Standard for Regression)
    
    # Training Loop (Over 1 Match for demonstration)
    # In reality, you'd loop over your 3,464 matches here [cite: 51]
    match_id = 8658
    
    print(f"\n--- Starting Training on Match {match_id} ---")
    for epoch in range(1, 11): # Run 10 Epochs
        avg_loss = train_one_match(match_id, model, optimizer, criterion, device)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
    print("\n--- Training Complete! ---")
    
    # Test Prediction
    print("Testing on a sample window...")
    model.eval()
    dummy_data = build_graph(get_rolling_windows(fetch_match_data(match_id))[15]).to(device)
    dummy_data.batch = torch.zeros(dummy_data.num_nodes, dtype=torch.long).to(device)
    pred = model(dummy_data)
    print(f"Predicted Threat Score: {pred.item():.4f}")