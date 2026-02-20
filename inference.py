import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT

# --- CONFIGURATION ---
DATASET_PATH = "./data_v3" 
DATASET_NAME = "offline_mix"
RAW_DATA_DIR = "./data/raw_events"
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize(preds_or_targets):
    """
    Reverses the target normalization applied in graph_builder.py
    so scatter plots show real-world values.
    
    Column mapping:
    [0] Threat Rate:  already [0, 1] -> no change
    [1] Press Height: was /120.0 -> multiply back by 120
    [2] Field Tilt:   already [0, 1] -> no change
    [3] Verticality:  was (v+1)/2 -> reverse: v*2 - 1
    """
    out = preds_or_targets.copy()
    out[:, 1] = out[:, 1] * 120.0       # Press Height -> real coordinate
    out[:, 3] = out[:, 3] * 2.0 - 1.0   # Verticality -> back to [-1, 1]
    return out

def load_data_and_model():
    print("1. Loading Data...")
    dataset = TacticalDataset(root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME, window_size=5, stride=1)
    
    # Simulate the same split
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"2. Loading Model from {MODEL_PATH}...")
    sample = dataset[0]
    model = TacticalGAT(
        num_node_features=sample.x.shape[1], 
        num_classes=4, 
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1]
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    return model, loader

def analyze_predictions(model, loader):
    print("3. Running Inference...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            target = batch.y.view(-1, 4)
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
    # Concatenate all batches (these are in NORMALIZED space)
    preds_norm = np.vstack(all_preds)
    targets_norm = np.vstack(all_targets)
    
    # Denormalize for interpretable reporting and plots
    preds = denormalize(preds_norm)
    targets = denormalize(targets_norm)
    
    # Column Names
    metrics = ['Threat Rate', 'Press Height', 'Field Tilt', 'Verticality']
    
    # --- REPORT 1: Mean Absolute Error (MAE) ---
    print("\n" + "="*40)
    print("   MODEL PERFORMANCE REPORT")
    print("="*40)
    
    mae = np.mean(np.abs(preds - targets), axis=0)
    
    for i, name in enumerate(metrics):
        print(f"{name:15s} | MAE: {mae[i]:.4f}")
        
    print("-" * 40)
    print("INTERPRETATION:")
    print(f"* Threat Rate: Off by {mae[0]:.3f} (0-1 scale, fraction of passes into box).")
    print(f"* Press Height: Off by {mae[1]:.1f} units (out of 120 pitch length).")
    print(f"* Field Tilt: Off by {mae[2]:.3f} (0-1 scale, opponent half dominance).")
    print(f"* Verticality: Off by {mae[3]:.3f} (-1 to 1 scale).")
    print("-" * 40)

    # --- REPORT 2: The "Eye Test" (Random Samples) ---
    print("\n" + "="*40)
    print("   SAMPLE PREDICTIONS (Eye Test)")
    print("="*40)
    
    indices = np.random.choice(len(preds), 5, replace=False)
    
    print(f"{'Metric':<15} | {'Actual':<10} | {'Predicted':<10} | {'Diff':<10}")
    print("-" * 55)
    
    for idx in indices:
        print(f"--- Sample ID {idx} ---")
        for i, name in enumerate(metrics):
            act = targets[idx][i]
            pre = preds[idx][i]
            diff = pre - act
            print(f"{name:<15} | {act:<10.3f} | {pre:<10.3f} | {diff:<+10.3f}")
        print("")

    # --- REPORT 3: Visual Scatter Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if len(preds) > 1000:
            sample_idx = np.random.choice(len(preds), 1000, replace=False)
            x_vals = targets[sample_idx, i]
            y_vals = preds[sample_idx, i]
        else:
            x_vals = targets[:, i]
            y_vals = preds[:, i]
            
        ax.scatter(x_vals, y_vals, alpha=0.3, s=10)
        
        # Perfect prediction line (y=x)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        
        ax.set_title(f"{metrics[i]} Correlation")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("inference_analysis.png")
    print("\nSaved visualization to 'inference_analysis.png'")

if __name__ == "__main__":
    model, loader = load_data_and_model()
    analyze_predictions(model, loader)
