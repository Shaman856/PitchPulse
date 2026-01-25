import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- IMPORTS ---
from dataset import TacticalDataset
from model import TacticalGAT

# --- CONFIGURATION ---
DATASET_PATH = "./data_v2" 
DATASET_NAME = "international_mix"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0 # NEW: Prevents exploding gradients from the 20x xG weight

# --- WEIGHTED LOSS CONFIGURATION ---
# Index 0: xG (Weight 20.0) - The most critical metric
# Index 1: Press Height (Weight 1.0)
# Index 2: Field Tilt (Weight 1.0)
# Index 3: Verticality (Weight 1.5)
LOSS_WEIGHTS = torch.tensor([20.0, 1.0, 1.0, 1.5]).to(DEVICE)

def weighted_mse_loss(input, target, weights):
    """
    Custom Loss: (Prediction - Target)^2 * Weight
    """
    loss = (input - target) ** 2
    weighted_loss = loss * weights
    return weighted_loss.mean()

def train():
    print(f"--- STARTING TRAINING ON {DEVICE} ---")
    
    # 1. Load Data
    print("Loading Dataset...")
    # NOTE: Ensure window_size/stride match what you generated in dataset.py
    dataset = TacticalDataset(root=DATASET_PATH, competitions=[], dataset_name=DATASET_NAME, window_size=5, stride=1)
    
    # 2. Split (80% Train, 20% Test)
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")
    
    # 3. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    model = TacticalGAT(num_node_features=3, num_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Metrics Tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in loop:
            batch = batch.to(DEVICE)
            
            # Forward
            out = model(batch)
            
            # FIX 1: Robust Reshaping
            # Ensure target is exactly [Batch_Size, 4] regardless of input oddities
            target = batch.y.view(-1, 4)
            
            # Calculate Loss
            loss = weighted_mse_loss(out, target, LOSS_WEIGHTS)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # FIX 2: Gradient Clipping
            # Prevents the "Exploding Gradient" from the high xG weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                
                # Apply same Shape Safety in validation
                target = batch.y.view(-1, 4)
                
                loss = weighted_mse_loss(out, target, LOSS_WEIGHTS)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> New Best Model Saved!")

    # --- PLOT RESULTS ---
    print("\nTraining Complete. Plotting Loss Curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Tactical GAT Training (Weighted xG={LOSS_WEIGHTS[0].item()})')
    plt.xlabel('Epochs')
    plt.ylabel('Weighted MSE Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    train()