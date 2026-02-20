import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- IMPORTS ---
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT

# --- CONFIGURATION ---
DATASET_PATH = "./data_v3" 
DATASET_NAME = "offline_mix"
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230  # Set to None for all matches
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0 # Prevents exploding gradients

# --- WEIGHTED LOSS CONFIGURATION ---
# All targets now on [0, 1] scale with reasonable distributions.
# Threat Rate replaces xG — it's continuous and well-distributed,
# so it no longer needs heavy upweighting.
# Index 0: Threat Rate (Weight 2.0)
# Index 1: Press Height (Weight 1.0)
# Index 2: Field Tilt (Weight 1.5) - Slightly upweighted since hardest metric
# Index 3: Verticality (Weight 1.5)
LOSS_WEIGHTS = torch.tensor([2.0, 1.0, 1.5, 1.5]).to(DEVICE)

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
    dataset = TacticalDataset(root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME, window_size=5, stride=1, max_matches=MAX_MATCHES)
    
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
    # AUTO-DETECT all dimensions from the dataset
    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    global_dim = sample.u.shape[1]
    
    print(f"Detected: {num_node_features} node features, {edge_dim} edge features, {global_dim} global features")
    
    model = TacticalGAT(
        num_node_features=num_node_features, 
        num_classes=4, 
        edge_dim=edge_dim,
        global_dim=global_dim
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler — cosine decay lets the model fine-tune in later epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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
            
            # Robust Reshaping
            target = batch.y.view(-1, 4)
            
            # Calculate Loss
            loss = weighted_mse_loss(out, target, LOSS_WEIGHTS)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
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
                
                target = batch.y.view(-1, 4)
                
                loss = weighted_mse_loss(out, target, LOSS_WEIGHTS)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Step the scheduler
        scheduler.step()
        
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
    plt.title(f'Tactical GAT Training (Threat={LOSS_WEIGHTS[0].item()}, Tilt={LOSS_WEIGHTS[2].item()})')
    plt.xlabel('Epochs')
    plt.ylabel('Weighted MSE Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    train()