import torch
import torch.nn.functional as F
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
DATASET_NAME = "offline_mix_v4"   # v4: balanced offensive style classes + def posture threshold fix
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230  # Set to None for all matches
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0

# --- LOSS CONFIGURATION ---
# Regression weights: [cum_xT, press_height, field_tilt, verticality, tempo]
# xT gets higher weight since it's the primary offensive metric.
# Tempo is new and well-distributed, moderate weight.
REG_WEIGHTS = torch.tensor([2.5, 1.0, 1.5, 1.5, 1.5]).to(DEVICE)

# Classification loss scaling relative to regression loss
CLS_WEIGHT_DEF = 0.5    # Defensive posture classification weight
CLS_WEIGHT_OFF = 1.0    # Offensive style — raised from 0.5 since classes are now balanced


def weighted_mse_loss(pred, target, weights):
    """Per-column weighted MSE loss."""
    loss = (pred - target) ** 2
    return (loss * weights).mean()


def composite_loss(reg_pred, reg_target, cls_def_pred, cls_def_target, 
                   cls_off_pred, cls_off_target):
    """
    Combined loss for the Tactical Suite:
      L = weighted_MSE(regression) 
        + w_def * CrossEntropy(def_posture) 
        + w_off * CrossEntropy(off_style)
    """
    # 1. Regression Loss (5 targets)
    loss_reg = weighted_mse_loss(reg_pred, reg_target, REG_WEIGHTS)
    
    # 2. Classification Losses
    loss_def = F.cross_entropy(cls_def_pred, cls_def_target)
    loss_off = F.cross_entropy(cls_off_pred, cls_off_target)
    
    total = loss_reg + CLS_WEIGHT_DEF * loss_def + CLS_WEIGHT_OFF * loss_off
    
    return total, loss_reg.item(), loss_def.item(), loss_off.item()


def train():
    print(f"--- STARTING TRAINING ON {DEVICE} ---")
    print(f"    Regression targets: 5 (xT, PressHeight, Tilt, Vert, Tempo)")
    print(f"    Classification targets: 2 (DefPosture[3], OffStyle[3])")
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME, 
        window_size=5, stride=1, max_matches=MAX_MATCHES
    )
    
    # 2. Split (80% Train, 20% Test)
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")
    
    # 3. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model (auto-detect dimensions)
    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    global_dim = sample.u.shape[1]
    
    print(f"Detected: {num_node_features} node features, {edge_dim} edge features, {global_dim} global features")
    
    model = TacticalGAT(
        num_node_features=num_node_features, 
        num_reg_targets=5, 
        num_def_classes=3,
        num_off_classes=3,
        edge_dim=edge_dim,
        global_dim=global_dim
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Metrics Tracking
    train_losses = []
    val_losses = []
    val_reg_losses = []
    val_def_accs = []
    val_off_accs = []
    best_val_loss = float('inf')
    
    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in loop:
            batch = batch.to(DEVICE)
            
            # Forward (3 outputs)
            reg_pred, cls_def_pred, cls_off_pred = model(batch)
            
            # Targets
            reg_target = batch.y.view(-1, 5)
            cls_targets = batch.y_cls.view(-1, 2)
            cls_def_target = cls_targets[:, 0]   # [B]
            cls_off_target = cls_targets[:, 1]   # [B]
            
            # Composite Loss
            loss, _, _, _ = composite_loss(
                reg_pred, reg_target, 
                cls_def_pred, cls_def_target,
                cls_off_pred, cls_off_target
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        total_val_reg = 0
        correct_def = 0
        correct_off = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                reg_pred, cls_def_pred, cls_off_pred = model(batch)
                
                reg_target = batch.y.view(-1, 5)
                cls_targets = batch.y_cls.view(-1, 2)
                cls_def_target = cls_targets[:, 0]
                cls_off_target = cls_targets[:, 1]
                
                loss, reg_l, def_l, off_l = composite_loss(
                    reg_pred, reg_target,
                    cls_def_pred, cls_def_target,
                    cls_off_pred, cls_off_target
                )
                
                total_val_loss += loss.item()
                total_val_reg += reg_l
                
                # Classification accuracy
                correct_def += (cls_def_pred.argmax(dim=1) == cls_def_target).sum().item()
                correct_off += (cls_off_pred.argmax(dim=1) == cls_off_target).sum().item()
                total_samples += cls_def_target.size(0)
                
        avg_val_loss = total_val_loss / len(test_loader)
        avg_val_reg = total_val_reg / len(test_loader)
        def_acc = correct_def / total_samples if total_samples > 0 else 0.0
        off_acc = correct_off / total_samples if total_samples > 0 else 0.0
        
        val_losses.append(avg_val_loss)
        val_reg_losses.append(avg_val_reg)
        val_def_accs.append(def_acc)
        val_off_accs.append(off_acc)
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"Reg: {avg_val_reg:.4f} | DefAcc: {def_acc:.1%} | OffAcc: {off_acc:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> New Best Model Saved!")

    # --- PLOT RESULTS ---
    print("\nTraining Complete. Plotting...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Total Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Total Composite Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Regression Loss Only
    axes[1].plot(val_reg_losses, label='Val Regression Loss', color='green')
    axes[1].set_title('Regression Loss (Val)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Weighted MSE')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Classification Accuracy
    axes[2].plot(val_def_accs, label='Def Posture Acc', color='red')
    axes[2].plot(val_off_accs, label='Off Style Acc', color='blue')
    axes[2].set_title('Classification Accuracy (Val)')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    print("Saved training_curve.png")
    plt.show()

if __name__ == "__main__":
    train()