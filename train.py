import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT


def match_level_split(dataset, train_ratio=0.8, seed=42):
    match_ids = [dataset[i].match_id for i in range(len(dataset))]
    unique_matches = sorted(set(match_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)
    n_train = int(len(unique_matches) * train_ratio)
    train_matches = set(unique_matches[:n_train])
    train_idx = [i for i, m in enumerate(match_ids) if m in train_matches]
    test_idx = [i for i, m in enumerate(match_ids) if m not in train_matches]
    print(f"Match-Level Split: {len(train_matches)} train, {len(unique_matches)-len(train_matches)} test")
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)} windows")
    return train_idx, test_idx


# --- CONFIGURATION ---
DATASET_PATH = "./data_v3" 
DATASET_NAME = "offline_mix_v7_suite"  # Same dataset, no rebuild needed
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 80
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0
PATIENCE = 12
MIN_DELTA = 0.0005

# Loss weights
REG_WEIGHTS = torch.tensor([2.5, 1.0, 1.5, 1.5, 1.5]).to(DEVICE)
CLS_WEIGHT_DEF = 0.5
CLS_WEIGHT_OFF = 1.0
CLS_WEIGHT_OUTCOME = 0.4  # Slightly increased from 0.3

# SUITE v2: Huber loss delta for xT regression
HUBER_DELTA = 0.15  # Reduces sensitivity to extreme high-xT outliers


def compute_outcome_class_weights(dataset, train_indices):
    """
    SUITE v2: Compute inverse-frequency weights for outcome classes.
    Draw is severely underrepresented (~1354 vs ~3500 Win/Loss), 
    so we upweight it to improve recall.
    """
    outcome_counts = Counter()
    for i in train_indices:
        outcome = int(dataset[i].y_cls[0, 2].item())
        outcome_counts[outcome] += 1
    
    total = sum(outcome_counts.values())
    n_classes = 3
    weights = torch.zeros(n_classes)
    
    for c in range(n_classes):
        if outcome_counts[c] > 0:
            # Inverse frequency: rarer classes get higher weight
            weights[c] = total / (n_classes * outcome_counts[c])
    
    print(f"  Outcome class distribution: {dict(outcome_counts)}")
    print(f"  Outcome class weights: Loss={weights[0]:.2f}, Draw={weights[1]:.2f}, Win={weights[2]:.2f}")
    
    return weights.to(DEVICE)


def weighted_mse_with_huber_xt(pred, target, weights, huber_delta):
    """
    SUITE v2: Hybrid loss — Huber for cumulative xT, MSE for rest.
    
    Huber loss is less sensitive to outliers than MSE. For high-xT windows
    (rare attacking surges), MSE penalizes heavily, causing the model to 
    underpredict. Huber transitions to linear loss beyond delta, reducing
    this effect.
    """
    loss = torch.zeros_like(pred)
    
    # Column 0 (Cumulative xT): Huber loss
    diff = pred[:, 0] - target[:, 0]
    abs_diff = torch.abs(diff)
    huber = torch.where(
        abs_diff <= huber_delta,
        0.5 * diff ** 2,
        huber_delta * (abs_diff - 0.5 * huber_delta)
    )
    loss[:, 0] = huber
    
    # Columns 1-4 (Press Height, Field Tilt, Verticality, Tempo): MSE
    loss[:, 1:] = (pred[:, 1:] - target[:, 1:]) ** 2
    
    return (loss * weights).mean()


def composite_loss(reg_pred, reg_target, cls_def_pred, cls_def_target, 
                   cls_off_pred, cls_off_target, cls_out_pred, cls_out_target,
                   outcome_weights=None):
    """SUITE v2: Uses Huber for xT + class-weighted CE for outcome."""
    
    loss_reg = weighted_mse_with_huber_xt(reg_pred, reg_target, REG_WEIGHTS, HUBER_DELTA)
    loss_def = F.cross_entropy(cls_def_pred, cls_def_target)
    loss_off = F.cross_entropy(cls_off_pred, cls_off_target)
    
    # Class-weighted cross entropy for outcome
    if outcome_weights is not None:
        loss_out = F.cross_entropy(cls_out_pred, cls_out_target, weight=outcome_weights)
    else:
        loss_out = F.cross_entropy(cls_out_pred, cls_out_target)
    
    total = (loss_reg + CLS_WEIGHT_DEF * loss_def + 
             CLS_WEIGHT_OFF * loss_off + CLS_WEIGHT_OUTCOME * loss_out)
    
    return total, loss_reg.item(), loss_def.item(), loss_off.item(), loss_out.item()


def train():
    print(f"=== SUITE v2 — TRAINING ON {DEVICE} ===")
    print(f"    Changes: Huber loss for xT + class-weighted outcome + attention extraction")
    
    print("\nLoading Dataset...")
    dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME, 
        window_size=5, stride=1, max_matches=MAX_MATCHES
    )
    
    sample = dataset[0]
    print(f"  Dims: Node={sample.x.shape[1]}, Edge={sample.edge_attr.shape[1]}, Global={sample.u.shape[1]}")
    
    train_indices, test_indices = match_level_split(dataset, train_ratio=0.8, seed=42)
    
    # SUITE v2: Compute outcome class weights from training set
    print("\nComputing class weights...")
    outcome_weights = compute_outcome_class_weights(dataset, train_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TacticalGAT(
        num_node_features=sample.x.shape[1], num_reg_targets=5,
        num_def_classes=3, num_off_classes=3, num_outcome_classes=3,
        edge_dim=sample.edge_attr.shape[1], global_dim=sample.u.shape[1]
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, val_losses, val_reg_losses = [], [], []
    val_def_accs, val_off_accs, val_out_accs = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    stopped_epoch = EPOCHS
    
    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in loop:
            batch = batch.to(DEVICE)
            reg_pred, cls_def_pred, cls_off_pred, cls_out_pred = model(batch)
            
            reg_target = batch.y.view(-1, 5)
            cls_targets = batch.y_cls.view(-1, 3)
            
            loss, _, _, _, _ = composite_loss(
                reg_pred, reg_target, 
                cls_def_pred, cls_targets[:, 0],
                cls_off_pred, cls_targets[:, 1],
                cls_out_pred, cls_targets[:, 2],
                outcome_weights=outcome_weights,
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_reg = 0
        correct_def, correct_off, correct_out = 0, 0, 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                reg_pred, cls_def_pred, cls_off_pred, cls_out_pred = model(batch)
                
                reg_target = batch.y.view(-1, 5)
                cls_targets = batch.y_cls.view(-1, 3)
                
                loss, reg_l, _, _, _ = composite_loss(
                    reg_pred, reg_target,
                    cls_def_pred, cls_targets[:, 0],
                    cls_off_pred, cls_targets[:, 1],
                    cls_out_pred, cls_targets[:, 2],
                    outcome_weights=outcome_weights,
                )
                
                total_val_loss += loss.item()
                total_val_reg += reg_l
                correct_def += (cls_def_pred.argmax(1) == cls_targets[:, 0]).sum().item()
                correct_off += (cls_off_pred.argmax(1) == cls_targets[:, 1]).sum().item()
                correct_out += (cls_out_pred.argmax(1) == cls_targets[:, 2]).sum().item()
                total_samples += cls_targets.size(0)
                
        avg_val_loss = total_val_loss / len(test_loader)
        avg_val_reg = total_val_reg / len(test_loader)
        def_acc = correct_def / total_samples
        off_acc = correct_off / total_samples
        out_acc = correct_out / total_samples
        
        val_losses.append(avg_val_loss)
        val_reg_losses.append(avg_val_reg)
        val_def_accs.append(def_acc)
        val_off_accs.append(off_acc)
        val_out_accs.append(out_acc)
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"Reg: {avg_val_reg:.4f} | DefAcc: {def_acc:.1%} | OffAcc: {off_acc:.1%} | "
              f"OutAcc: {out_acc:.1%} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> New Best Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n   Early stopping at epoch {epoch+1}")
                stopped_epoch = epoch + 1
                break

    actual_epochs = len(train_losses)
    print(f"\nTraining Complete ({actual_epochs} epochs). Plotting...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    if stopped_epoch < EPOCHS:
        axes[0].axvline(x=stopped_epoch-1, color='red', linestyle='--', alpha=0.5, label=f'Stop ({stopped_epoch})')
    axes[0].set_title('Composite Loss'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(val_reg_losses, color='green', label='Val Reg Loss')
    axes[1].set_title('Regression Loss (Val)'); axes[1].legend(); axes[1].grid(True)
    
    axes[2].plot(val_def_accs, label='Def Posture', color='red')
    axes[2].plot(val_off_accs, label='Off Style', color='blue')
    axes[2].plot(val_out_accs, label='Outcome', color='purple')
    axes[2].set_title('Classification Accuracy (Val)'); axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True)
    
    plt.suptitle(f"SUITE v2 (stopped epoch {stopped_epoch})", fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig('training_curve.png')
    print("Saved training_curve.png")

if __name__ == "__main__":
    train()
