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
DATASET_NAME = "offline_mix_v4"   # Must match train.py
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Regression target names and denormalization info
REG_NAMES = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']
DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
OFF_CLASSES = ['Patient', 'Balanced', 'Counter']


def denormalize(preds_or_targets):
    """
    Reverses target normalization from graph_builder.py for interpretable plots.
    
    [0] Cum xT:       was log1p(xT) -> expm1 to reverse
    [1] Press Height:  was /120.0 -> *120
    [2] Field Tilt:    already [0, 1] -> no change
    [3] Verticality:   was (v+1)/2 -> v*2 - 1
    [4] Tempo:         was /30.0 -> *30
    """
    out = preds_or_targets.copy()
    out[:, 0] = np.expm1(out[:, 0])        # Reverse log1p
    out[:, 1] = out[:, 1] * 120.0           # Press Height -> pitch coordinate
    out[:, 3] = out[:, 3] * 2.0 - 1.0       # Verticality -> [-1, 1]
    out[:, 4] = out[:, 4] * 30.0            # Tempo -> passes/min
    return out


def load_data_and_model():
    print("1. Loading Data...")
    dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME, 
        window_size=5, stride=1, max_matches=MAX_MATCHES
    )
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"2. Loading Model from {MODEL_PATH}...")
    sample = dataset[0]
    model = TacticalGAT(
        num_node_features=sample.x.shape[1], 
        num_reg_targets=5,
        num_def_classes=3,
        num_off_classes=3,
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1]
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    return model, loader


def compute_r2(actual, predicted):
    """Compute R² (coefficient of determination) per column."""
    ss_res = np.sum((actual - predicted) ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def analyze_predictions(model, loader):
    print("3. Running Inference...")
    
    all_reg_preds = []
    all_reg_targets = []
    all_def_preds = []
    all_def_targets = []
    all_off_preds = []
    all_off_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            reg_out, cls_def_out, cls_off_out = model(batch)
            
            reg_target = batch.y.view(-1, 5)
            cls_targets = batch.y_cls.view(-1, 2)
            
            all_reg_preds.append(reg_out.cpu().numpy())
            all_reg_targets.append(reg_target.cpu().numpy())
            all_def_preds.append(cls_def_out.argmax(dim=1).cpu().numpy())
            all_def_targets.append(cls_targets[:, 0].cpu().numpy())
            all_off_preds.append(cls_off_out.argmax(dim=1).cpu().numpy())
            all_off_targets.append(cls_targets[:, 1].cpu().numpy())
            
    # Stack
    preds_norm = np.vstack(all_reg_preds)
    targets_norm = np.vstack(all_reg_targets)
    def_preds = np.concatenate(all_def_preds)
    def_targets = np.concatenate(all_def_targets)
    off_preds = np.concatenate(all_off_preds)
    off_targets = np.concatenate(all_off_targets)
    
    # Denormalize regression for interpretable reporting
    preds = denormalize(preds_norm)
    targets = denormalize(targets_norm)
    
    # =====================================================================
    # REPORT 1: REGRESSION PERFORMANCE
    # =====================================================================
    print("\n" + "="*60)
    print("   REGRESSION PERFORMANCE REPORT")
    print("="*60)
    
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = compute_r2(targets, preds)
    
    print(f"{'Metric':<18} | {'MAE':<10} | {'R²':<10}")
    print("-" * 45)
    for i, name in enumerate(REG_NAMES):
        print(f"{name:<18} | {mae[i]:<10.4f} | {r2[i]:<10.4f}")
    
    print("-" * 45)
    print("INTERPRETATION:")
    print(f"  Cumulative xT: Off by {mae[0]:.4f} xT on average.")
    print(f"  Press Height:  Off by {mae[1]:.1f} units (out of 120).")
    print(f"  Field Tilt:    Off by {mae[2]:.3f} (0-1 scale).")
    print(f"  Verticality:   Off by {mae[3]:.3f} (-1 to 1 scale).")
    print(f"  Tempo:         Off by {mae[4]:.1f} passes/min.")
    
    # =====================================================================
    # REPORT 2: CLASSIFICATION PERFORMANCE
    # =====================================================================
    print("\n" + "="*60)
    print("   CLASSIFICATION PERFORMANCE REPORT")
    print("="*60)
    
    def_acc = (def_preds == def_targets).mean()
    off_acc = (off_preds == off_targets).mean()
    
    print(f"Defensive Posture Accuracy: {def_acc:.1%}")
    print(f"Offensive Style Accuracy:   {off_acc:.1%}")
    
    # Per-class accuracy
    print("\n--- Defensive Posture (Per-Class) ---")
    for c, name in enumerate(DEF_CLASSES):
        mask = def_targets == c
        if mask.sum() > 0:
            acc = (def_preds[mask] == c).mean()
            print(f"  {name:<12} | Acc: {acc:.1%} | Support: {mask.sum()}")
    
    print("\n--- Offensive Style (Per-Class) ---")
    for c, name in enumerate(OFF_CLASSES):
        mask = off_targets == c
        if mask.sum() > 0:
            acc = (off_preds[mask] == c).mean()
            print(f"  {name:<12} | Acc: {acc:.1%} | Support: {mask.sum()}")
    
    # =====================================================================
    # REPORT 3: SAMPLE PREDICTIONS (Eye Test)
    # =====================================================================
    print("\n" + "="*60)
    print("   SAMPLE PREDICTIONS (Eye Test)")
    print("="*60)
    
    indices = np.random.choice(len(preds), 5, replace=False)
    
    for idx in indices:
        print(f"\n--- Sample {idx} ---")
        for i, name in enumerate(REG_NAMES):
            act = targets[idx][i]
            pre = preds[idx][i]
            print(f"  {name:<18} | Actual: {act:<8.3f} | Pred: {pre:<8.3f} | Diff: {pre-act:<+8.3f}")
        
        def_act = DEF_CLASSES[int(def_targets[idx])]
        def_pre = DEF_CLASSES[int(def_preds[idx])]
        off_act = OFF_CLASSES[int(off_targets[idx])]
        off_pre = OFF_CLASSES[int(off_preds[idx])]
        print(f"  {'Def Posture':<18} | Actual: {def_act:<12} | Pred: {def_pre:<12} | {'✓' if def_act == def_pre else '✗'}")
        print(f"  {'Off Style':<18} | Actual: {off_act:<12} | Pred: {off_pre:<12} | {'✓' if off_act == off_pre else '✗'}")

    # =====================================================================
    # REPORT 4: VISUALIZATIONS
    # =====================================================================
    
    # --- Figure 1: Regression Scatter Plots (5 subplots) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        
        if len(preds) > 1000:
            sample_idx = np.random.choice(len(preds), 1000, replace=False)
            x_vals = targets[sample_idx, i]
            y_vals = preds[sample_idx, i]
        else:
            x_vals = targets[:, i]
            y_vals = preds[:, i]
            
        ax.scatter(x_vals, y_vals, alpha=0.3, s=10)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        
        ax.set_title(f"{REG_NAMES[i]} (R²={r2[i]:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig("inference_regression.png")
    print("\nSaved inference_regression.png")
    
    # --- Figure 2: Classification Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (preds_cls, targets_cls, names, title) in zip(axes, [
        (def_preds, def_targets, DEF_CLASSES, 'Defensive Posture'),
        (off_preds, off_targets, OFF_CLASSES, 'Offensive Style'),
    ]):
        n_classes = len(names)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(targets_cls, preds_cls):
            cm[int(t), int(p)] += 1
        
        # Normalize rows (recall per class)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{title} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(names)
        
        # Annotate cells with count and percentage
        for i in range(n_classes):
            for j in range(n_classes):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})', 
                        ha='center', va='center', color=color, fontsize=9)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("inference_classification.png")
    print("Saved inference_classification.png")


if __name__ == "__main__":
    model, loader = load_data_and_model()
    analyze_predictions(model, loader)