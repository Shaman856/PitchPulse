"""
BASELINE DIAGNOSTIC: Random Model vs Trained Model
====================================================
Compares a randomly initialized (untrained) model against the trained model
to measure how much value training actually adds.

Generates:
  - baseline_regression.png:     Side-by-side scatter plots (Random vs Trained)
  - baseline_classification.png: Side-by-side confusion matrices (Random vs Trained)
  - Console report with R², MAE, accuracy comparisons

Usage:
    python baseline_diagnostic.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT

# --- CONFIGURATION ---
DATASET_PATH = "./data_v3"
DATASET_NAME = "offline_mix_v4"   # Match your current dataset name
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REG_NAMES = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']
DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
OFF_CLASSES = ['Patient', 'Balanced', 'Counter']


def denormalize(preds_or_targets):
    """Reverse target normalization for interpretable plots."""
    out = preds_or_targets.copy()
    out[:, 0] = np.expm1(out[:, 0])        # Cumulative xT
    out[:, 1] = out[:, 1] * 120.0           # Press Height
    out[:, 3] = out[:, 3] * 2.0 - 1.0       # Verticality
    out[:, 4] = out[:, 4] * 30.0            # Tempo
    return out


def compute_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def run_inference(model, loader):
    """Run inference and return raw normalized predictions + targets."""
    model.eval()
    
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
    
    return {
        'reg_preds': np.vstack(all_reg_preds),
        'reg_targets': np.vstack(all_reg_targets),
        'def_preds': np.concatenate(all_def_preds),
        'def_targets': np.concatenate(all_def_targets),
        'off_preds': np.concatenate(all_off_preds),
        'off_targets': np.concatenate(all_off_targets),
    }


def evaluate(results):
    """Compute all metrics from inference results."""
    preds = results['reg_preds']
    targets = results['reg_targets']
    
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = compute_r2(targets, preds)
    
    def_acc = (results['def_preds'] == results['def_targets']).mean()
    off_acc = (results['off_preds'] == results['off_targets']).mean()
    
    return {
        'mae': mae,
        'r2': r2,
        'def_acc': def_acc,
        'off_acc': off_acc,
    }


def mean_baseline(results):
    """Naive baseline: predict dataset mean / majority class."""
    targets = results['reg_targets']
    
    mean_preds = np.tile(targets.mean(axis=0), (len(targets), 1))
    mae = np.mean(np.abs(mean_preds - targets), axis=0)
    r2 = np.zeros(targets.shape[1])
    
    from collections import Counter
    def_majority = Counter(results['def_targets'].tolist()).most_common(1)[0][0]
    off_majority = Counter(results['off_targets'].tolist()).most_common(1)[0][0]
    
    def_acc = (results['def_targets'] == def_majority).mean()
    off_acc = (results['off_targets'] == off_majority).mean()
    
    return {
        'mae': mae,
        'r2': r2,
        'def_acc': def_acc,
        'off_acc': off_acc,
    }


def plot_regression_comparison(random_results, trained_results, random_metrics, trained_metrics):
    """
    Side-by-side scatter plots: Random (top row) vs Trained (bottom row).
    5 regression targets × 2 rows = 10 subplots.
    """
    rand_preds = denormalize(random_results['reg_preds'])
    rand_targets = denormalize(random_results['reg_targets'])
    train_preds = denormalize(trained_results['reg_preds'])
    train_targets = denormalize(trained_results['reg_targets'])
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    for col in range(5):
        for row, (preds, targets, metrics, label, color) in enumerate([
            (rand_preds, rand_targets, random_metrics, 'RANDOM (Untrained)', '#d9534f'),
            (train_preds, train_targets, trained_metrics, 'TRAINED', '#5bc0de'),
        ]):
            ax = axes[row, col]
            
            # Downsample for plotting
            if len(preds) > 1000:
                idx = np.random.choice(len(preds), 1000, replace=False)
                x_vals = targets[idx, col]
                y_vals = preds[idx, col]
            else:
                x_vals = targets[:, col]
                y_vals = preds[:, col]
            
            ax.scatter(x_vals, y_vals, alpha=0.3, s=10, color=color)
            
            # y=x line
            all_vals = np.concatenate([x_vals, y_vals])
            lims = [np.min(all_vals) - 0.05, np.max(all_vals) + 0.05]
            ax.plot(lims, lims, 'k-', alpha=0.5, linewidth=1.5)
            
            r2 = metrics['r2'][col]
            ax.set_title(f"{REG_NAMES[col]}\nR²={r2:.3f}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Actual", fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if col == 0:
                ax.set_ylabel(f"{label}\nPredicted", fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel("Predicted", fontsize=8)
    
    plt.suptitle("BASELINE DIAGNOSTIC: Random Model (top, red) vs Trained Model (bottom, blue)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("baseline_regression.png", bbox_inches='tight', dpi=150)
    print("Saved baseline_regression.png")


def plot_classification_comparison(random_results, trained_results, random_metrics, trained_metrics):
    """
    Side-by-side confusion matrices: Random (left) vs Trained (right).
    2 classification tasks × 2 models = 4 subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    configs = [
        (0, 0, random_results['def_preds'], random_results['def_targets'], DEF_CLASSES, 'RANDOM - Def Posture', 'Reds'),
        (0, 1, trained_results['def_preds'], trained_results['def_targets'], DEF_CLASSES, 'TRAINED - Def Posture', 'Blues'),
        (1, 0, random_results['off_preds'], random_results['off_targets'], OFF_CLASSES, 'RANDOM - Off Style', 'Reds'),
        (1, 1, trained_results['off_preds'], trained_results['off_targets'], OFF_CLASSES, 'TRAINED - Off Style', 'Blues'),
    ]
    
    for row, col, preds_cls, targets_cls, names, title, cmap in configs:
        ax = axes[row, col]
        n_classes = len(names)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(targets_cls, preds_cls):
            cm[int(t), int(p)] += 1
        
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        
        acc = (preds_cls == targets_cls).mean()
        ax.set_title(f"{title}\nAccuracy: {acc:.1%}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(names)
        
        for i in range(n_classes):
            for j in range(n_classes):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                        ha='center', va='center', color=color, fontsize=9)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("BASELINE DIAGNOSTIC: Random (left, red) vs Trained (right, blue)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("baseline_classification.png", bbox_inches='tight', dpi=150)
    print("Saved baseline_classification.png")


def main():
    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME,
        window_size=5, stride=1, max_matches=MAX_MATCHES
    )
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Auto-detect dimensions
    sample = dataset[0]
    model_kwargs = dict(
        num_node_features=sample.x.shape[1],
        num_reg_targets=5,
        num_def_classes=3,
        num_off_classes=3,
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1],
    )
    
    # 3. RANDOM MODEL (untrained)
    print("\nRunning RANDOM (untrained) model...")
    random_model = TacticalGAT(**model_kwargs).to(DEVICE)
    random_results = run_inference(random_model, loader)
    random_metrics = evaluate(random_results)
    
    # 4. TRAINED MODEL
    print("Running TRAINED model...")
    trained_model = TacticalGAT(**model_kwargs).to(DEVICE)
    trained_model.load_state_dict(torch.load(MODEL_PATH))
    trained_results = run_inference(trained_model, loader)
    trained_metrics = evaluate(trained_results)
    
    # 5. MEAN BASELINE
    mean_metrics = mean_baseline(trained_results)
    
    # ================================================================
    # CONSOLE REPORT
    # ================================================================
    print("\n" + "=" * 75)
    print("   BASELINE DIAGNOSTIC REPORT")
    print("   Random Init vs Trained vs Mean Baseline")
    print("=" * 75)
    
    print("\n--- REGRESSION (R²) ---")
    print(f"{'Metric':<18} | {'Mean Baseline':<14} | {'Random Model':<14} | {'Trained Model':<14} | {'Training Gain':<14}")
    print("-" * 75)
    
    for i, name in enumerate(REG_NAMES):
        r2_mean = mean_metrics['r2'][i]
        r2_rand = random_metrics['r2'][i]
        r2_train = trained_metrics['r2'][i]
        gain = r2_train - r2_rand
        print(f"{name:<18} | {r2_mean:<14.4f} | {r2_rand:<14.4f} | {r2_train:<14.4f} | {gain:<+14.4f}")
    
    print("\n--- REGRESSION (MAE) ---")
    print(f"{'Metric':<18} | {'Mean Baseline':<14} | {'Random Model':<14} | {'Trained Model':<14} | {'Training Gain':<14}")
    print("-" * 75)
    
    for i, name in enumerate(REG_NAMES):
        mae_mean = mean_metrics['mae'][i]
        mae_rand = random_metrics['mae'][i]
        mae_train = trained_metrics['mae'][i]
        gain = mae_rand - mae_train
        print(f"{name:<18} | {mae_mean:<14.4f} | {mae_rand:<14.4f} | {mae_train:<14.4f} | {gain:<+14.4f}")
    
    print("\n--- CLASSIFICATION (Accuracy) ---")
    print(f"{'Task':<18} | {'Majority Class':<14} | {'Random Model':<14} | {'Trained Model':<14} | {'Training Gain':<14}")
    print("-" * 75)
    
    print(f"{'Def Posture':<18} | {mean_metrics['def_acc']:<14.1%} | {random_metrics['def_acc']:<14.1%} | {trained_metrics['def_acc']:<14.1%} | {trained_metrics['def_acc'] - random_metrics['def_acc']:<+14.1%}")
    print(f"{'Off Style':<18} | {mean_metrics['off_acc']:<14.1%} | {random_metrics['off_acc']:<14.1%} | {trained_metrics['off_acc']:<14.1%} | {trained_metrics['off_acc'] - random_metrics['off_acc']:<+14.1%}")
    
    # --- VERDICT ---
    avg_r2_gain = np.mean(trained_metrics['r2'] - random_metrics['r2'])
    
    print("\n" + "=" * 75)
    print("   VERDICT")
    print("=" * 75)
    print(f"Average R² gain from training: {avg_r2_gain:+.4f}")
    
    if avg_r2_gain > 0.3:
        print("STRONG: Training adds substantial value. The GAT is learning real patterns.")
    elif avg_r2_gain > 0.15:
        print("MODERATE: Training helps, but a significant portion comes from easy features.")
        print("   -> Consider improving edge construction (recipient-based edges).")
    elif avg_r2_gain > 0.05:
        print("WEAK: Training adds minimal value beyond what random weights capture.")
        print("   -> The graph structure may not be informative enough. Priority: fix edges.")
    else:
        print("NEGLIGIBLE: The model is not learning from the graph.")
        print("   -> Major rework needed: edge construction + feature engineering.")
    
    print("=" * 75)
    
    # ================================================================
    # PLOTS
    # ================================================================
    print("\nGenerating plots...")
    plot_regression_comparison(random_results, trained_results, random_metrics, trained_metrics)
    plot_classification_comparison(random_results, trained_results, random_metrics, trained_metrics)
    print("\nDone!")


if __name__ == "__main__":
    main()