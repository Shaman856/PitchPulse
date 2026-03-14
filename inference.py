import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT

DATASET_PATH = "./data_v3" 
DATASET_NAME = "offline_mix_v7_suite"
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REG_NAMES = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']
DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
OFF_CLASSES = ['Patient', 'Balanced', 'Counter']
OUT_CLASSES = ['Loss', 'Draw', 'Win']


def match_level_split(dataset, train_ratio=0.8, seed=42):
    match_ids = [dataset[i].match_id for i in range(len(dataset))]
    unique_matches = sorted(set(match_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)
    n_train = int(len(unique_matches) * train_ratio)
    train_matches = set(unique_matches[:n_train])
    train_idx = [i for i, m in enumerate(match_ids) if m in train_matches]
    test_idx = [i for i, m in enumerate(match_ids) if m not in train_matches]
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test windows")
    return train_idx, test_idx


def denormalize(arr):
    out = arr.copy()
    out[:, 0] = np.expm1(out[:, 0])
    out[:, 1] = out[:, 1] * 120.0
    out[:, 3] = out[:, 3] * 2.0 - 1.0
    out[:, 4] = out[:, 4] * 30.0
    return out


def compute_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def run():
    print("1. Loading Data...")
    dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR, dataset_name=DATASET_NAME,
        window_size=5, stride=1, max_matches=MAX_MATCHES
    )
    _, test_idx = match_level_split(dataset)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    print("2. Loading Model...")
    sample = dataset[0]
    model = TacticalGAT(
        num_node_features=sample.x.shape[1], num_reg_targets=5,
        num_def_classes=3, num_off_classes=3, num_outcome_classes=3,
        edge_dim=sample.edge_attr.shape[1], global_dim=sample.u.shape[1]
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("3. Running Inference...")
    all_rp, all_rt = [], []
    all_dp, all_dt = [], []
    all_op, all_ot = [], []
    all_wp, all_wt = [], []  # outcome (win/draw/loss)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            reg, cls_def, cls_off, cls_out = model(batch)
            
            cls_targets = batch.y_cls.view(-1, 3)
            
            all_rp.append(reg.cpu().numpy())
            all_rt.append(batch.y.view(-1, 5).cpu().numpy())
            all_dp.append(cls_def.argmax(1).cpu().numpy())
            all_dt.append(cls_targets[:, 0].cpu().numpy())
            all_op.append(cls_off.argmax(1).cpu().numpy())
            all_ot.append(cls_targets[:, 1].cpu().numpy())
            all_wp.append(cls_out.argmax(1).cpu().numpy())
            all_wt.append(cls_targets[:, 2].cpu().numpy())
    
    preds_n = np.vstack(all_rp); targets_n = np.vstack(all_rt)
    dp = np.concatenate(all_dp); dt = np.concatenate(all_dt)
    op = np.concatenate(all_op); ot = np.concatenate(all_ot)
    wp = np.concatenate(all_wp); wt = np.concatenate(all_wt)
    
    preds = denormalize(preds_n); targets = denormalize(targets_n)
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = compute_r2(targets, preds)
    
    # =====================================================================
    # REPORT
    # =====================================================================
    print("\n" + "="*60)
    print("   COMPLETE TACTICAL SUITE — RESULTS")
    print("="*60)
    
    print(f"\n{'--- REGRESSION ---':^60}")
    print(f"{'Metric':<18} | {'MAE':<10} | {'R²':<10}")
    print("-" * 45)
    for i, name in enumerate(REG_NAMES):
        print(f"{name:<18} | {mae[i]:<10.4f} | {r2[i]:<10.4f}")
    
    def_acc = (dp == dt).mean()
    off_acc = (op == ot).mean()
    out_acc = (wp == wt).mean()
    
    print(f"\n{'--- CLASSIFICATION ---':^60}")
    print(f"Defensive Posture Accuracy:  {def_acc:.1%}")
    print(f"Offensive Style Accuracy:    {off_acc:.1%}")
    print(f"Match Outcome Accuracy:      {out_acc:.1%}  [NEW]")
    
    print(f"\n--- Defensive Posture (Per-Class) ---")
    for c, name in enumerate(DEF_CLASSES):
        mask = dt == c
        if mask.sum() > 0:
            print(f"  {name:<12} | Acc: {(dp[mask]==c).mean():.1%} | Support: {mask.sum()}")
    
    print(f"\n--- Offensive Style (Per-Class) ---")
    for c, name in enumerate(OFF_CLASSES):
        mask = ot == c
        if mask.sum() > 0:
            print(f"  {name:<12} | Acc: {(op[mask]==c).mean():.1%} | Support: {mask.sum()}")
    
    print(f"\n--- Match Outcome (Per-Class) ---  [NEW]")
    for c, name in enumerate(OUT_CLASSES):
        mask = wt == c
        if mask.sum() > 0:
            print(f"  {name:<12} | Acc: {(wp[mask]==c).mean():.1%} | Support: {mask.sum()}")
    
    # Sample predictions
    print(f"\n{'--- SAMPLE PREDICTIONS ---':^60}")
    indices = np.random.choice(len(preds), 5, replace=False)
    for idx in indices:
        print(f"\n--- Sample {idx} ---")
        for i, name in enumerate(REG_NAMES):
            a, p = targets[idx][i], preds[idx][i]
            print(f"  {name:<18} | Actual: {a:<8.3f} | Pred: {p:<8.3f} | Diff: {p-a:<+8.3f}")
        da = DEF_CLASSES[int(dt[idx])]; dd = DEF_CLASSES[int(dp[idx])]
        oa = OFF_CLASSES[int(ot[idx])]; od = OFF_CLASSES[int(op[idx])]
        wa = OUT_CLASSES[int(wt[idx])]; wd = OUT_CLASSES[int(wp[idx])]
        print(f"  {'Def Posture':<18} | Actual: {da:<12} | Pred: {dd:<12} | {'✓' if da==dd else '✗'}")
        print(f"  {'Off Style':<18} | Actual: {oa:<12} | Pred: {od:<12} | {'✓' if oa==od else '✗'}")
        print(f"  {'Outcome':<18} | Actual: {wa:<12} | Pred: {wd:<12} | {'✓' if wa==wd else '✗'}")
    
    # =====================================================================
    # PLOTS
    # =====================================================================
    # Regression
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i in range(5):
        ax = axes[i]
        if len(preds) > 1000:
            idx = np.random.choice(len(preds), 1000, replace=False)
            xv, yv = targets[idx, i], preds[idx, i]
        else:
            xv, yv = targets[:, i], preds[:, i]
        ax.scatter(xv, yv, alpha=0.3, s=10)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
        ax.set_title(f"{REG_NAMES[i]} (R²={r2[i]:.3f})")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.grid(True)
    axes[5].axis('off')
    plt.suptitle("COMPLETE SUITE — Regression", fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig("inference_regression.png")
    print("\nSaved inference_regression.png")
    
    # Classification (3 confusion matrices)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (pc, tc, names, title) in zip(axes, [
        (dp, dt, DEF_CLASSES, 'Defensive Posture'),
        (op, ot, OFF_CLASSES, 'Offensive Style'),
        (wp, wt, OUT_CLASSES, 'Match Outcome'),
    ]):
        n = len(names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(tc, pc): cm[int(t), int(p)] += 1
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{title}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})', ha='center', va='center', color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("COMPLETE SUITE — Classification (incl. Outcome)", fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig("inference_classification.png")
    print("Saved inference_classification.png")


if __name__ == "__main__":
    run()
