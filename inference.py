import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT

# --- CONFIGURATION ---
DATASET_PATH = "./data_v3" 
DATASET_NAME = "offline_mix_v4"
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Regression target names
REG_NAMES = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']

# === CLASSIFICATION DISABLED ===
# DEF_CLASSES = ['Low Block', 'Mid Block', 'High Press']
# OFF_CLASSES = ['Patient', 'Balanced', 'Counter']


def match_level_split(dataset, train_ratio=0.8, seed=42):
    """
    Match-level split (must match train.py exactly)
    """
    match_ids = []
    for i in range(len(dataset)):
        match_ids.append(dataset[i].match_id)

    unique_matches = sorted(set(match_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)

    n_train = int(len(unique_matches) * train_ratio)
    train_matches = set(unique_matches[:n_train])
    test_matches = set(unique_matches[n_train:])

    train_indices = [i for i, mid in enumerate(match_ids) if mid in train_matches]
    test_indices = [i for i, mid in enumerate(match_ids) if mid in test_matches]

    print(f"Match-Level Split: {len(train_matches)} train / {len(test_matches)} test matches")
    print(f"  Test windows: {len(test_indices)}")

    return train_indices, test_indices


def denormalize(preds_or_targets):
    """
    Reverse normalization from graph_builder.py
    """
    out = preds_or_targets.copy()
    out[:, 0] = np.expm1(out[:, 0])
    out[:, 1] = out[:, 1] * 120.0
    out[:, 3] = out[:, 3] * 2.0 - 1.0
    out[:, 4] = out[:, 4] * 30.0
    return out


def load_data_and_model():
    print("1. Loading Data...")
    dataset = TacticalDataset(
        root=DATASET_PATH,
        raw_dir=RAW_DATA_DIR,
        dataset_name=DATASET_NAME,
        window_size=5,
        stride=1,
        max_matches=MAX_MATCHES
    )

    # Use same match-level split as training
    _, test_indices = match_level_split(dataset, train_ratio=0.8, seed=42)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"2. Loading Model from {MODEL_PATH}...")
    sample = dataset[0]

    model = TacticalGAT(
        num_node_features=sample.x.shape[1],
        num_reg_targets=5,
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1]
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    return model, loader


def compute_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def analyze_predictions(model, loader):
    print("3. Running Inference...")

    all_reg_preds = []
    all_reg_targets = []

    # === CLASSIFICATION DISABLED ===
    # all_def_preds = []
    # all_def_targets = []
    # all_off_preds = []
    # all_off_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            # === CLASSIFICATION DISABLED ===
            # reg_out, cls_def_out, cls_off_out = model(batch)

            reg_out = model(batch)
            reg_target = batch.y.view(-1, 5)

            all_reg_preds.append(reg_out.cpu().numpy())
            all_reg_targets.append(reg_target.cpu().numpy())

    preds_norm = np.vstack(all_reg_preds)
    targets_norm = np.vstack(all_reg_targets)

    preds = denormalize(preds_norm)
    targets = denormalize(targets_norm)

    # ============================================================
    # REPORT 1: REGRESSION PERFORMANCE
    # ============================================================

    print("\n" + "="*60)
    print("   REGRESSION PERFORMANCE REPORT")
    print("="*60)

    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = compute_r2(targets, preds)

    print(f"{'Metric':<18} | {'MAE':<10} | {'R²':<10}")
    print("-" * 45)

    for i, name in enumerate(REG_NAMES):
        print(f"{name:<18} | {mae[i]:<10.4f} | {r2[i]:<10.4f}")

    # ============================================================
    # SAMPLE PREDICTIONS
    # ============================================================

    print("\n" + "="*60)
    print("   SAMPLE PREDICTIONS")
    print("="*60)

    indices = np.random.choice(len(preds), 5, replace=False)

    for idx in indices:
        print(f"\n--- Sample {idx} ---")
        for i, name in enumerate(REG_NAMES):
            act = targets[idx][i]
            pre = preds[idx][i]
            print(
                f"  {name:<18} | Actual: {act:<8.3f} | "
                f"Pred: {pre:<8.3f} | Diff: {pre-act:<+8.3f}"
            )

    # ============================================================
    # VISUALIZATION
    # ============================================================

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
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75)

        ax.set_title(f"{REG_NAMES[i]} (R²={r2[i]:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)

    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig("inference_regression.png")
    print("\nSaved inference_regression.png")


if __name__ == "__main__":
    model, loader = load_data_and_model()
    analyze_predictions(model, loader)