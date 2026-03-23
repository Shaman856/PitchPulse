# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from torch.utils.data import DataLoader, Subset

from preprocessing.dataset import TacticalDataset
from preprocessing.sequence_dataset import SequenceTacticalDataset, sequence_collate_fn
from models.model import TacticalGATLSTM

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH    = "./data_v3"
DATASET_NAME    = "offline_mix_v7_suite"
RAW_DATA_DIR    = "./data/raw_events"
MAX_MATCHES     = 230
SEQ_LEN         = 5
MODEL_PATH      = "best_model.pth"
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This file is created by train.py — contains the explicit match ID sets
# for all three splits so we can guarantee zero contamination
SPLIT_SAVE_PATH = "train_val_test_split.json"

REG_NAMES  = ['Cumulative xT', 'Press Height', 'Field Tilt', 'Verticality', 'Tempo']
DEF_LABELS = ['Low Block', 'Mid Block', 'High Press']
OFF_LABELS = ['Patient', 'Balanced', 'Counter']


def load_test_indices(base_dataset):
    """
    Loads only the TEST split from the saved JSON file.

    Why not recompute the split here:
        Recomputing requires the same seed, same dataset size, and same
        MAX_MATCHES. If any of these change between train.py and inference.py,
        the split silently shifts and test matches might overlap with training
        or validation. Loading from file makes the split a fixed artifact.

    Why specifically the TEST split and not the validation split:
        The validation split was used to guide early stopping during training.
        The model was implicitly tuned toward it. Using validation data for
        final reporting would make the metrics optimistic.
        The test split was NEVER touched during training in any way.
        These are the honest final metrics.

    Also performs three safety checks:
        1. No overlap between train and test match IDs
        2. No overlap between val and test match IDs
        3. Warns if dataset size changed since training

    Returns:
        test_idx: List of window indices for test matches only
    """
    # The split file must exist — it is written by train.py
    if not os.path.exists(SPLIT_SAVE_PATH):
        raise FileNotFoundError(
            f"\nSplit file not found: {SPLIT_SAVE_PATH}\n"
            f"You must run train.py before inference.py.\n"
            f"train.py creates this file automatically during training."
        )

    # Load the saved split
    with open(SPLIT_SAVE_PATH, 'r') as f:
        split_data = json.load(f)

    # Extract all three match ID sets
    train_match_ids = set(split_data["train_match_ids"])
    val_match_ids   = set(split_data["val_match_ids"])
    test_match_ids  = set(split_data["test_match_ids"])

    # Safety check 1: confirm train and test are disjoint
    train_test_overlap = train_match_ids & test_match_ids
    if train_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE: {len(train_test_overlap)} matches appear in "
            f"both train and test. Split file may be corrupted. "
            f"Delete {SPLIT_SAVE_PATH} and retrain."
        )

    # Safety check 2: confirm val and test are disjoint
    val_test_overlap = val_match_ids & test_match_ids
    if val_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE: {len(val_test_overlap)} matches appear in "
            f"both val and test. Split file may be corrupted. "
            f"Delete {SPLIT_SAVE_PATH} and retrain."
        )

    # Collect match_id for every window currently in the dataset
    match_ids = [int(base_dataset[i].match_id) for i in range(len(base_dataset))]
    current_matches = set(match_ids)

    # Safety check 3: warn if dataset size changed since training
    # (e.g. more .pkl files were added or MAX_MATCHES was changed)
    saved_total = split_data["total_matches"]
    current_total = len(current_matches)
    if current_total != saved_total:
        print(f"\n  WARNING: Dataset has {current_total} matches but split was "
              f"created with {saved_total} matches.")
        print(f"  Test results may differ from training conditions.")
        print(f"  To fix: delete {SPLIT_SAVE_PATH} and retrain.")

    # Get window indices for test matches only
    # Windows from train or val matches are completely excluded
    test_idx = [i for i, m in enumerate(match_ids) if m in test_match_ids]

    # Print a clear summary so the user can verify correctness
    print(f"\n  Split loaded from: {SPLIT_SAVE_PATH}")
    print(f"  Training split:   {len(train_match_ids)} matches "
          f"({split_data['train_ratio']*100:.0f}%)")
    print(f"  Validation split: {len(val_match_ids)} matches "
          f"({split_data['val_ratio']*100:.0f}%) — used for early stopping")
    print(f"  Test split:       {len(test_match_ids)} matches "
          f"({split_data['test_ratio']*100:.0f}%) — evaluating on this set now")
    print(f"  Test windows:     {len(test_idx)}")
    print(f"  Leakage check:    PASSED — test set is disjoint from train and val")

    return test_idx


def denormalize(arr):
    """
    Reverses the normalisation applied in graph_builder.py.
    Metrics are returned in original human-interpretable units.
    """
    out = arr.copy()
    # xT stored as log1p → reverse with expm1
    out[:, 0] = np.expm1(out[:, 0])
    # Press height divided by 120 → multiply back to pitch units
    out[:, 1] = out[:, 1] * 120.0
    # Verticality shifted from [-1,1] to [0,1] → reverse
    out[:, 3] = out[:, 3] * 2.0 - 1.0
    # Tempo divided by 30 → multiply back to passes/minute
    out[:, 4] = out[:, 4] * 30.0
    return out


def compute_r2(actual, predicted):
    """
    R² per metric. 1.0=perfect, 0.0=mean baseline, negative=worse than mean.
    Small epsilon prevents division by zero for constant-valued targets.
    """
    ss_res = np.sum((actual - predicted) ** 2, axis=0)
    ss_tot = np.sum((actual - np.mean(actual, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def plot_confusion_matrix(ax, true_labels, pred_labels, class_names, title):
    """
    Row-normalised confusion matrix.
    Each cell: raw count on top, recall percentage below.
    Diagonal cells have green borders (correct predictions).
    """
    n = len(class_names)

    # Build raw count matrix: cm[actual_class, predicted_class]
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1

    # Row-normalise: each row sums to 1.0 → shows recall per class
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / (row_sums + 1e-8)

    # Heatmap with fixed 0→1 colour scale
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label',    fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate every cell with count and percentage
    for i in range(n):
        for j in range(n):
            # White text on dark cells for readability
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            cell_text  = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"
            ax.text(
                j, i, cell_text,
                ha='center', va='center',
                color=text_color, fontsize=9,
                # Bold on diagonal to highlight correct predictions
                fontweight='bold' if i == j else 'normal'
            )

    # Colourbar labelled as recall percentage
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Recall (%)', fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Green border on diagonal cells to visually highlight correct predictions
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=2, edgecolor='green',
            facecolor='none', boxstyle='square,pad=0'
        )
        ax.add_patch(rect)

    return im


def plot_scatter_grid(preds, targets, r2):
    """
    2×3 grid of scatter plots for 5 regression metrics.
    6th subplot shows R² summary text box.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i in range(5):
        ax = axes[i]

        # Downsample to 1000 points for rendering speed
        if len(preds) > 1000:
            idx = np.random.choice(len(preds), 1000, replace=False)
            xv, yv = targets[idx, i], preds[idx, i]
        else:
            xv, yv = targets[:, i], preds[:, i]

        # Each point: one window's actual value vs predicted value
        ax.scatter(xv, yv, alpha=0.3, s=12, color='steelblue', edgecolors='none')

        # Perfect prediction diagonal (y=x)
        combined = np.concatenate([xv, yv])
        lim = [combined.min() - 0.05, combined.max() + 0.05]
        ax.plot(lim, lim, 'r-', alpha=0.75, linewidth=1.5, zorder=0, label='Perfect')

        ax.set_title(f"{REG_NAMES[i]}  (R²={r2[i]:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Actual (next window)", fontsize=9)
        ax.set_ylabel("Predicted",            fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6th subplot: R² summary instead of blank space
    ax_summary = axes[5]
    ax_summary.axis('off')
    summary_lines = ["R² Summary — TEST SET\n(Honest held-out metrics)\n"]
    for i, name in enumerate(REG_NAMES):
        # Simple ASCII bar proportional to R²
        bar = '█' * int(max(r2[i], 0) * 20)
        summary_lines.append(f"{name:<18}  {r2[i]:+.3f}  {bar}")
    summary_lines.append("\n0.0 = predicting mean")
    summary_lines.append("1.0 = perfect prediction")
    ax_summary.text(
        0.05, 0.95, '\n'.join(summary_lines),
        transform=ax_summary.transAxes,
        fontsize=10, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.suptitle(
        f"GAT-LSTM — Regression on HELD-OUT TEST SET  (seq_len={SEQ_LEN})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("inference_regression.png", dpi=150)
    print("Saved inference_regression.png")


def plot_confusion_matrices(dp, dt, op, ot):
    """Side-by-side confusion matrices for both classification heads."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_confusion_matrix(
        axes[0], dt, dp, DEF_LABELS,
        "Defensive Posture — TEST SET\n(Low Block / Mid Block / High Press)"
    )
    plot_confusion_matrix(
        axes[1], ot, op, OFF_LABELS,
        "Offensive Style — TEST SET\n(Patient / Balanced / Counter)"
    )

    plt.suptitle(
        f"GAT-LSTM — Classification on HELD-OUT TEST SET  (seq_len={SEQ_LEN})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("inference_classification.png", dpi=150)
    print("Saved inference_classification.png")


def print_report(preds, targets, mae, r2, dp, dt, op, ot):
    """Console report: regression table + per-class precision/recall + samples."""
    print("\n" + "=" * 65)
    print("   GAT-LSTM — HELD-OUT TEST SET RESULTS")
    print("   (Model never saw these matches during training or validation)")
    print("=" * 65)

    # Regression metrics table
    print(f"\n{'--- REGRESSION (next-window prediction) ---':^65}")
    print(f"{'Metric':<20} | {'MAE':>10} | {'R²':>10} | {'Quality'}")
    print("-" * 65)
    for i, name in enumerate(REG_NAMES):
        # Simple quality label based on R² value
        if r2[i] > 0.6:    quality = "Good"
        elif r2[i] > 0.3:  quality = "Moderate"
        elif r2[i] > 0.0:  quality = "Weak"
        else:               quality = "Worse than mean"
        print(f"{name:<20} | {mae[i]:>10.4f} | {r2[i]:>10.4f} | {quality}")

    # Classification accuracy summary
    def_acc = (dp == dt).mean()
    off_acc = (op == ot).mean()
    print(f"\n{'--- CLASSIFICATION ---':^65}")
    print(f"Defensive Posture Accuracy:  {def_acc:.1%}")
    print(f"Offensive Style Accuracy:    {off_acc:.1%}")

    # Per-class precision and recall for each classifier
    for label_name, preds_cls, targets_cls, names in [
        ("Defensive Posture", dp, dt, DEF_LABELS),
        ("Offensive Style",   op, ot, OFF_LABELS),
    ]:
        print(f"\n--- {label_name} (Per-Class) ---")
        print(f"{'Class':<14} | {'Recall':>8} | {'Precision':>10} | {'Support':>8}")
        print("-" * 50)
        for c, name in enumerate(names):
            true_mask = targets_cls == c
            pred_mask = preds_cls   == c
            support   = true_mask.sum()
            if support > 0:
                # Recall: fraction of actual class c correctly predicted
                recall    = (preds_cls[true_mask] == c).mean()
                # Precision: fraction of predicted class c that were actually c
                precision = (targets_cls[pred_mask] == c).mean() if pred_mask.sum() > 0 else 0.0
                print(f"{name:<14} | {recall:>8.1%} | {precision:>10.1%} | {support:>8}")

    # 5 random samples for manual inspection
    print(f"\n{'--- SAMPLE PREDICTIONS (5 random) ---':^65}")
    indices = np.random.choice(len(preds), 5, replace=False)
    for idx in indices:
        print(f"\n  Sample {idx}:")
        for i, name in enumerate(REG_NAMES):
            a, p = targets[idx][i], preds[idx][i]
            print(f"    {name:<20} | Actual: {a:<8.3f} | Pred: {p:<8.3f} | Diff: {p-a:<+8.3f}")
        da = DEF_LABELS[int(dt[idx])]; dd = DEF_LABELS[int(dp[idx])]
        oa = OFF_LABELS[int(ot[idx])]; od = OFF_LABELS[int(op[idx])]
        print(f"    {'Def Posture':<20} | Actual: {da:<14} | Pred: {dd:<14} | {'✓' if da==dd else '✗'}")
        print(f"    {'Off Style':<20} | Actual: {oa:<14} | Pred: {od:<14} | {'✓' if oa==od else '✗'}")

    print("\n" + "=" * 65)


def run():
    print("=== GAT-LSTM INFERENCE — HELD-OUT TEST SET ===")
    print(f"Device: {DEVICE} | Seq Len: {SEQ_LEN}")
    print("NOTE: Evaluating on test matches not seen during training OR validation.")

    # Load the full base dataset
    print("\n1. Loading dataset...")
    base_dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR,
        dataset_name=DATASET_NAME, window_size=5,
        stride=1, max_matches=MAX_MATCHES
    )

    # Load the test indices from the file saved by train.py
    # This guarantees the test set is completely disjoint from both
    # the training set and the validation set used during training
    test_idx  = load_test_indices(base_dataset)
    test_base = Subset(base_dataset, test_idx)

    # Wrap in sequence dataset: (5 input windows) → predict window t+1
    test_seq = SequenceTacticalDataset(test_base, seq_len=SEQ_LEN)

    # Collate function handles batching PyG graphs across timesteps
    loader = DataLoader(
        test_seq, batch_size=32, shuffle=False,
        collate_fn=sequence_collate_fn, num_workers=0
    )

    # Load model with the same architecture used during training
    print("\n2. Loading model...")
    sample = base_dataset[0]
    model  = TacticalGATLSTM(
        num_node_features=sample.x.shape[1],
        num_reg_targets=5,
        num_def_classes=3,
        num_off_classes=3,
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1],
        lstm_hidden=128,
        lstm_layers=2,
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    # Load the saved weights — best_model.pth was selected by validation loss,
    # but we are now evaluating on the separate test set
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Collect predictions and ground truth across the full test set
    print("\n3. Running inference on held-out test set...")
    all_reg_preds, all_reg_targets = [], []
    all_def_preds, all_def_targets = [], []
    all_off_preds, all_off_targets = [], []

    with torch.no_grad():
        for sequence, target_batch in loader:
            # Move input sequence to device
            sequence     = [g.to(DEVICE) for g in sequence]
            target_batch = target_batch.to(DEVICE)

            # Predict next window from the 5-window input sequence
            reg_pred, cls_def_pred, cls_off_pred = model(sequence)

            # Target labels are from window t+1 (the one being predicted)
            reg_target  = target_batch.y.view(-1, 5)
            cls_targets = target_batch.y_cls.view(-1, 3)
            # y_cls[:, 2] is outcome — not predicted by this model

            all_reg_preds.append(reg_pred.cpu().numpy())
            all_reg_targets.append(reg_target.cpu().numpy())
            # argmax converts logits to predicted class index
            all_def_preds.append(cls_def_pred.argmax(1).cpu().numpy())
            all_def_targets.append(cls_targets[:, 0].cpu().numpy())
            all_off_preds.append(cls_off_pred.argmax(1).cpu().numpy())
            all_off_targets.append(cls_targets[:, 1].cpu().numpy())

    # Concatenate all batches into single arrays
    preds_n   = np.vstack(all_reg_preds)
    targets_n = np.vstack(all_reg_targets)
    dp = np.concatenate(all_def_preds);  dt = np.concatenate(all_def_targets)
    op = np.concatenate(all_off_preds);  ot = np.concatenate(all_off_targets)

    # Reverse normalisation to human-readable units
    preds   = denormalize(preds_n)
    targets = denormalize(targets_n)

    # Compute metrics
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2  = compute_r2(targets, preds)

    # Print full console report
    print_report(preds, targets, mae, r2, dp, dt, op, ot)

    # Generate and save plots
    print("\n4. Generating plots...")
    plot_scatter_grid(preds, targets, r2)
    plot_confusion_matrices(dp, dt, op, ot)

    print("\nDone. Output files:")
    print("  inference_regression.png     — scatter plots (test set)")
    print("  inference_classification.png — confusion matrices (test set)")


if __name__ == "__main__":
    run()