# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from collections import Counter

from preprocessing.dataset import TacticalDataset
from preprocessing.sequence_dataset import SequenceTacticalDataset, sequence_collate_fn
from models.model import TacticalGATLSTM

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH  = "./data_v3"
DATASET_NAME  = "offline_mix_v7_suite"
RAW_DATA_DIR  = "./data/raw_events"
MAX_MATCHES   = 230
SEQ_LEN       = 5
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
EPOCHS        = 80
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0
PATIENCE      = 12
MIN_DELTA     = 0.0005

# Path where the 3-way split is saved so inference.py can load the test set
# without any risk of using training or validation matches
SPLIT_SAVE_PATH = "train_val_test_split.json"

# Regression loss weights: [xT, press_height, field_tilt, verticality, tempo]
REG_WEIGHTS    = torch.tensor([2.5, 1.0, 1.5, 1.5, 1.5]).to(DEVICE)
FOCAL_GAMMA    = 2.0
HUBER_DELTA    = 0.15
CLS_WEIGHT_DEF = 0.5
CLS_WEIGHT_OFF = 1.0


def make_three_way_split(base_dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Creates a 3-way match-level split: train / validation / test.

    Why 3-way instead of 2-way:
        In the old 2-way split, the validation set served two purposes:
        (1) guiding early stopping and (2) being reported as the final
        test metrics. This is circular — the model was selected based on
        performance on the same set it is evaluated on, making the
        reported metrics optimistic.

        A 3-way split fixes this:
        - Train:      model sees these examples, gradients flow
        - Validation: model never trains on these, but early stopping
                      uses them — so the model is still implicitly
                      tuned toward this set
        - Test:       completely untouched during all of training.
                      inference.py uses ONLY this set.
                      These metrics are honest.

    Split ratios:
        70% train / 15% validation / 15% test
        With 230 matches: ~161 train / ~35 val / ~34 test

    Saves all three sets to SPLIT_SAVE_PATH so inference.py can load
    the test set directly without any risk of accidentally using
    training or validation matches.

    Args:
        base_dataset: Full TacticalDataset
        train_ratio:  Fraction for training (default 0.70)
        val_ratio:    Fraction for validation (default 0.15)
                      test_ratio is implicitly 1 - train_ratio - val_ratio = 0.15
        seed:         Random seed for reproducibility

    Returns:
        train_idx: Window indices for training
        val_idx:   Window indices for validation
        test_idx:  Window indices for testing (held out completely)
    """
    # Collect the match_id for every window in the dataset
    match_ids = [int(base_dataset[i].match_id) for i in range(len(base_dataset))]

    # Get the sorted unique match IDs
    unique_matches = sorted(set(match_ids))
    total_matches  = len(unique_matches)

    # Reproducible shuffle using fixed seed
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_matches)

    # Compute match-level split boundaries
    n_train = int(total_matches * train_ratio)
    n_val   = int(total_matches * val_ratio)
    # Test gets whatever remains to ensure all matches are assigned
    # This avoids off-by-one rounding issues

    # Assign match IDs to each split
    train_matches = set(unique_matches[:n_train])
    val_matches   = set(unique_matches[n_train: n_train + n_val])
    test_matches  = set(unique_matches[n_train + n_val:])

    # Verify the three sets are disjoint (no match in more than one split)
    assert len(train_matches & val_matches)  == 0, "Train/Val overlap detected"
    assert len(train_matches & test_matches) == 0, "Train/Test overlap detected"
    assert len(val_matches   & test_matches) == 0, "Val/Test overlap detected"

    # Save the split to disk so inference.py loads it without recomputation
    # This is critical — if inference.py recomputed the split, a change in
    # MAX_MATCHES or seed would silently shift which matches are in the test set
    split_data = {
        "train_match_ids": sorted(list(train_matches)),
        "val_match_ids":   sorted(list(val_matches)),
        "test_match_ids":  sorted(list(test_matches)),
        "train_ratio":     train_ratio,
        "val_ratio":       val_ratio,
        "test_ratio":      round(1.0 - train_ratio - val_ratio, 4),
        "seed":            seed,
        "total_matches":   total_matches,
    }
    with open(SPLIT_SAVE_PATH, 'w') as f:
        json.dump(split_data, f, indent=2)

    # Derive window-level indices from match-level sets
    train_idx = [i for i, m in enumerate(match_ids) if m in train_matches]
    val_idx   = [i for i, m in enumerate(match_ids) if m in val_matches]
    test_idx  = [i for i, m in enumerate(match_ids) if m in test_matches]

    # Print split summary for verification
    print(f"3-Way Match-Level Split:")
    print(f"  Train:      {len(train_matches):4d} matches | {len(train_idx):6d} windows  "
          f"({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_matches):4d} matches | {len(val_idx):6d} windows  "
          f"({val_ratio*100:.0f}%) ← used for early stopping only")
    print(f"  Test:       {len(test_matches):4d} matches | {len(test_idx):6d} windows  "
          f"({(1-train_ratio-val_ratio)*100:.0f}%) ← NEVER seen during training")
    print(f"  Split saved to: {SPLIT_SAVE_PATH}")

    return train_idx, val_idx, test_idx


def focal_loss(logits, targets, gamma=2.0):
    """
    Focal loss for classification heads.

    Downweights easy correctly-classified examples so the model
    focuses training budget on hard boundary cases (Mid Block, Balanced).
    Factor (1-p)^gamma: p=0.9 (easy) → 0.01 weight. p=0.1 (hard) → 0.81 weight.
    """
    # Per-sample cross-entropy, no reduction yet
    ce_loss = F.cross_entropy(logits, targets, reduction='none')

    # Probability assigned to the correct class
    pt = torch.exp(-ce_loss)

    # Downweight easy examples exponentially
    focal = (1.0 - pt) ** gamma * ce_loss

    return focal.mean()


def weighted_mse_with_huber_xt(pred, target, weights, huber_delta):
    """
    Hybrid regression loss:
        Column 0 (xT):   Huber — robust to rare high-xT outlier windows
        Columns 1-4:     Weighted MSE — standard regression
    """
    loss = torch.zeros_like(pred)

    # Huber for xT: quadratic below delta, linear above
    diff     = pred[:, 0] - target[:, 0]
    abs_diff = torch.abs(diff)
    loss[:, 0] = torch.where(
        abs_diff <= huber_delta,
        0.5 * diff ** 2,
        huber_delta * (abs_diff - 0.5 * huber_delta)
    )

    # MSE for the other 4 regression targets
    loss[:, 1:] = (pred[:, 1:] - target[:, 1:]) ** 2

    # Weight each target and average over batch
    return (loss * weights).mean()


def composite_loss(reg_pred, reg_target,
                   cls_def_pred, cls_def_target,
                   cls_off_pred, cls_off_target):
    """
    Combined regression + focal classification loss.
    Outcome head intentionally removed.
    """
    loss_reg = weighted_mse_with_huber_xt(reg_pred, reg_target, REG_WEIGHTS, HUBER_DELTA)
    loss_def = focal_loss(cls_def_pred, cls_def_target, gamma=FOCAL_GAMMA)
    loss_off = focal_loss(cls_off_pred, cls_off_target, gamma=FOCAL_GAMMA)

    total = loss_reg + CLS_WEIGHT_DEF * loss_def + CLS_WEIGHT_OFF * loss_off
    return total, loss_reg.item(), loss_def.item(), loss_off.item()


def train():
    print(f"=== GAT-LSTM TRAINING (3-way split: train/val/test) ===")
    print(f"Device: {DEVICE} | Seq Len: {SEQ_LEN} | Batch: {BATCH_SIZE}")
    print(f"Fixes: focal_loss(γ={FOCAL_GAMMA}) + 3-way split + increased dropout")

    # Load the base per-window dataset
    print("\nLoading base window dataset...")
    base_dataset = TacticalDataset(
        root=DATASET_PATH, raw_dir=RAW_DATA_DIR,
        dataset_name=DATASET_NAME, window_size=5,
        stride=1, max_matches=MAX_MATCHES
    )

    # Auto-detect feature dimensions from a real sample to avoid hardcoding
    sample = base_dataset[0]
    print(f"  Node features: {sample.x.shape[1]} | "
          f"Edge features: {sample.edge_attr.shape[1]} | "
          f"Global features: {sample.u.shape[1]}")

    # Create and save the 3-way split
    # test_idx is saved to disk and NOT used here during training at all
    train_idx, val_idx, test_idx = make_three_way_split(
        base_dataset, train_ratio=0.70, val_ratio=0.15, seed=42
    )

    # Create Subset views — no data is copied, only index lists
    train_base = Subset(base_dataset, train_idx)
    val_base   = Subset(base_dataset, val_idx)
    # test_base intentionally NOT created here — inference.py handles it

    # Wrap in sequence dataset for next-window prediction
    # Each item: (5 consecutive input windows, target window t+1)
    print("\nBuilding sequence datasets...")
    train_seq = SequenceTacticalDataset(train_base, seq_len=SEQ_LEN)
    val_seq   = SequenceTacticalDataset(val_base,   seq_len=SEQ_LEN)

    # Use custom collate_fn so PyG graphs batch correctly across timesteps
    train_loader = DataLoader(
        train_seq, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_seq, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_collate_fn, num_workers=0
    )

    # Initialise model with auto-detected dimensions
    model = TacticalGATLSTM(
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # AdamW: Adam with decoupled weight decay regularisation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    # Cosine annealing: smooth LR decay from LEARNING_RATE → ~0 over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # Tracking variables
    train_losses, val_losses   = [], []
    val_def_accs, val_off_accs = [], []
    best_val_loss    = float('inf')
    patience_counter = 0
    stopped_epoch    = EPOCHS

    for epoch in range(EPOCHS):
        # ── TRAINING ──────────────────────────────────────────────────────
        # Model sees training data only — gradients flow from training loss
        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for sequence, target_batch in loop:
            # Move the sequence and target to GPU/CPU
            sequence     = [g.to(DEVICE) for g in sequence]
            target_batch = target_batch.to(DEVICE)

            # Forward pass: predict window t+1 from windows t-4 to t
            reg_pred, cls_def_pred, cls_off_pred = model(sequence)

            # Labels come from the TARGET window (t+1), not the input sequence
            reg_target     = target_batch.y.view(-1, 5)
            cls_targets    = target_batch.y_cls.view(-1, 3)
            cls_def_target = cls_targets[:, 0]   # Defensive posture at t+1
            cls_off_target = cls_targets[:, 1]   # Offensive style at t+1

            loss, _, _, _ = composite_loss(
                reg_pred, reg_target,
                cls_def_pred, cls_def_target,
                cls_off_pred, cls_off_target,
            )

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent LSTM gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── VALIDATION ────────────────────────────────────────────────────
        # Validation set guides early stopping — model never trains on these
        # matches, but best_model.pth is selected based on val performance.
        # This is why we need a separate test set for honest final evaluation.
        model.eval()
        total_val_loss = 0
        correct_def, correct_off, total_samples = 0, 0, 0

        with torch.no_grad():
            for sequence, target_batch in val_loader:
                sequence     = [g.to(DEVICE) for g in sequence]
                target_batch = target_batch.to(DEVICE)

                reg_pred, cls_def_pred, cls_off_pred = model(sequence)

                reg_target     = target_batch.y.view(-1, 5)
                cls_targets    = target_batch.y_cls.view(-1, 3)
                cls_def_target = cls_targets[:, 0]
                cls_off_target = cls_targets[:, 1]

                loss, _, _, _ = composite_loss(
                    reg_pred, reg_target,
                    cls_def_pred, cls_def_target,
                    cls_off_pred, cls_off_target,
                )

                total_val_loss += loss.item()
                correct_def    += (cls_def_pred.argmax(1) == cls_def_target).sum().item()
                correct_off    += (cls_off_pred.argmax(1) == cls_off_target).sum().item()
                total_samples  += cls_def_target.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        def_acc = correct_def / total_samples
        off_acc = correct_off / total_samples

        val_losses.append(avg_val_loss)
        val_def_accs.append(def_acc)
        val_off_accs.append(off_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | DefAcc: {def_acc:.1%} | "
              f"OffAcc: {off_acc:.1%} | LR: {current_lr:.6f}")

        # Advance cosine LR schedule after each epoch
        scheduler.step()

        # Save best model based on validation loss
        # Note: this selection is based on val set, not test set
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> New Best Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                stopped_epoch = epoch + 1
                break

    # ── PLOT ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves — validation here is the val set, NOT the test set
    axes[0].plot(train_losses, label='Train Loss',      color='steelblue')
    axes[0].plot(val_losses,   label='Val Loss (early stopping set)', color='orange')
    if stopped_epoch < EPOCHS:
        axes[0].axvline(
            x=stopped_epoch - 1, color='red',
            linestyle='--', alpha=0.5,
            label=f'Early Stop ({stopped_epoch})'
        )
    axes[0].set_title('GAT-LSTM Loss (Val = early stopping set only)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Classification accuracy on the validation set
    axes[1].plot(val_def_accs, label='Defensive Posture', color='red')
    axes[1].plot(val_off_accs, label='Offensive Style',   color='blue')
    axes[1].set_title('Classification Accuracy — Validation Set\n(not the final test set)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"GAT-LSTM — 3-way split | seq_len={SEQ_LEN} | stopped epoch {stopped_epoch}\n"
        f"Run inference.py for final honest test metrics",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    print(f"\nTraining complete. Saved training_curve.png")
    print(f"Split saved to: {SPLIT_SAVE_PATH}")
    print(f"Run inference.py to evaluate on the held-out test set.")


if __name__ == "__main__":
    train()
