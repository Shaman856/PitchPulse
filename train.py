import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- IMPORTS ---
from preprocessing.dataset import TacticalDataset
from models.model import TacticalGAT


def match_level_split(dataset, train_ratio=0.8, seed=42):
    """
    Splits dataset by MATCH, not by window.
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

    print(f"Match-Level Split: {len(train_matches)} train matches, {len(test_matches)} test matches")
    print(f"  Train windows: {len(train_indices)} | Test windows: {len(test_indices)}")

    return train_indices, test_indices


# --- CONFIGURATION ---
DATASET_PATH = "./data_v3"
DATASET_NAME = "offline_mix_v4"
RAW_DATA_DIR = "./data/raw_events"
MAX_MATCHES = 230
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRADIENT_CLIP = 1.0

# --- LOSS CONFIGURATION ---
REG_WEIGHTS = torch.tensor([2.5, 1.0, 1.5, 1.5, 1.5]).to(DEVICE)

# === CLASSIFICATION DISABLED ===
# CLS_WEIGHT_DEF = 0.5
# CLS_WEIGHT_OFF = 1.0


def weighted_mse_loss(pred, target, weights):
    """Per-column weighted MSE loss."""
    loss = (pred - target) ** 2
    return (loss * weights).mean()


# === CLASSIFICATION DISABLED ===
# def composite_loss(...):
#     ...


def train():
    print(f"--- STARTING REGRESSION-ONLY TRAINING ON {DEVICE} ---")

    # 1. Load Data
    dataset = TacticalDataset(
        root=DATASET_PATH,
        raw_dir=RAW_DATA_DIR,
        dataset_name=DATASET_NAME,
        window_size=5,
        stride=1,
        max_matches=MAX_MATCHES
    )

    # 2. Match-level split
    train_indices, test_indices = match_level_split(dataset, train_ratio=0.8, seed=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Train Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    sample = dataset[0]

    model = TacticalGAT(
        num_node_features=sample.x.shape[1],
        num_reg_targets=5,
        edge_dim=sample.edge_attr.shape[1],
        global_dim=sample.u.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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

            # === CLASSIFICATION DISABLED ===
            # reg_pred, cls_def_pred, cls_off_pred = model(batch)

            reg_pred = model(batch)
            reg_target = batch.y.view(-1, 5)

            loss = weighted_mse_loss(reg_pred, reg_target, REG_WEIGHTS)

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

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)

                # === CLASSIFICATION DISABLED ===
                # reg_pred, cls_def_pred, cls_off_pred = model(batch)

                reg_pred = model(batch)
                reg_target = batch.y.view(-1, 5)

                loss = weighted_mse_loss(reg_pred, reg_target, REG_WEIGHTS)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> New Best Model Saved!")

    # --- PLOT RESULTS ---
    print("\nTraining Complete. Plotting...")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Regression Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    print("Saved training_curve.png")
    plt.show()


if __name__ == "__main__":
    train()