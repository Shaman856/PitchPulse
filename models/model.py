# models/model.py
import torch
import torch.nn.functional as F
from torch.nn import Linear, LSTM
from torch_geometric.nn import GATv2Conv, global_mean_pool


class TacticalGATLSTM(torch.nn.Module):
    """
    GAT-LSTM model for genuine next-window tactical prediction.

    FIX (vs previous version):
        Dropout increased throughout to address the ~0.09 train/val gap
        observed in the training curve. Specifically:
        - GAT layers: 0.05 → 0.15 (more regularisation on attention)
        - Feature dropout after GAT1: 0.1 → 0.2
        - LSTM inter-layer dropout: 0.1 → 0.2
        These changes reduce overfitting without reducing model capacity.
    """

    def __init__(
        self,
        num_node_features,      # Number of features per node (12 in current setup)
        num_reg_targets=5,      # Regression targets: xT, press, tilt, vert, tempo
        num_def_classes=3,      # Defensive posture: Low/Mid/High
        num_off_classes=3,      # Offensive style: Patient/Balanced/Counter
        edge_dim=9,             # Edge feature dimension (9 in current setup)
        global_dim=5,           # Global context features (5 in current setup)
        lstm_hidden=128,        # LSTM hidden state size
        lstm_layers=2,          # Stacked LSTM depth
        seq_len=5,              # Number of past windows fed to LSTM
    ):
        super(TacticalGATLSTM, self).__init__()

        # Store for use in encode_graph
        self.global_dim = global_dim
        self.seq_len    = seq_len

        # ── GAT SPATIAL ENCODER ──────────────────────────────────────────────
        # Layer 1: 12 node features → 64 * 4 heads = 256-dim
        # FIX: dropout increased from 0.05 → 0.15 to reduce overfitting
        # The attention mechanism was fitting noise in training graphs
        self.gat1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=64,
            heads=4,
            edge_dim=edge_dim,
            concat=True,        # Concatenate 4 heads → 256-dim output
            dropout=0.15,       # FIX: was 0.05 — increased for regularisation
        )

        # Layer 2: 256-dim → 128-dim, single head
        # FIX: dropout increased from 0.05 → 0.10
        self.gat2 = GATv2Conv(
            in_channels=64 * 4,   # 256 from layer 1
            out_channels=128,
            heads=1,
            edge_dim=edge_dim,
            concat=False,         # Single head → 128-dim output
            dropout=0.10,         # FIX: was 0.05 — increased for regularisation
        )

        # ── LSTM TEMPORAL MODULE ─────────────────────────────────────────────
        # Input: 128 (GAT team embedding) + 5 (global context) = 133 per timestep
        # FIX: LSTM dropout increased from 0.1 → 0.2
        # LSTM was memorising match-specific patterns rather than general tactics
        self.lstm = LSTM(
            input_size=128 + global_dim,   # 133-dim per timestep
            hidden_size=lstm_hidden,        # 128-dim hidden state
            num_layers=lstm_layers,         # 2 stacked layers
            batch_first=True,              # Shape: [batch, seq_len, features]
            dropout=0.20,                  # FIX: was 0.10 — applied between LSTM layers
        )

        # ── SHARED MLP TRUNK ─────────────────────────────────────────────────
        # Compresses LSTM's last hidden state to 64-dim shared representation
        # All prediction heads branch from this shared layer
        self.shared = Linear(lstm_hidden, 64)

        # ── PREDICTION HEADS ─────────────────────────────────────────────────
        # Outcome head intentionally removed — see MODEL_LIMITATIONS for reasoning

        # Regression: 5 continuous tactical metrics for the next window
        self.reg_head = Linear(64, num_reg_targets)

        # Defensive posture classification: Low Block / Mid Block / High Press
        self.cls_def_head = Linear(64, num_def_classes)

        # Offensive style classification: Patient / Balanced / Counter
        self.cls_off_head = Linear(64, num_off_classes)
        # Apply principled weight initialisation
        self._init_weights()

    def _init_weights(self):
        """
        Xavier for linear layers: keeps gradient variance stable at initialisation.
        Kaiming for GAT projections: better for layers followed by ELU activation.
        """
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif hasattr(module, 'lin_l'):
                # GATv2Conv left projection
                torch.nn.init.kaiming_uniform_(module.lin_l.weight, nonlinearity='relu')
            elif hasattr(module, 'lin_r'):
                # GATv2Conv right projection
                torch.nn.init.kaiming_uniform_(module.lin_r.weight, nonlinearity='relu')

    def encode_graph(self, graph):
        """
        Encode a single window's graph into a 133-dim team embedding.

        This is the spatial encoder. It reads the 12-node passing/action
        graph and compresses it into one vector representing the team's
        tactical shape in that 5-minute window.

        Args:
            graph: A batched PyG Data object for one timestep

        Returns:
            Tensor of shape [batch_size, 133]
        """
        x          = graph.x            # Node features [N*B, 12]
        edge_index = graph.edge_index   # Edge connectivity [2, E]
        edge_attr  = graph.edge_attr    # Edge features [E, 9]
        batch      = graph.batch        # Node-to-graph mapping [N*B]
        u          = graph.u            # Global context [B, 5]

        # GAT layer 1: attention-weighted message passing across all edges
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        # ELU activation: handles negative values better than ReLU for GNNs
        x = F.elu(x)
        # FIX: dropout increased to 0.2 during training only
        x = F.dropout(x, p=0.20, training=self.training)

        # GAT layer 2: compresses 256-dim to 128-dim per node
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # Global mean pool: average all 12 node embeddings into one team vector
        # Shape after pooling: [batch_size, 128]
        x = global_mean_pool(x, batch)

        # Append global context (score, period, match progress, etc.)
        # Shape after concat: [batch_size, 133]
        x = torch.cat([x, u], dim=1)

        return x

    def forward(self, sequence):
        """
        Process a sequence of consecutive window graphs → predict the NEXT window.

        Args:
            sequence: List of seq_len batched PyG Data objects
                      sequence[0] = oldest window (t - seq_len + 1)
                      sequence[-1] = most recent window (t)
                      Prediction target = window t+1 (not in sequence)

        Returns:
            reg_out:     [batch, 5]  — regression predictions
            cls_def_out: [batch, 3]  — defensive posture logits
            cls_off_out: [batch, 3]  — offensive style logits
        """
        # Encode every window through the shared GAT spatial encoder
        embeddings = []
        for graph in sequence:
            # Each call returns [batch_size, 133]
            emb = self.encode_graph(graph)
            embeddings.append(emb)

        # Stack along the time dimension → [batch_size, seq_len, 133]
        seq_tensor = torch.stack(embeddings, dim=1)

        # LSTM processes the full sequence and learns temporal patterns
        # lstm_out shape: [batch_size, seq_len, lstm_hidden]
        lstm_out, _ = self.lstm(seq_tensor)

        # Take the last timestep: this is the LSTM's summary after seeing the sequence
        # Shape: [batch_size, lstm_hidden] = [batch_size, 128]
        last_hidden = lstm_out[:, -1, :]

        # Shared MLP trunk: compresses to 64-dim
        shared = F.relu(self.shared(last_hidden))

        # Each head independently predicts from the shared representation
        reg_out     = self.reg_head(shared)
        cls_def_out = self.cls_def_head(shared)
        cls_off_out = self.cls_off_head(shared)

        return reg_out, cls_def_out, cls_off_out

    def extract_attention_weights(self, graph):
        """
        Extract GAT attention weights for key player identification.
        Used by key_player.py — operates on a single graph, not a sequence.
        """
        x          = graph.x
        edge_index = graph.edge_index
        edge_attr  = graph.edge_attr

        # Layer 1 with attention weight extraction
        x1, (attn1_index, attn1_weights) = self.gat1(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )
        x1 = F.elu(x1)

        # Layer 2 with attention weight extraction
        x2, (attn2_index, attn2_weights) = self.gat2(
            x1, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )
        x2 = F.elu(x2)

        return attn1_index, attn1_weights, attn2_index, attn2_weights, x2