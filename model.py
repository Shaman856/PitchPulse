import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, edge_dim=3):
        """
        Refined GAT-LSTM Model.
        Fixes:
        1. Removed "Killing ReLU" before final head.
        2. Added Dropout for regularization.
        3. Tuned hidden dimensions for 12-node graphs.
        """
        super(TacticalGAT, self).__init__()
        
        # --- 1. SPATIAL ENCODER (GAT Layers) ---
        # Layer 1: Input (3) -> Hidden (32 * 4 heads = 128)
        self.gat1 = GATv2Conv(
            in_channels=num_node_features, 
            out_channels=32, 
            heads=4, 
            edge_dim=edge_dim, 
            concat=True,
            dropout=0.2 # Internal attention dropout
        )
        
        # Layer 2: Hidden (128) -> Embedding (64)
        self.gat2 = GATv2Conv(
            in_channels=32 * 4, # 128
            out_channels=64, 
            heads=1, 
            edge_dim=edge_dim, 
            concat=False,
            dropout=0.2
        )

        # --- 2. REGRESSION HEAD (Predictor) ---
        # Input: 64 (Graph Embedding) + 1 (Global Context u) = 65
        self.lin1 = Linear(65, 32)
        
        # Output: 4 Metrics
        # We remove the extra layer to keep gradients cleaner for regression
        self.head = Linear(32, num_classes) 

    def forward(self, data):
        """
        Flow: Nodes -> Edges -> Graph Embedding -> Concat(Context) -> Prediction
        """
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        # --- Block 1: Graph Attention ---
        # GAT 1
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) # ELU is often better than ReLU for GNNs (handles negatives better)
        x = F.dropout(x, p=0.2, training=self.training) # Feature Dropout
        
        # GAT 2
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # --- Block 2: Readout (Global Pooling) ---
        # Averages the 12 nodes into 1 "Team Vector" [Batch, 64]
        x = global_mean_pool(x, batch)
        
        # --- Block 3: Fusion & Prediction ---
        # Inject Global Context (Opponent Density)
        x = torch.cat([x, u], dim=1) # [Batch, 65]

        # Final Regression MLP
        x = self.lin1(x)
        x = F.relu(x) # This ReLU is fine (hidden layer)
        
        # FIX: No ReLU/Dropout immediately before the final projection
        # This allows the Linear layer to see the full range of signals
        return self.head(x)

# --- MODEL INSPECTION ---
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    # Dummy Batch (12 Nodes, 3 Features)
    dummy_x = torch.rand(12, 3) 
    dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_edge_attr = torch.rand(2, 3)
    dummy_u = torch.rand(1, 1)
    dummy_batch = torch.zeros(12, dtype=torch.long)
    
    data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, u=dummy_u, batch=dummy_batch)
    
    model = TacticalGAT(num_node_features=3, num_classes=4, edge_dim=3)
    
    # Switch to eval mode (disables dropout for this test)
    model.eval()
    
    print("Testing Model Forward Pass...")
    out = model(data)
    
    print("\n--- Model Output Check ---")
    print(f"Input Shape: {dummy_x.shape}")
    print(f"Output Shape: {out.shape} (Should be [1, 4])")
    print(f"Predictions: {out.detach().numpy()}")
    print("\nSuccess! The architecture is valid.")