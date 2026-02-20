import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, edge_dim=5, global_dim=5):
        """
        GAT Model with dynamic dimensions.
        
        Args:
            num_node_features: Number of features per node (auto-detected)
            num_classes: Number of output targets (4)
            edge_dim: Number of edge features (auto-detected)
            global_dim: Number of global context features (auto-detected)
        """
        super(TacticalGAT, self).__init__()
        
        self.global_dim = global_dim
        
        # --- 1. SPATIAL ENCODER (GAT Layers) ---
        # Layer 1: Input -> Hidden (64 * 4 heads = 256)
        self.gat1 = GATv2Conv(
            in_channels=num_node_features, 
            out_channels=64, 
            heads=4, 
            edge_dim=edge_dim, 
            concat=True,
            dropout=0.05
        )
        
        # Layer 2: Hidden (256) -> Embedding (128)
        self.gat2 = GATv2Conv(
            in_channels=64 * 4,  # 256
            out_channels=128, 
            heads=1, 
            edge_dim=edge_dim, 
            concat=False,
            dropout=0.05
        )

        # --- 2. REGRESSION HEAD ---
        # Input: 128 (Graph Embedding) + global_dim (Global Context)
        self.lin1 = Linear(128 + global_dim, 64)
        self.head = Linear(64, num_classes) 

    def forward(self, data):
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        # --- Block 1: Graph Attention ---
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # --- Block 2: Readout ---
        x = global_mean_pool(x, batch)  # [Batch, 128]
        
        # --- Block 3: Fusion & Prediction ---
        x = torch.cat([x, u], dim=1)    # [Batch, 128 + global_dim]
        
        x = self.lin1(x)
        x = F.relu(x)
        
        return self.head(x)

# --- MODEL INSPECTION ---
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    # Dummy Batch (12 Nodes, 11 Features, 5 Edge Features, 5 Global Features)
    dummy_x = torch.rand(12, 11) 
    dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_edge_attr = torch.rand(2, 5)
    dummy_u = torch.rand(1, 5)
    dummy_batch = torch.zeros(12, dtype=torch.long)
    
    data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, u=dummy_u, batch=dummy_batch)
    
    model = TacticalGAT(num_node_features=11, num_classes=4, edge_dim=5, global_dim=5)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    model.eval()
    
    print("Testing Model Forward Pass...")
    out = model(data)
    
    print(f"\n--- Model Output Check ---")
    print(f"Input Node Shape: {dummy_x.shape}")
    print(f"Input Edge Shape: {dummy_edge_attr.shape}")
    print(f"Input Global Shape: {dummy_u.shape}")
    print(f"Output Shape: {out.shape} (Should be [1, 4])")
    print(f"Total Parameters: {total_params:,}")
    print(f"\nSuccess!")
