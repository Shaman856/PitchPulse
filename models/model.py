import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_reg_targets=5, 
                 num_def_classes=3, num_off_classes=3,  # kept for compatibility
                 edge_dim=5, global_dim=5):
        """
        Regression-Only GAT Model (Classification Heads Disabled)
        """
        super(TacticalGAT, self).__init__()
        
        self.global_dim = global_dim
        
        # --- 1. SPATIAL ENCODER (GAT Layers) ---
        self.gat1 = GATv2Conv(
            in_channels=num_node_features, 
            out_channels=64, 
            heads=4, 
            edge_dim=edge_dim, 
            concat=True,
            dropout=0.05
        )
        
        self.gat2 = GATv2Conv(
            in_channels=64 * 4,
            out_channels=128, 
            heads=1, 
            edge_dim=edge_dim, 
            concat=False,
            dropout=0.05
        )

        # --- 2. SHARED TRUNK ---
        trunk_dim = 128 + global_dim
        self.shared = Linear(trunk_dim, 64)

        # --- 3. REGRESSION HEAD (5 targets) ---
        self.reg_head = Linear(64, num_reg_targets)
        
        # === CLASSIFICATION DISABLED ===
        # --- 4. CLASSIFICATION HEADS ---
        # self.cls_def_head = Linear(64, num_def_classes)
        # self.cls_off_head = Linear(64, num_off_classes)

    def forward(self, data):
        """
        Returns:
            reg_out: [B, 5] - Regression predictions
        """
        x, edge_index, edge_attr, batch, u = (
            data.x, data.edge_index, data.edge_attr, data.batch, data.u
        )
        
        # --- Block 1: Graph Attention ---
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # --- Block 2: Readout ---
        x = global_mean_pool(x, batch)
        
        # --- Block 3: Fusion ---
        x = torch.cat([x, u], dim=1)
        
        # --- Block 4: Shared Trunk ---
        shared = self.shared(x)
        shared = F.relu(shared)
        
        # --- Block 5: Regression Head Only ---
        reg_out = self.reg_head(shared)
        
        # === CLASSIFICATION DISABLED ===
        # cls_def_out = self.cls_def_head(shared)
        # cls_off_out = self.cls_off_head(shared)
        # return reg_out, cls_def_out, cls_off_out
        
        return reg_out
# --- MODEL INSPECTION ---
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    dummy_x = torch.rand(12, 11) 
    dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_edge_attr = torch.rand(2, 5)
    dummy_u = torch.rand(1, 5)
    dummy_batch = torch.zeros(12, dtype=torch.long)
    
    data = Data(x=dummy_x, edge_index=dummy_edge_index, 
                edge_attr=dummy_edge_attr, 
                u=dummy_u, batch=dummy_batch)
    
    model = TacticalGAT(num_node_features=11, num_reg_targets=5, 
                        edge_dim=5, global_dim=5)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    model.eval()
    
    print("Testing Model Forward Pass...")
    
    # === CLASSIFICATION DISABLED ===
    # reg_out, cls_def, cls_off = model(data)
    
    reg_out = model(data)
    
    print(f"\n--- Model Output Check ---")
    print(f"Input Node Shape:    {dummy_x.shape}")
    print(f"Input Edge Shape:    {dummy_edge_attr.shape}")
    print(f"Input Global Shape:  {dummy_u.shape}")
    print(f"Regression Output:   {reg_out.shape} (Should be [1, 5])")
    print(f"Total Parameters:    {total_params:,}")
    print(f"\nRegression values:   {reg_out.detach().numpy()}")
    print(f"\nSuccess!")