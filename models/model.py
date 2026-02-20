import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_reg_targets=5, num_def_classes=3, 
                 num_off_classes=3, edge_dim=5, global_dim=5):
        """
        Dual-Head GAT Model for the Tactical Suite.
        
        Architecture:
          Shared Backbone: GAT (2 layers) → Global Mean Pool → Fusion with global context
          Head 1 (Regression):     Predicts 5 continuous tactical metrics
          Head 2 (Classification): Predicts Defensive Posture (3 classes)
          Head 3 (Classification): Predicts Offensive Style (3 classes)
        
        Args:
            num_node_features: Node feature dimension (auto-detected, typically 11)
            num_reg_targets:   Number of regression outputs (5)
            num_def_classes:   Classes for defensive posture (3: Low/Mid/High)
            num_off_classes:   Classes for offensive style (3: Patient/Balanced/Counter)
            edge_dim:          Edge feature dimension (auto-detected, typically 5)
            global_dim:        Global context dimension (auto-detected, typically 5)
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

        # --- 2. SHARED TRUNK ---
        # Input: 128 (Graph Embedding) + global_dim (Global Context)
        trunk_dim = 128 + global_dim
        self.shared = Linear(trunk_dim, 64)

        # --- 3. REGRESSION HEAD (5 targets) ---
        self.reg_head = Linear(64, num_reg_targets)
        
        # --- 4. CLASSIFICATION HEADS ---
        # Defensive Posture: 3 classes (Low Block / Mid Block / High Press)
        self.cls_def_head = Linear(64, num_def_classes)
        
        # Offensive Style: 3 classes (Patient / Balanced / Counter)
        self.cls_off_head = Linear(64, num_off_classes)

    def forward(self, data):
        """
        Returns:
            reg_out:     [B, 5]  - Regression predictions
            cls_def_out: [B, 3]  - Defensive posture logits
            cls_off_out: [B, 3]  - Offensive style logits
        """
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        # --- Block 1: Graph Attention ---
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # --- Block 2: Readout ---
        x = global_mean_pool(x, batch)  # [Batch, 128]
        
        # --- Block 3: Fusion ---
        x = torch.cat([x, u], dim=1)    # [Batch, 128 + global_dim]
        
        # --- Block 4: Shared Trunk ---
        shared = self.shared(x)
        shared = F.relu(shared)          # [Batch, 64]
        
        # --- Block 5: Task-Specific Heads ---
        reg_out = self.reg_head(shared)          # [Batch, 5]
        cls_def_out = self.cls_def_head(shared)  # [Batch, 3]
        cls_off_out = self.cls_off_head(shared)  # [Batch, 3]
        
        return reg_out, cls_def_out, cls_off_out

# --- MODEL INSPECTION ---
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    # Dummy Batch (12 Nodes, 11 Features, 5 Edge Features, 5 Global Features)
    dummy_x = torch.rand(12, 11) 
    dummy_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_edge_attr = torch.rand(2, 5)
    dummy_u = torch.rand(1, 5)
    dummy_batch = torch.zeros(12, dtype=torch.long)
    
    data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, 
                u=dummy_u, batch=dummy_batch)
    
    model = TacticalGAT(num_node_features=11, num_reg_targets=5, edge_dim=5, global_dim=5)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    model.eval()
    
    print("Testing Model Forward Pass...")
    reg_out, cls_def, cls_off = model(data)
    
    print(f"\n--- Model Output Check ---")
    print(f"Input Node Shape:    {dummy_x.shape}")
    print(f"Input Edge Shape:    {dummy_edge_attr.shape}")
    print(f"Input Global Shape:  {dummy_u.shape}")
    print(f"Regression Output:   {reg_out.shape} (Should be [1, 5])")
    print(f"Def Class Output:    {cls_def.shape} (Should be [1, 3])")
    print(f"Off Class Output:    {cls_off.shape} (Should be [1, 3])")
    print(f"Total Parameters:    {total_params:,}")
    print(f"\nRegression values:   {reg_out.detach().numpy()}")
    print(f"Def Class logits:    {cls_def.detach().numpy()}")
    print(f"Off Class logits:    {cls_off.detach().numpy()}")
    print(f"\nSuccess!")