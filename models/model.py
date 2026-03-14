import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_reg_targets=5, num_def_classes=3, 
                 num_off_classes=3, num_outcome_classes=3, edge_dim=5, global_dim=5):
        """
        Complete Tactical Suite GAT Model.
        
        SUITE v2 CHANGES:
        - extract_attention_weights() method replaces extract_node_embeddings()
        - Uses GATv2Conv's return_attention_weights for sharp player attribution
        - Architecture and forward pass unchanged
        """
        super(TacticalGAT, self).__init__()
        
        self.global_dim = global_dim
        
        self.gat1 = GATv2Conv(
            in_channels=num_node_features, 
            out_channels=64, heads=4, 
            edge_dim=edge_dim, concat=True, dropout=0.05
        )
        
        self.gat2 = GATv2Conv(
            in_channels=64 * 4, 
            out_channels=128, heads=1, 
            edge_dim=edge_dim, concat=False, dropout=0.05
        )

        trunk_dim = 128 + global_dim
        self.shared = Linear(trunk_dim, 64)

        self.reg_head = Linear(64, num_reg_targets)
        self.cls_def_head = Linear(64, num_def_classes)
        self.cls_off_head = Linear(64, num_off_classes)
        self.cls_outcome_head = Linear(64, num_outcome_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch, u = data.x, data.edge_index, data.edge_attr, data.batch, data.u
        
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        x = torch.cat([x, u], dim=1)
        
        shared = self.shared(x)
        shared = F.relu(shared)
        
        reg_out = self.reg_head(shared)
        cls_def_out = self.cls_def_head(shared)
        cls_off_out = self.cls_off_head(shared)
        cls_outcome_out = self.cls_outcome_head(shared)
        
        return reg_out, cls_def_out, cls_off_out, cls_outcome_out
    
    def extract_attention_weights(self, data):
        """
        SUITE v2: Extracts GAT attention weights for key player identification.
        
        Returns attention coefficients from BOTH GAT layers. The attention
        weights tell us which edges (player interactions) the model considers
        most important — much sharper than embedding magnitudes.
        
        Returns:
            attn1_index: [2, E] edge index for layer 1
            attn1_weights: [E, num_heads] attention coefficients from layer 1
            attn2_index: [2, E] edge index for layer 2
            attn2_weights: [E, 1] attention coefficients from layer 2
            node_emb: [num_nodes, 128] node embeddings after layer 2
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1 with attention weights
        x1, (attn1_index, attn1_weights) = self.gat1(
            x, edge_index, edge_attr=edge_attr, 
            return_attention_weights=True
        )
        x1 = F.elu(x1)
        
        # Layer 2 with attention weights
        x2, (attn2_index, attn2_weights) = self.gat2(
            x1, edge_index, edge_attr=edge_attr, 
            return_attention_weights=True
        )
        x2 = F.elu(x2)
        
        return attn1_index, attn1_weights, attn2_index, attn2_weights, x2
    
    def extract_node_embeddings(self, data):
        """Backward compatibility: still available for legacy code."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        return x, data.batch
