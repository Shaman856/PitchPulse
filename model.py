import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class PitchPulseGAT(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=1):
        super(PitchPulseGAT, self).__init__()
        
        # Layer 1: Attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        
        # Layer 2: Refinement
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        
        # Layer 3: Prediction
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data, return_attention=False):
        """
        Modified Forward Pass.
        If return_attention=True, it returns the Attention Weights of the first layer
        instead of the final prediction.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Attention Layer
        # If we want to visualize, we ask GATConv to return the weights
        if return_attention:
            # return_attention_weights=True gives us ((edge_indices), (weights))
            x, (att_edge_index, att_weights) = self.conv1(x, edge_index, return_attention_weights=True)
            return att_edge_index, att_weights
            
        # Normal Training Path
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # 2. Refinement Layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # 3. Pooling
        x = global_mean_pool(x, batch)  
        
        # 4. Prediction
        x = self.lin(x)
        
        return x