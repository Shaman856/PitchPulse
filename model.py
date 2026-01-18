import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class PitchPulseGAT(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=1):
        """
        Args:
            in_channels: 2 (Input is just x, y coordinates)
            hidden_channels: 32 (Size of the "brain" for each node)
            out_channels: 1 (We want one single score as output: xT)
        """
        super(PitchPulseGAT, self).__init__()
        
        # --- Layer 1: The "Attention" Layer ---
        # Heads=4: The model looks at the pitch with 4 different "perspectives"
        # Input: [Nodes, 2] -> Output: [Nodes, 32 * 4]
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        
        # --- Layer 2: Refining the Features ---
        # We merge the 4 heads back into 1
        # Input: [Nodes, 128] -> Output: [Nodes, 32]
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        
        # --- Layer 3: The Prediction ---
        # A simple linear layer to give us our final score
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # 1. Unpack Data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 2. Graph Convolutions (The "Thinking" Phase)
        # The model passes messages along the edges (passes)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU is the standard activation for GATs
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # 3. Pooling (The "Summary" Phase)
        # Aggregate all player nodes into one "Team Vector"
        x = global_mean_pool(x, batch)  
        
        # 4. Final Prediction
        x = self.lin(x)
        
        return x

# --- Test Block (Run this on your RTX 3060) ---
if __name__ == "__main__":
    from graph_builder import build_graph
    from data_pipeline import fetch_match_data
    from window_slicer import get_rolling_windows
    
    # 1. Setup GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Prepare Data (One Window)
    print("\nPreparing Data...")
    df = fetch_match_data(8658)
    windows = get_rolling_windows(df)
    
    # Pick a random window (e.g., Window 10)
    data = build_graph(windows[10])
    
    # Add 'batch' vector (Required when running single graphs not in a DataLoader)
    # It tells PyG that all these nodes belong to "Graph 0"
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    # Move Data to GPU
    data = data.to(device)

    # 3. Initialize Model
    model = PitchPulseGAT().to(device)
    
    # 4. Forward Pass (The Magic Moment)
    # We feed the graph into the model and ask for a prediction
    output = model(data)
    
    print("\n--- Forward Pass Successful ---")
    print(f"Input Shape: {data.x.shape} (Players, Coords)")
    print(f"Output Shape: {output.shape} (Should be [1, 1])")
    print(f"Predicted Score (untrained): {output.item():.4f}")