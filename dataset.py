import torch
import os
import glob
import pandas as pd
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

# --- IMPORT YOUR PIPELINE MODULES ---
# Ensure these match your local filenames
from data_pipeline import fetch_match_data
from utils import encode_features
from window_slicer import get_rolling_windows
from graph_builder import build_graph_from_window

class TacticalDataset(InMemoryDataset):
    def __init__(self, root, raw_dir, dataset_name, window_size=5, stride=1, transform=None, pre_transform=None):
        """
        Args:
            root (str): Folder where the processed .pt file will be saved.
            raw_dir (str): Path to the folder containing raw .pkl files (e.g. "./data/raw_events")
            dataset_name (str): Unique name for this collection (e.g. "offline_mix").
            window_size (int): Size of window in minutes.
            stride (int): Step size in minutes.
        """
        self.raw_event_dir = raw_dir
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.stride = stride
        
        # Initialize the Parent Class
        # This checks if the processed file exists. If not, it calls process()
        super().__init__(root, transform, pre_transform)
        
        # --- FIX FOR PYTORCH 2.6+ ---
        # Explicitly set weights_only=False to allow loading PyG Graph objects
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We don't rely on the automatic download/check mechanism for raw files
        return []

    @property
    def processed_file_names(self):
        # Naming the file based on config ensures we re-process if params change
        return [f'tactical_{self.dataset_name}_w{self.window_size}_s{self.stride}.pt']

    def download(self):
        # Data is already downloaded via download_raw.py
        pass

    def process(self):
        print(f"\n[Dataset] Initializing Offline Processing for {self.dataset_name}...")
        print(f"[Config] Window: {self.window_size}m | Stride: {self.stride}m")
        print(f"[Source] Reading raw events from: {self.raw_event_dir}")
        
        # 1. Get all .pkl files in the raw folder
        # This is instantaneous compared to API calls
        file_paths = glob.glob(os.path.join(self.raw_event_dir, "*.pkl"))
        
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .pkl files found in {self.raw_event_dir}. Did you run download_raw.py?")
            
        print(f"[Dataset] Found {len(file_paths)} local match files. Starting pipeline...")
        
        data_list = []
        
        # 2. Process Local Files
        for file_path in tqdm(file_paths, desc="Building Graphs"):
            try:
                # Extract Match ID from filename (e.g., "8658.pkl" -> 8658)
                match_id_str = os.path.basename(file_path).replace(".pkl", "")
                match_id = int(match_id_str)
                
                # --- A. LOAD FROM DISK (Instant) ---
                # We read the Pickle file directly into a DataFrame
                raw_events_df = pd.read_pickle(file_path)
                
                # --- B. PIPELINE (Inject Raw Data) ---
                # We pass the loaded DF so fetch_match_data DOES NOT call the API
                # NOTE: Ensure you updated data_pipeline_updated.py to accept 'raw_events'
                processed_data = fetch_match_data(match_id, raw_events=raw_events_df)
                
                if processed_data['passes'].empty:
                    continue

                # --- C. ENCODE & SLICE ---
                processed_data['passes'] = encode_features(processed_data['passes'])
                
                # Slice the match into windows
                windows = get_rolling_windows(processed_data, match_id, self.window_size, self.stride)
                
                # --- D. BUILD GRAPH ---
                for window in windows:
                    graph = build_graph_from_window(window)
                    
                    # Validation check (12 Nodes)
                    if graph.x.shape[0] == 12:
                        data_list.append(graph)
                        
            except Exception as e:
                # Print error but keep going so one bad file doesn't stop the whole process
                # print(f"[!] Error processing {file_path}: {e}")
                continue

        # 3. Save to Disk
        print(f"\n[Dataset] Collating {len(data_list)} graphs...")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # PyG optimization: Merge thousands of graphs into one object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Dataset] Success! Saved to {self.processed_paths[0]}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # 1. Define Paths
    # This must match where 'download_raw.py' saved the files
    RAW_DIR = "./data/raw_events" 
    
    # This is where the processed .pt file will be saved
    ROOT_DIR = "./data_v3"        
    
    # Name for this dataset configuration
    NAME = "offline_mix"
    
    print(f"--- STARTING OFFLINE DATASET BUILD ({NAME}) ---")
    
    # 2. Run
    dataset = TacticalDataset(
        root=ROOT_DIR, 
        raw_dir=RAW_DIR, 
        dataset_name=NAME,
        window_size=5, 
        stride=1
    )
    
    print(f"\nDataset Ready!")
    print(f"Total Graphs: {len(dataset)}")
    print(f"Features: {dataset.num_features}")
    print(f"Saved at: {dataset.processed_paths[0]}")