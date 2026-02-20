import torch
import os
import glob
import pandas as pd
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

# --- IMPORT YOUR PIPELINE MODULES ---
from .data_pipeline import fetch_match_data
from .utils import encode_features
from .window_slicer import get_rolling_windows
from .graph_builder import build_graph_from_window

class TacticalDataset(InMemoryDataset):
    def __init__(self, root, raw_dir, dataset_name, window_size=5, stride=1, 
                 max_matches=None, transform=None, pre_transform=None):
        """
        Args:
            root (str): Folder where the processed .pt file will be saved.
            raw_dir (str): Path to the folder containing raw .pkl files
            dataset_name (str): Unique name for this collection (e.g. "offline_mix_v2").
            window_size (int): Size of window in minutes.
            stride (int): Step size in minutes.
            max_matches (int or None): Limit number of matches to process.
        """
        self.raw_event_dir = raw_dir
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.stride = stride
        self.max_matches = max_matches
        
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        match_tag = f"_m{self.max_matches}" if self.max_matches else "_all"
        return [f'tactical_{self.dataset_name}_w{self.window_size}_s{self.stride}{match_tag}.pt']

    def download(self):
        pass

    def process(self):
        print(f"\n[Dataset] Initializing Offline Processing for {self.dataset_name}...")
        print(f"[Config] Window: {self.window_size}m | Stride: {self.stride}m")
        print(f"[Source] Reading raw events from: {self.raw_event_dir}")
        
        file_paths = sorted(glob.glob(os.path.join(self.raw_event_dir, "*.pkl")))
        
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .pkl files found in {self.raw_event_dir}. Did you run download_raw.py?")
        
        if self.max_matches and self.max_matches < len(file_paths):
            file_paths = file_paths[:self.max_matches]
            print(f"[Dataset] Limited to {self.max_matches} matches (out of {len(glob.glob(os.path.join(self.raw_event_dir, '*.pkl')))} available)")
        else:
            print(f"[Dataset] Using all {len(file_paths)} match files")
        
        data_list = []
        
        for file_path in tqdm(file_paths, desc="Building Graphs"):
            try:
                match_id_str = os.path.basename(file_path).replace(".pkl", "")
                match_id = int(match_id_str)
                
                raw_events_df = pd.read_pickle(file_path)
                processed_data = fetch_match_data(match_id, raw_events=raw_events_df)
                
                if processed_data['passes'].empty:
                    continue

                processed_data['passes'] = encode_features(processed_data['passes'])
                windows = get_rolling_windows(processed_data, match_id, self.window_size, self.stride)
                
                for window in windows:
                    graph = build_graph_from_window(window)
                    
                    if graph.x.shape[0] == 12:
                        data_list.append(graph)
                        
            except Exception as e:
                continue

        print(f"\n[Dataset] Collating {len(data_list)} graphs...")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Dataset] Success! Saved to {self.processed_paths[0]}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    
    RAW_DIR = "./data/raw_events" 
    ROOT_DIR = "./data_v3"        
    NAME = "offline_mix_v4"       # v4: balanced offensive style + def posture threshold fix
    
    print(f"--- STARTING OFFLINE DATASET BUILD ({NAME}) ---")
    
    dataset = TacticalDataset(
        root=ROOT_DIR, 
        raw_dir=RAW_DIR, 
        dataset_name=NAME,
        window_size=5, 
        stride=1,
        max_matches=230
    )
    
    print(f"\nDataset Ready!")
    print(f"Total Graphs: {len(dataset)}")
    print(f"Node Features: {dataset[0].x.shape}")
    print(f"Reg Targets (y): {dataset[0].y.shape}")
    print(f"Cls Targets (y_cls): {dataset[0].y_cls.shape}")
    print(f"Saved at: {dataset.processed_paths[0]}")