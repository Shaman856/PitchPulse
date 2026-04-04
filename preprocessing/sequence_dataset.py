# preprocessing/sequence_dataset.py
"""
SequenceTacticalDataset: wraps the existing per-window dataset and
groups consecutive windows into (input_sequence, target) pairs.

Why a separate file instead of modifying dataset.py:
    dataset.py still builds the per-window graphs (unchanged).
    This file adds a lightweight indexing layer on top.
    If you want to go back to same-window mode, you just use dataset.py directly.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class SequenceTacticalDataset(Dataset):
    """
    Wraps TacticalDataset to produce (sequence_of_graphs, target_graph) pairs.

    For each valid position in a match, creates:
        Input:  graphs at windows [t-seq_len, ..., t-1]  (seq_len graphs)
        Target: labels from window [t]                    (next window)

    Validity constraints:
        - All seq_len+1 windows must belong to the same match AND same team
        - Windows must be consecutive (window_id increases by 1 each step)
        - The match must have at least seq_len+1 windows for that team

    This ensures the LSTM is never fed graphs from different matches or teams,
    which would be meaningless (Croatia window 5 → France window 6 is nonsense).
    """

    def __init__(self, base_dataset, seq_len=5):
        """
        Args:
            base_dataset: A TacticalDataset (or Subset) of per-window graphs
            seq_len:      Number of input windows per sequence (default 5)
        """
        self.base = base_dataset
        self.seq_len = seq_len

        # Build an index of all valid sequences
        # Each entry is a list of seq_len+1 indices into base_dataset
        # where index[:-1] are inputs and index[-1] is the prediction target
        self.sequences = self._build_sequence_index()

        print(f"[SequenceDataset] {len(self.sequences)} valid sequences "
              f"from {len(self.base)} windows (seq_len={seq_len})")

    def _build_sequence_index(self):
        """
        Scans all windows and groups them into valid consecutive sequences.

        Strategy:
            1. Group window indices by (match_id, team_name)
            2. Within each group, sort by window_id
            3. Find all runs of consecutive window_ids (gap=1)
            4. Extract all length-(seq_len+1) sliding windows from each run
        """
        # Step 1: group by match + team, collecting (window_id, dataset_idx)
        from collections import defaultdict
        groups = defaultdict(list)

        for idx in range(len(self.base)):
            graph = self.base[idx]
            # Use match_id and team_name as the grouping key
            key = (int(graph.match_id), str(graph.team_name))
            window_id = int(graph.window_id)
            groups[key].append((window_id, idx))

        sequences = []

        for key, entries in groups.items():
            # Step 2: sort by window_id within each (match, team) group
            entries.sort(key=lambda x: x[0])

            window_ids = [e[0] for e in entries]
            dataset_indices = [e[1] for e in entries]

            # Step 3: find runs of consecutive window_ids
            # A run breaks when window_id[i+1] != window_id[i] + 1
            run_start = 0
            for i in range(1, len(window_ids) + 1):
                # Check if this is the end of a consecutive run
                is_end = (i == len(window_ids) or
                          window_ids[i] != window_ids[i - 1] + 1)

                if is_end:
                    run = dataset_indices[run_start:i]
                    run_len = len(run)

                    # Step 4: extract sliding windows of length seq_len+1
                    # Need seq_len inputs + 1 target = seq_len+1 consecutive windows
                    for start in range(run_len - self.seq_len):
                        # input_indices: the seq_len windows before the target
                        input_indices = run[start: start + self.seq_len]
                        # target_index: the window immediately after the inputs
                        target_index = run[start + self.seq_len]
                        sequences.append(input_indices + [target_index])

                    run_start = i

        return sequences

    def __len__(self):
        # Number of valid (sequence, target) pairs
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns a tuple:
            (input_graphs, target_graph)

        input_graphs: list of seq_len Data objects (the input sequence)
        target_graph: one Data object whose .y and .y_cls are the prediction targets

        Note: we return individual Data objects here, not batched.
        The custom collate_fn handles batching across the batch dimension.
        """
        indices = self.sequences[idx]

        # Collect input sequence (all but last index)
        input_graphs = [self.base[i] for i in indices[:-1]]

        # Target is the last graph — we only need its labels (.y and .y_cls)
        target_graph = self.base[indices[-1]]

        return input_graphs, target_graph


def sequence_collate_fn(batch):
    """
    Custom collate function for SequenceTacticalDataset.

    Problem: each item is (list_of_seq_len_graphs, target_graph).
    PyG's default collate doesn't know how to batch a list-of-lists-of-graphs.

    Solution:
        - Transpose: group all timestep-0 graphs together, all timestep-1 together, etc.
        - Batch each timestep group into one large PyG Batch object
        - Batch all target graphs into one Batch object

    Args:
        batch: list of (input_graphs, target_graph) tuples, one per dataset item

    Returns:
        (batched_sequences, batched_targets)
        batched_sequences: list of seq_len Batch objects
        batched_targets:   one Batch object containing all target labels
    """
    # Separate inputs from targets
    # batch_inputs[i] = list of seq_len graphs for the i-th sample
    # batch_targets[i] = target graph for the i-th sample
    batch_inputs, batch_targets = zip(*batch)

    seq_len = len(batch_inputs[0])

    # Transpose: batched_sequences[t] = all t-th timestep graphs across the batch
    # Before: batch_inputs[sample][timestep]
    # After:  batched_sequences[timestep] = Batch of all samples at that timestep
    batched_sequences = []
    for t in range(seq_len):
        # Collect the t-th graph from every sample in this batch
        graphs_at_t = [batch_inputs[sample_idx][t] for sample_idx in range(len(batch_inputs))]
        # Batch them together into a single PyG Batch object
        batched_sequences.append(Batch.from_data_list(graphs_at_t))

    # Batch all target graphs together
    batched_targets = Batch.from_data_list(list(batch_targets))

    return batched_sequences, batched_targets