================================================================================
INSTALLATION
================================================================================

1. Clone the repository (if applicable) or navigate to your project folder.

2. Install dependencies using the provided requirements file:
   
   Command:
   pip install -r requirements.txt

   NOTE: This project requires 'torch' and 'torch_geometric'. If you have issues 
   installing them via pip, visit the PyTorch Geometric Installation Guide for 
   your specific OS/CUDA version:
   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html


================================================================================
USAGE GUIDE (RUNNING FROM SCRATCH)
================================================================================

Follow these steps in strict order:


STEP 1: DOWNLOAD RAW DATA
--------------------------------------------------------------------------------
Fetch match event data from the StatsBomb Open Data API.

   Command:
   python preprocessing/download_raw.py

   Output: Saves .pkl files to "data/raw_events/"


STEP 2: BUILD THE DATASET
--------------------------------------------------------------------------------
Process the raw events into a Graph Dataset.
*** IMPORTANT: You must run this as a module (-m) because of relative imports ***

   Command:
   python -m preprocessing.dataset

   Output: Creates "data_v3/processed/tactical_offline_mix_w5_s1.pt"


STEP 3: TRAIN THE MODEL
--------------------------------------------------------------------------------
Train the Graph Attention Network (GAT) to predict tactical metrics.

   Command:
   python train.py

   Output: Saves "best_model.pth" and generates "training_curve.png"


STEP 4: RUN INFERENCE
--------------------------------------------------------------------------------
Evaluate the trained model on a test set and visualize predictions.

   Command:
   python inference.py

   Output: Prints error metrics to console and saves "inference_analysis.png"