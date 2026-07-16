# GPS-GNN
The code of GPS-GNN

This repository is the official implementation of the paper **"A Group Semantic-Guided Self-Supervised Graph Attention Network for Plant Spatial Transcriptomics Analysis"**. GPS-GNN is a graph neural network (GNN) based framework that integrates spatial coordinates and gene expression profiles for downstream tasks such as cell-type identification and spatial domain detection in spatial transcriptomics data. The code supports both single‑dataset training/evaluation and batch experiments.

## Key Features

- Built on **PyTorch Geometric**, leveraging graph representations of spatial proximity between cells/spots.
- Multi‑modal fusion of **spatial coordinates**, **gene expression**, **cell‑level features**, and **gene‑level features**.
- Two execution modes:
  - `GPS_GNN_main.py` – single run.
  - `GPS_GNN_main_batch.py` – batch processing for multiple datasets or hyperparameter grids, facilitating cross‑validation and comparative studies.
- Data are pre‑processed into `.npy` format and directly loaded as `torch.Tensor` and `torch_geometric.data.Data` objects.

## Data Preparation

### Source of Raw Data

The data used in this work are from publicly available spatial transcriptomics datasets (refer to the manuscript for citations). Please download the raw data following the instructions provided in the original publications.

### Preprocessing Steps

1. Organise the raw data (gene expression matrix, spatial coordinates, cell/spot metadata, etc.) into the following NumPy arrays and save them as `.npy` files:
   - `X_tensor.npy` – gene expression tensor, shape `(P, N, G)`  
     - `P`: number of tissue sections (or batches)  
     - `N`: number of cells/spots per section (assumed constant across sections)  
     - `G`: number of genes  
   - `P_tensor.npy` – spatial coordinates, shape `(P, N, 2)` (x, y)  
   - `X_cell_tensor.npy` – cell‑level features, shape `(P, N, F)`, where `F` is the feature dimension (e.g., batch ID, cell‑cycle scores, etc.)  
   - `X_gene.npy` – gene‑level features, shape `(G, Fg)`, where `Fg` is the gene‑feature dimension (e.g., GO categories, pathway information)  
   - `cell_patch.npy` – tissue patch labels for each cell, shape `(P, N)` (or as needed by your experiment)

2. Place all these files in a single data directory (e.g., `./data/`).  
   The loading function (located in `utils/data_loader.py` or within the main scripts) reads these files sequentially and automatically constructs the spatial graph (kNN or radius graph based on coordinates).

> **Note**: If your data dimensions differ from the above, adjust the loading logic in the main scripts accordingly. The current implementation assumes constant `N` across sections. For variable‑sized sections, consider padding or aligning during preprocessing.

## Environment Dependencies

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- NumPy
- Pandas
- Scikit‑learn
- Matplotlib (for visualisation)
- tqdm (progress bars)

We recommend using conda to create a virtual environment:

```bash
conda create -n gpsgnn python=3.9
conda activate gpsgnn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   # example for CUDA 11.8
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu118.html
pip install numpy pandas scikit-learn matplotlib tqdm
