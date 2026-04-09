# GCN for Single-Cell Genomic Regulatory Networks

Graph Convolutional Networks applied to scRNA-seq data for Gene Regulatory
Network modelling and disease-state classification.

---

## Project Structure

```
scgrn_gcn/
├── data/
│   └── simulate_data.py        # Synthetic scRNA-seq dataset generator
├── modules/
│   ├── preprocessing.py        # QC, normalisation, HVG selection (Scanpy)
│   ├── graph_construction.py   # Gene co-expression graph (adjacency matrix)
│   ├── gcn_model.py            # GCN architecture (PyTorch Geometric)
│   ├── training.py             # Training loop (Adam + CrossEntropyLoss)
│   ├── evaluation.py           # Metrics + confusion matrix
│   ├── visualization.py        # Network plots, driver genes, training curves
│   └── prediction.py           # Inference on new cells
├── outputs/                    # Auto-created; all plots + model weights saved here
├── main.py                     # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install PyTorch first (CPU version shown; adjust for CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
cd scgrn_gcn
python main.py
```

The pipeline runs all 7 steps automatically and prints progress to the console.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data/simulate_data.py` | Generate 300 cells × 500 genes synthetic scRNA-seq data |
| 2 | `modules/preprocessing.py` | Filter, normalise, select 200 highly variable genes |
| 3 | `modules/graph_construction.py` | Build gene co-expression graph (Pearson r ≥ 0.3) |
| 4 | `modules/training.py` | Train 2-layer GCN (100 epochs, Adam, CrossEntropyLoss) |
| 5 | `modules/evaluation.py` | Accuracy / Precision / Recall / F1 + confusion matrix |
| 6 | `modules/visualization.py` | Training curves, gene network, driver gene bar chart |
| 7 | `modules/prediction.py` | Predict disease state for 5 new simulated cells |

---

## Outputs

All files are written to `outputs/`:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Heatmap of true vs predicted labels |
| `training_curves.png` | Loss and validation accuracy per epoch |
| `gene_network.png` | Spring-layout gene co-expression network |
| `driver_genes.png` | Bar chart of top-20 driver genes by network degree |
| `gcn_model.pt` | Saved model weights (PyTorch state dict) |

---

## Example Console Output

```
STEP 1: Data Simulation
AnnData object with n_obs × n_vars = 300 × 500

STEP 2: Preprocessing
  After QC filter: 300 cells, 498 genes
  Highly variable genes selected: 200

STEP 3: Graph Construction
  Nodes: 200  |  Edges: 3842  (threshold=0.3)

STEP 4: Training
  Epoch   1/100  loss=0.7124  val_acc=0.5167
  Epoch  10/100  loss=0.5831  val_acc=0.7333
  ...
  Epoch 100/100  loss=0.2104  val_acc=0.9167

STEP 5: Evaluation
  Test Accuracy : 0.9167
  Precision     : 0.9189
  Recall        : 0.9167
  F1-score      : 0.9164

STEP 7: Prediction on New Cells
  NewCell_000  →  Class_1  (confidence: 0.7823)
  NewCell_001  →  Class_0  (confidence: 0.8541)
  ...
```

---

## Key Concepts

- **GCN (Graph Convolutional Network)**: Learns node embeddings by aggregating
  features from neighbouring nodes — here, cells sharing similar expression profiles.
- **Gene Regulatory Network (GRN)**: Modelled as a graph where nodes are genes
  and edges represent significant co-expression (|Pearson r| ≥ threshold).
- **Driver Genes**: High-degree nodes in the GRN — genes with many co-expression
  partners, likely regulatory hubs.
- **Disease-State Classification**: Each cell is classified into a disease state
  based on its GCN-learned embedding.

---

## Extending to Real Data

Replace the simulation step with your own AnnData file:

```python
import scanpy as sc
adata = sc.read_h5ad("your_data.h5ad")
# Make sure adata.obs['label'] contains integer class labels
```

Public datasets to try:
- [10x Genomics PBMC 3k](https://www.10xgenomics.com/datasets)
- [Human Cell Atlas](https://www.humancellatlas.org/)
- [GEO scRNA-seq datasets](https://www.ncbi.nlm.nih.gov/geo/)
# CNN
