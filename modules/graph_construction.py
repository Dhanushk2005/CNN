"""
graph_construction.py
---------------------
Builds a Gene Regulatory Network (GRN) graph from the expression matrix:
  - Nodes  : genes (HVGs)
  - Edges  : pairs of genes whose Pearson correlation exceeds a threshold
  - Returns: adjacency matrix + PyTorch Geometric edge_index tensor
"""

import numpy as np
import torch
import anndata as ad
from scipy.stats import pearsonr


def build_gene_graph(
    adata: ad.AnnData,
    corr_threshold: float = 0.3,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """
    Compute pairwise gene correlations and build a graph.

    Parameters
    ----------
    adata          : preprocessed AnnData (cells x HVGs)
    corr_threshold : minimum |Pearson r| to add an edge

    Returns
    -------
    adj_matrix  : (n_genes, n_genes) binary adjacency matrix
    edge_index  : (2, n_edges) LongTensor for PyG
    gene_names  : array of gene names (node labels)
    """
    print("[Graph Construction] Computing gene-gene correlations...")

    # Expression matrix: shape (n_cells, n_genes)
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    n_genes = X.shape[1]
    gene_names = np.array(adata.var_names)

    # Correlation matrix via numpy (fast vectorised)
    # np.corrcoef expects (n_features, n_samples)
    corr_matrix = np.corrcoef(X.T)          # (n_genes, n_genes)
    corr_matrix = np.nan_to_num(corr_matrix)  # replace NaN with 0

    # Threshold: keep edges where |r| >= threshold (exclude self-loops)
    adj_matrix = (np.abs(corr_matrix) >= corr_threshold).astype(np.float32)
    np.fill_diagonal(adj_matrix, 0)

    n_edges = int(adj_matrix.sum())
    print(f"  Nodes: {n_genes}  |  Edges: {n_edges}  (threshold={corr_threshold})")

    # Build edge_index for PyTorch Geometric
    src, dst = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    return adj_matrix, edge_index, gene_names
