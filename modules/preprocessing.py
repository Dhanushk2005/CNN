"""
preprocessing.py
----------------
Handles scRNA-seq data preprocessing using Scanpy:
  1. Quality-control filtering
  2. Normalization (library-size + log1p)
  3. Highly variable gene (HVG) selection
"""

import numpy as np
import scanpy as sc
import anndata as ad


def preprocess(
    adata: ad.AnnData,
    min_genes: int = 20,
    min_cells: int = 5,
    n_top_genes: int = 200,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    adata       : raw AnnData (cells x genes)
    min_genes   : minimum genes expressed per cell
    min_cells   : minimum cells a gene must appear in
    n_top_genes : number of highly variable genes to keep
    target_sum  : library-size normalisation target

    Returns
    -------
    adata : preprocessed AnnData with HVG subset
    """
    print("[Preprocessing] Starting pipeline...")

    # --- 1. Basic QC filtering ---
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  After QC filter: {adata.n_obs} cells, {adata.n_vars} genes")

    # --- 2. Normalise to fixed library size then log-transform ---
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Store normalised values in a layer for reference
    adata.layers["log_norm"] = adata.X.copy()

    # --- 3. Highly variable gene selection ---
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  Highly variable genes selected: {n_hvg}")

    # Subset to HVGs only
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  Final shape: {adata.shape}")

    return adata
