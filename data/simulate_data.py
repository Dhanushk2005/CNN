"""
simulate_data.py
----------------
Generates a synthetic scRNA-seq dataset for demonstration.
Produces an AnnData object with:
  - 300 cells x 500 genes
  - 2 disease states (0 = healthy, 1 = disease)
  - Biologically-inspired expression patterns
"""

import numpy as np
import pandas as pd
import anndata as ad


def simulate_scrna_data(
    n_cells: int = 300,
    n_genes: int = 500,
    n_classes: int = 2,
    seed: int = 42,
) -> ad.AnnData:
    """
    Simulate a scRNA-seq count matrix with class labels.

    Returns
    -------
    adata : AnnData
        .X  -> raw count matrix (cells x genes)
        .obs['label'] -> integer disease label
    """
    rng = np.random.default_rng(seed)

    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]

    # Assign labels evenly
    labels = np.array([i % n_classes for i in range(n_cells)])

    # Base expression: negative-binomial-like counts
    counts = rng.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Inject class-specific signal into the first 50 genes
    for cls in range(n_classes):
        mask = labels == cls
        # Upregulate a distinct gene block per class
        start = cls * 25
        counts[np.ix_(mask, np.arange(start, start + 25))] += rng.poisson(
            lam=15, size=(mask.sum(), 25)
        )

    obs = pd.DataFrame({"label": labels}, index=cell_names)
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=counts, obs=obs, var=var)
    return adata


if __name__ == "__main__":
    adata = simulate_scrna_data()
    print(adata)
    print("Label distribution:", adata.obs["label"].value_counts().to_dict())
