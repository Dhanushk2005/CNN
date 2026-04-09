"""
training.py
-----------
Trains the GCN model on the scRNA-seq graph dataset.

Strategy
--------
Each cell is treated as a node in a fully-connected "cell graph"
built from the gene co-expression edge_index.  Node features are
the HVG expression values for that cell.  Labels are disease states.

We use a transductive split: all nodes are in the graph, but loss
is computed only on train nodes; evaluation on val/test nodes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from modules.gcn_model import GCNClassifier


def build_cell_graph(
    X: np.ndarray,
    labels: np.ndarray,
    edge_index: torch.Tensor,
) -> Data:
    """
    Build a PyG Data object where nodes = cells.

    Because the gene co-expression graph has gene-level edges, we
    re-use the same edge_index structure but treat each cell as a
    node with its own feature vector.  For a proper cell graph we
    build a k-NN graph on the expression matrix.

    Parameters
    ----------
    X          : (n_cells, n_genes) expression matrix
    labels     : (n_cells,) integer class labels
    edge_index : gene-level edge_index (used for shape reference)

    Returns
    -------
    data : PyG Data with x, y, and train/val/test masks
    """
    n_cells = X.shape[0]

    # Build a simple cell-cell k-NN graph based on correlation
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    # Keep top-k neighbours per cell (k=10)
    k = min(10, n_cells - 1)
    cell_edges_src, cell_edges_dst = [], []
    for i in range(n_cells):
        top_k = np.argsort(sim[i])[-k:]
        for j in top_k:
            cell_edges_src.append(i)
            cell_edges_dst.append(j)

    cell_edge_index = torch.tensor(
        [cell_edges_src, cell_edges_dst], dtype=torch.long
    )

    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Train / val / test split (60 / 20 / 20)
    idx = np.arange(n_cells)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels[train_idx], random_state=42)

    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    val_mask   = torch.zeros(n_cells, dtype=torch.bool)
    test_mask  = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data = Data(
        x=x,
        y=y,
        edge_index=cell_edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data


def train_model(
    data: Data,
    n_classes: int = 2,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    dropout: float = 0.3,
    device: str = "cpu",
) -> tuple[GCNClassifier, list[float], list[float]]:
    """
    Train the GCN and return the model + loss/accuracy histories.

    Returns
    -------
    model        : trained GCNClassifier
    train_losses : list of per-epoch training losses
    val_accs     : list of per-epoch validation accuracies
    """
    print("[Training] Initialising model...")
    data = data.to(device)

    model = GCNClassifier(
        in_channels=data.x.shape[1],
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []

    print(f"  Training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            preds = val_logits[data.val_mask].argmax(dim=1)
            val_acc = (preds == data.y[data.val_mask]).float().mean().item()

        train_losses.append(loss.item())
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    print("[Training] Done.")
    return model, train_losses, val_accs
