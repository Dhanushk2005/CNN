"""
prediction.py
-------------
Predicts disease state for new (unseen) cell expression profiles.

Usage
-----
    from modules.prediction import predict_new_cells
    results = predict_new_cells(model, new_X, edge_index, class_names)
"""

import numpy as np
import torch
from torch_geometric.data import Data

from modules.gcn_model import GCNClassifier


def predict_new_cells(
    model: GCNClassifier,
    new_X: np.ndarray,
    edge_index: torch.Tensor,
    class_names: list[str] | None = None,
    device: str = "cpu",
) -> list[dict]:
    """
    Predict disease state for new cells.

    Parameters
    ----------
    model       : trained GCNClassifier
    new_X       : (n_new_cells, n_genes) expression matrix (same HVGs as training)
    edge_index  : gene co-expression edge_index from training graph
    class_names : optional list of class label strings
    device      : 'cpu' or 'cuda'

    Returns
    -------
    results : list of dicts with keys 'cell', 'predicted_class', 'confidence'
    """
    model.eval()
    model.to(device)

    x = torch.tensor(new_X, dtype=torch.float).to(device)

    # Build a minimal cell graph (self-loops only) for new cells
    n = new_X.shape[0]
    self_loops = torch.arange(n, dtype=torch.long)
    new_edge_index = torch.stack([self_loops, self_loops], dim=0).to(device)

    with torch.no_grad():
        logits = model(x, new_edge_index)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

    if class_names is None:
        class_names = [str(i) for i in range(probs.shape[1])]

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "cell":            f"NewCell_{i:03d}",
            "predicted_class": class_names[pred],
            "confidence":      round(float(prob[pred]), 4),
            "probabilities":   {class_names[j]: round(float(p), 4) for j, p in enumerate(prob)},
        })

    return results
