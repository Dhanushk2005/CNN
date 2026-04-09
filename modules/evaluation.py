"""
evaluation.py
-------------
Evaluates the trained GCN on the test split.
Reports: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch_geometric.data import Data

from modules.gcn_model import GCNClassifier


def evaluate(
    model: GCNClassifier,
    data: Data,
    class_names: list[str] | None = None,
    save_path: str = "outputs/confusion_matrix.png",
) -> dict:
    """
    Run evaluation on the test mask and print a full report.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        true  = data.y[data.test_mask].cpu().numpy()

    if class_names is None:
        class_names = [str(c) for c in np.unique(true)]

    print("\n[Evaluation] Classification Report:")
    print(classification_report(true, preds, target_names=class_names))

    metrics = {
        "accuracy":  accuracy_score(true, preds),
        "precision": precision_score(true, preds, average="weighted", zero_division=0),
        "recall":    recall_score(true, preds, average="weighted", zero_division=0),
        "f1":        f1_score(true, preds, average="weighted", zero_division=0),
    }

    # --- Confusion matrix plot ---
    cm = confusion_matrix(true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")

    return metrics
