"""
main.py
-------
End-to-end pipeline for GCN-based disease-state classification
from single-cell RNA-seq data.

Run:
    python main.py
"""

import os
import numpy as np
import torch

# ── Ensure output directory exists ──────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ── 1. Simulate / load data ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Data Simulation")
print("=" * 60)
from data.simulate_data import simulate_scrna_data
adata = simulate_scrna_data(n_cells=300, n_genes=500, n_classes=2)
print(adata)

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)
from modules.preprocessing import preprocess
adata = preprocess(adata, n_top_genes=200)

# ── 3. Graph construction ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Graph Construction")
print("=" * 60)
from modules.graph_construction import build_gene_graph
adj_matrix, gene_edge_index, gene_names = build_gene_graph(adata, corr_threshold=0.3)

# ── 4. Build cell graph & train ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)
from modules.training import build_cell_graph, train_model

if hasattr(adata.X, "toarray"):
    X = adata.X.toarray()
else:
    X = np.array(adata.X)

labels = adata.obs["label"].values.astype(int)
n_classes = len(np.unique(labels))
class_names = [f"Class_{i}" for i in range(n_classes)]

data = build_cell_graph(X, labels, gene_edge_index)

model, train_losses, val_accs = train_model(
    data,
    n_classes=n_classes,
    hidden_dim=64,
    epochs=100,
    lr=1e-3,
)

# ── 5. Evaluation ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)
from modules.evaluation import evaluate
metrics = evaluate(model, data, class_names=class_names)
print(f"  Test Accuracy : {metrics['accuracy']:.4f}")
print(f"  Precision     : {metrics['precision']:.4f}")
print(f"  Recall        : {metrics['recall']:.4f}")
print(f"  F1-score      : {metrics['f1']:.4f}")

# ── 6. Visualisation ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Visualization")
print("=" * 60)
from modules.visualization import (
    plot_training_curves,
    plot_gene_network,
    plot_driver_genes,
)
plot_training_curves(train_losses, val_accs)
driver_genes = plot_gene_network(adj_matrix, gene_names, top_n=50)
plot_driver_genes(adj_matrix, gene_names, top_n=20)

# ── 7. Prediction on new (simulated) cells ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Prediction on New Cells")
print("=" * 60)
from modules.prediction import predict_new_cells

rng = np.random.default_rng(99)
new_X = rng.random((5, X.shape[1])).astype(np.float32)

results = predict_new_cells(model, new_X, gene_edge_index, class_names=class_names)
for r in results:
    print(f"  {r['cell']}  →  {r['predicted_class']}  (confidence: {r['confidence']:.4f})")

# ── Save model ───────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "outputs/gcn_model.pt")
print("\nModel weights saved → outputs/gcn_model.pt")

print("\n" + "=" * 60)
print("Pipeline complete. All outputs in ./outputs/")
print("=" * 60)
