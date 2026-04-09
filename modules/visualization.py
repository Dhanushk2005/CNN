"""
visualization.py
----------------
Visualisation utilities:
  1. Training curves (loss + val accuracy)
  2. Gene regulatory network graph (top-N genes by degree)
  3. Driver gene bar chart (highest-degree nodes)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_training_curves(
    train_losses: list[float],
    val_accs: list[float],
    save_path: str = "outputs/training_curves.png",
) -> None:
    """Plot loss and validation accuracy over epochs."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, color="darkorange", linewidth=1.5)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Training curves saved → {save_path}")


def plot_gene_network(
    adj_matrix: np.ndarray,
    gene_names: np.ndarray,
    top_n: int = 50,
    save_path: str = "outputs/gene_network.png",
) -> list[str]:
    """
    Draw the gene co-expression network for the top-N highest-degree genes.

    Returns
    -------
    driver_genes : list of top-10 driver gene names (by degree)
    """
    print("[Visualization] Building gene network graph...")

    # Compute degree for each gene
    degrees = adj_matrix.sum(axis=1)
    top_idx = np.argsort(degrees)[-top_n:]

    # Sub-matrix for top genes
    sub_adj = adj_matrix[np.ix_(top_idx, top_idx)]
    sub_names = gene_names[top_idx]

    G = nx.from_numpy_array(sub_adj)
    mapping = {i: sub_names[i] for i in range(len(sub_names))}
    G = nx.relabel_nodes(G, mapping)

    # Node size proportional to degree
    node_degrees = dict(G.degree())
    node_sizes = [300 + node_degrees[n] * 40 for n in G.nodes()]

    # Colour driver genes (top 10 by degree) differently
    sorted_by_deg = sorted(node_degrees, key=node_degrees.get, reverse=True)
    driver_genes = sorted_by_deg[:10]
    node_colors = [
        "#e74c3c" if n in driver_genes else "#3498db" for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#e74c3c", label="Driver gene (top-10)"),
        Patch(color="#3498db", label="Other gene"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(f"Gene Co-expression Network (top {top_n} genes by degree)", fontsize=13)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gene network saved → {save_path}")
    print(f"  Driver genes (top-10): {driver_genes}")
    return driver_genes


def plot_driver_genes(
    adj_matrix: np.ndarray,
    gene_names: np.ndarray,
    top_n: int = 20,
    save_path: str = "outputs/driver_genes.png",
) -> None:
    """Bar chart of the top-N driver genes ranked by network degree."""
    degrees = adj_matrix.sum(axis=1)
    top_idx = np.argsort(degrees)[-top_n:][::-1]
    top_genes = gene_names[top_idx]
    top_degrees = degrees[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c"] * 10 + ["#3498db"] * (top_n - 10)
    ax.barh(top_genes[::-1], top_degrees[::-1], color=colors[::-1])
    ax.set_xlabel("Network Degree (number of co-expressed partners)")
    ax.set_title(f"Top {top_n} Driver Genes by Network Degree")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Driver gene chart saved → {save_path}")
