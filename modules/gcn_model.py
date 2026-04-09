"""
gcn_model.py
------------
Graph Convolutional Network (GCN) for disease-state classification.

Architecture
------------
  Input  -> GCNConv(hidden) -> ReLU -> Dropout
         -> GCNConv(hidden) -> ReLU -> Dropout
         -> Linear(n_classes)

Node features : per-cell gene expression vector (one node per cell,
                features = HVG expression values)
Graph         : gene co-expression graph shared across all cells
                (same edge_index for every sample in the batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """
    Two-layer GCN followed by a linear classifier.

    Parameters
    ----------
    in_channels  : number of input features (= n_HVGs)
    hidden_dim   : size of hidden GCN layers
    n_classes    : number of disease-state classes
    dropout      : dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (n_nodes, in_channels) node feature matrix
        edge_index : (2, n_edges) graph connectivity

        Returns
        -------
        logits : (n_nodes, n_classes)
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification head
        logits = self.classifier(x)
        return logits
