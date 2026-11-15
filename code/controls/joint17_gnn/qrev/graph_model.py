"""Shared distance-based message passing for variable-size molecular graphs."""
import torch
from torch import nn

from .graphs import DESCRIPTOR_DIM, MAX_ATOMS, GRAPH_DIM


class Interaction(nn.Module):
    def __init__(self, width, n_rbf, dropout):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(n_rbf, width), nn.GELU(), nn.Linear(width, width))
        self.source = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, radial, real_nodes, edges):
        # Axis 0 is the molecule batch. The reduction on axis 2 only sums neighbors in one molecule.
        message = self.filter(radial) * self.source(h)[:, None, :, :]
        aggregate = (message * edges[..., None]).sum(dim=2) / edges.sum(dim=2, keepdim=True).clamp_min(1)
        return self.norm(h + self.update(aggregate)) * real_nodes[..., None]


class MolecularGNN(nn.Module):
    """Legacy descriptors enter before message passing and again at readout."""
    def __init__(self, input_dim, output_dim, width=64, interactions=3, n_rbf=16,
                 head_hidden=174, dropout=0.1):
        super().__init__()
        if input_dim != GRAPH_DIM or n_rbf < 2 or interactions < 1:
            raise ValueError("Invalid joint molecular graph model dimensions")
        self.atoms = nn.Embedding(9, width, padding_idx=0)
        self.context = nn.Linear(DESCRIPTOR_DIM, width)
        self.register_buffer("centers", torch.linspace(0.0, 8.0, n_rbf))
        self.register_buffer("not_self", ~torch.eye(MAX_ATOMS, dtype=torch.bool))
        self.inv_width2 = ((n_rbf-1)/8.0)**2
        self.blocks = nn.ModuleList([Interaction(width, n_rbf, dropout) for _ in range(interactions)])
        self.head = nn.Sequential(nn.Linear(width+DESCRIPTOR_DIM, head_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, output_dim))
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        desc = x[:, :DESCRIPTOR_DIM]
        z = x[:, DESCRIPTOR_DIM:DESCRIPTOR_DIM+MAX_ATOMS].long()
        d = x[:, DESCRIPTOR_DIM+MAX_ATOMS:].reshape(-1, MAX_ATOMS, MAX_ATOMS)
        nodes = z.ne(0)
        edges = nodes[:, :, None] & nodes[:, None, :] & self.not_self
        radial = torch.exp(-self.inv_width2*(d[..., None] - self.centers).square())
        h = (self.atoms(z) + self.context(desc)[:, None, :]) * nodes[..., None]
        for block in self.blocks:
            h = block(h, radial, nodes, edges)
        pooled = h.sum(dim=1) / nodes.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(torch.cat([pooled, desc], dim=1))
