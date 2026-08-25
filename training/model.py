"""Multi-modal trajectory prediction network.

The output shape is fixed by the official submission format the evaluator
already reads: six modes, sixteen points each, plus a confidence per mode.

Structure follows the vectorized-encoder family. The target agent forms a single
query that attends to context tokens built from neighbouring agents and nearby
lane geometry. Cross-attention from one query is far cheaper than self-attention
over every token, which matters on a free-tier GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

NUM_MODES = 6
FUTURE_STEPS = 16
POSITION_SCALE = 20.0  # metres per unit, keeps inputs near unit variance


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[index + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class CrossBlock(nn.Module):
    """One round of the query attending to context, then a feed-forward pass."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_feedforward = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor, context_padding: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.norm_attention(query), context, context,
            key_padding_mask=context_padding, need_weights=False,
        )
        query = query + self.dropout(attended)
        return query + self.dropout(self.feedforward(self.norm_feedforward(query)))


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 8,
        blocks: int = 3,
        map_groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.map_groups = map_groups

        # Recurrent encoders capture the shape of a history, not just its endpoints.
        self.target_encoder = nn.GRU(7, dim, batch_first=True)
        self.neighbor_encoder = nn.GRU(10, dim, batch_first=True)
        self.map_encoder = _mlp([5, dim, dim])

        # The target's own class. Neighbours carry a type indicator, but without
        # this the network must infer whether it is predicting a car or a
        # pedestrian from motion alone, and defaults to the majority class.
        self.agent_type = nn.Embedding(4, dim)
        self.context_kind = nn.Embedding(2, dim)  # neighbour vs map token

        self.blocks = nn.ModuleList(
            CrossBlock(dim, heads, dropout) for _ in range(blocks)
        )
        self.head = _mlp([dim, dim * 2, NUM_MODES * (FUTURE_STEPS * 2 + 1)])

    def forward(
        self,
        target_history: Tensor,   # (B, 11, 7)
        neighbors: Tensor,        # (B, N, 11, 10)
        map_points: Tensor,       # (B, P, 5)
        object_type: Tensor | None = None,  # (B,) integer class code
    ) -> tuple[Tensor, Tensor]:
        batch, num_neighbors = neighbors.shape[0], neighbors.shape[1]

        target = target_history.clone()
        target[..., :2] /= POSITION_SCALE
        _, target_state = self.target_encoder(target)
        query = target_state[-1].unsqueeze(1)  # (B, 1, dim)
        if object_type is not None:
            query = query + self.agent_type(object_type).unsqueeze(1)

        flat = neighbors.reshape(batch * num_neighbors, neighbors.shape[2], -1).clone()
        flat[..., :2] /= POSITION_SCALE
        _, neighbor_state = self.neighbor_encoder(flat)
        neighbor_tokens = neighbor_state[-1].reshape(batch, num_neighbors, -1)
        # A neighbour is real if it was valid at any history step.
        neighbor_present = neighbors[..., 6].sum(dim=2) > 0

        points = map_points.clone()
        points[..., :2] /= POSITION_SCALE
        encoded = self.map_encoder(points)                       # (B, P, dim)
        grouped = encoded.reshape(batch, self.map_groups, -1, encoded.shape[-1])
        map_tokens = grouped.amax(dim=2)                         # PointNet-style pooling
        map_present = (
            map_points[..., 4].reshape(batch, self.map_groups, -1).sum(dim=2) > 0
        )

        kinds = self.context_kind(
            torch.cat([
                torch.zeros(num_neighbors, dtype=torch.long, device=neighbors.device),
                torch.ones(self.map_groups, dtype=torch.long, device=neighbors.device),
            ])
        )
        context = torch.cat([neighbor_tokens, map_tokens], dim=1) + kinds
        present = torch.cat([neighbor_present, map_present], dim=1)
        # An all-padded row makes attention produce NaN, so keep one slot open.
        padding = ~present
        padding[:, 0] = False

        for block in self.blocks:
            query = block(query, context, padding)

        raw = self.head(query.squeeze(1))
        modes = raw[:, : NUM_MODES * FUTURE_STEPS * 2]
        logits = raw[:, NUM_MODES * FUTURE_STEPS * 2 :]
        trajectories = modes.reshape(-1, NUM_MODES, FUTURE_STEPS, 2) * POSITION_SCALE
        return trajectories, logits


def multimodal_loss(
    trajectories: Tensor,   # (B, M, T, 2)
    logits: Tensor,         # (B, M)
    future: Tensor,         # (B, T, 2)
    future_mask: Tensor,    # (B, T)
) -> tuple[Tensor, Tensor, Tensor]:
    """Winner-takes-all regression plus classification of the winning mode.

    Averaging every mode toward the single observed future would collapse them
    onto one another and destroy the multi-modality the format exists to
    express. Only the closest mode is regressed; the classifier learns which
    mode that was.

    Roughly half of all agents leave the scene before the horizon ends, so the
    mask decides which steps count. Ignoring it would train the model to steer
    vanished agents toward the origin.
    """

    mask = future_mask.unsqueeze(1)                       # (B, 1, T)
    error = torch.linalg.vector_norm(
        trajectories - future.unsqueeze(1), dim=-1
    )                                                     # (B, M, T)
    counts = mask.sum(dim=-1).clamp(min=1.0)
    average_error = (error * mask).sum(dim=-1) / counts    # (B, M)
    best = average_error.argmin(dim=1)                     # (B,)

    rows = torch.arange(trajectories.shape[0], device=trajectories.device)
    chosen = trajectories[rows, best]                      # (B, T, 2)
    regression = nn.functional.smooth_l1_loss(
        chosen, future, reduction="none"
    ).sum(dim=-1)
    regression = (regression * future_mask).sum() / future_mask.sum().clamp(min=1.0)

    classification = nn.functional.cross_entropy(logits, best)
    return regression + classification, regression.detach(), classification.detach()
