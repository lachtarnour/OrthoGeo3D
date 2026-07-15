from __future__ import annotations

import torch
from torch import nn


def knn(x: torch.Tensor, k: int, chunk_size: int = 2048) -> torch.Tensor:
    """Return exact k-nearest-neighbor indices without allocating a full NxN matrix."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, C, N], got {tuple(x.shape)}")

    num_points = x.shape[-1]
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    k = min(k, num_points)

    indices = []
    with torch.no_grad():
        reference_norm = torch.sum(x**2, dim=1, keepdim=True)
        for start in range(0, num_points, chunk_size):
            query = x[:, :, start : start + chunk_size]
            inner = -2.0 * torch.matmul(query.transpose(2, 1), x)
            query_norm = torch.sum(query**2, dim=1).unsqueeze(-1)
            pairwise_distance = -query_norm - inner - reference_norm
            indices.append(pairwise_distance.topk(k=k, dim=-1)[1])
    return torch.cat(indices, dim=1)


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build EdgeConv features [x_j - x_i, x_i].

    Input:
        x: [B, C, N]

    Output:
        edge_features: [B, 2C, N, k]
    """

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, C, N], got {tuple(x.shape)}")

    batch_size, channels, num_points = x.shape
    if idx is None:
        idx = knn(x, k=k)

    k = idx.shape[-1]
    x_points = x.transpose(2, 1).contiguous()
    x_flat = x_points.reshape(batch_size * num_points, channels)
    batch_offsets = (
        torch.arange(batch_size, device=x.device).view(batch_size, 1, 1) * num_points
    )
    idx_global = idx + batch_offsets
    neighbors = x_flat[idx_global.reshape(-1)]
    neighbors = neighbors.view(batch_size, num_points, k, channels)
    centers = x_points.view(batch_size, num_points, 1, channels).expand(-1, -1, k, -1)
    edge_features = torch.cat([neighbors - centers, centers], dim=-1)
    return edge_features.permute(0, 3, 1, 2).contiguous()


def conv2d_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        group_norm(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    )


def conv1d_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        group_norm(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    )


def group_norm(num_channels: int) -> nn.GroupNorm:
    if num_channels % 8 != 0:
        raise ValueError(
            f"GroupNorm requires channels divisible by 8, got {num_channels}"
        )
    return nn.GroupNorm(num_groups=8, num_channels=num_channels)


class DGCNNSegmentation(nn.Module):
    """Dynamic Graph CNN for point-wise semantic segmentation.

    Input `x` is [B, N, C], output logits are [B, N, num_classes].
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 17,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.k = k
        self.emb_dims = emb_dims

        self.conv1 = conv2d_block(input_channels * 2, 64)
        self.conv2 = conv2d_block(64, 64)
        self.conv3 = conv2d_block(64 * 2, 64)
        self.conv4 = conv2d_block(64, 64)
        self.conv5 = conv2d_block(64 * 2, 64)

        self.global_conv = conv1d_block(64 * 3, emb_dims)
        self.head1 = conv1d_block(emb_dims + 64 * 3, 512)
        self.head2 = conv1d_block(512, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Conv1d(256, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, N, C] or [B, C, N], got {tuple(x.shape)}"
            )
        if x.shape[-1] == self.input_channels:
            x = x.transpose(2, 1).contiguous()
        elif x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input_channels={self.input_channels}, got shape {tuple(x.shape)}"
            )

        num_points = x.shape[-1]

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        local_features = torch.cat((x1, x2, x3), dim=1)
        global_features = self.global_conv(local_features)
        global_features = global_features.max(dim=-1, keepdim=True)[0]
        global_features = global_features.repeat(1, 1, num_points)

        x = torch.cat((global_features, local_features), dim=1)
        x = self.head1(x)
        x = self.head2(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits.transpose(2, 1).contiguous()
