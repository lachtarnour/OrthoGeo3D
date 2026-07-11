from __future__ import annotations

import torch
from torch import nn


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return k-nearest-neighbor indices for features shaped [B, C, N]."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, C, N], got {tuple(x.shape)}")

    num_points = x.shape[-1]
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, num_points)

    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]

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

    B, C, N = x.shape

    # 1. Compute nearest-neighbor indices if they are not provided
    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]

    k = idx.shape[-1]

    # 2. Rearrange points from [B, C, N] to [B, N, C]
    x_points = x.transpose(2, 1).contiguous()

    # 3. Flatten all batch points into one table: [B, N, C] -> [B*N, C]
    x_flat = x_points.reshape(B * N, C)

    # 4. Create a batch offset for each sample
    batch_offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N
    # shape: [B, 1, 1]

    # 5. Convert local neighbor indices into global indices
    idx_global = idx + batch_offsets
    # shape: [B, N, k]

    # 6. Flatten global indices for direct indexing
    idx_global_flat = idx_global.reshape(-1)
    # shape: [B*N*k]

    # 7. Gather neighbor features from the flattened point table
    neighbors = x_flat[idx_global_flat]
    # shape: [B*N*k, C]

    # 8. Restore neighbor tensor shape to [B, N, k, C]
    neighbors = neighbors.view(B, N, k, C)

    # 9. Repeat each center point k times to match its neighbors
    centers = x_points.view(B, N, 1, C).expand(-1, -1, k, -1)
    # shape: [B, N, k, C]

    # 10. Build EdgeConv features: [neighbor - center, center]
    edge_features = torch.cat([neighbors - centers, centers], dim=-1)
    # shape: [B, N, k, 2C]

    # 11. Rearrange to PyTorch Conv2d format: [B, channels, N, k]
    edge_features = edge_features.permute(0, 3, 1, 2).contiguous()
    # shape: [B, 2C, N, k]

    return edge_features


def conv2d_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    )


def conv1d_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    )

class DGCNNSegmentation(nn.Module):
    """Dynamic Graph CNN for point-wise semantic segmentation.

    Input `x` is [B, N, C], output logits are [B, N, num_classes].
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 17,
        k:int = 20,
        emb_dims:int = 1024,
        dropout: float = 0.5,
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
            raise ValueError(f"Expected x with shape [B, N, C] or [B, C, N], got {tuple(x.shape)}")
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
