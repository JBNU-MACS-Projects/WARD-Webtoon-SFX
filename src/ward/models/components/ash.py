import torch
import torch.nn as nn
from typing import Final, Optional
from einops import rearrange, reduce


class ASH(nn.Module):
    """
    ASH (Activation Shaping) Module: ASH-P variant.

    Dynamically prunes the bottom p% of activations in spatial dimensions
    to remove background noise and sharpen stroke features.

    Paper: "ASH: Activation Shaping for Out-of-Distribution Detection"
    """

    def __init__(
            self,
            percentile: int = 90,
            scaling: bool = True,
            eps: float = 1e-6
    ) -> None:
        super().__init__()
        # Final type hint for JIT compatibility
        self.percentile: Final[int] = percentile
        self.scaling: Final[bool] = scaling
        self.eps: Final[float] = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        """
        if self.percentile >= 100:
            return torch.zeros_like(x)
        if self.percentile <= 0:
            return x

        B, C, H, W = x.shape
        n_elements = H * W
        k = int(n_elements * (1.0 - self.percentile / 100.0))

        # 1. Spatial Flatten: (B, C, H, W) -> (B, C, S)
        x_flat = rearrange(x, 'b c h w -> b c (h w)')

        # 2. Dynamic Threshold Calculation (Per Channel)
        # Find the k-th largest value in each channel
        top_k_val, _ = torch.topk(x_flat, k, dim=-1, sorted=False)

        # Threshold shape: (B, C, 1) - Broadcasting ready
        threshold = top_k_val[..., -1].unsqueeze(-1)

        # 3. Hard Pruning (ASH-P)
        # Create binary mask where activation >= threshold
        mask = (x_flat >= threshold).float()
        x_pruned = x_flat * mask

        # 4. Energy Conservation Scaling (ASH-S concept)
        # Prevents signal magnitude collapse in deep networks
        if self.scaling:
            energy_orig = reduce(x_flat, 'b c s -> b c 1', 'sum')
            energy_pruned = reduce(x_pruned, 'b c s -> b c 1', 'sum')

            # Avoid division by zero
            scale_factor = energy_orig / (energy_pruned + self.eps)
            x_pruned = x_pruned * scale_factor

        # 5. Restore Geometry
        return rearrange(x_pruned, 'b c (h w) -> b c h w', h=H, w=W)

    def extra_repr(self) -> str:
        return f"percentile={self.percentile}, scaling={self.scaling}"