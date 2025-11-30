import torch
import torch.nn as nn
import math
from einops import rearrange
from typing import Optional, List


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention optimized with torch.einsum.
    """

    def __init__(self, dim: int, context_dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)

        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        return self.to_out(out) + x


class ResBlockWithTime(nn.Module):
    """ ResNet Block with Time Embedding Injection """

    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, 2, 1)  # Stride 2 reduces spatial dim

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.conv(self.up(x))


class DiffusionUNet(nn.Module):
    """
    Correctly structured U-Net with Down/Up sampling logic fixed.
    """

    def __init__(self, dim: int = 64, channels: int = 3, cond_dim: int = 2048):
        super().__init__()

        # Initial Conv
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)

        # Time Embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, time_dim),
        )

        # --- Down 1: 64 -> 128 ---
        self.down1 = nn.ModuleList([
            ResBlockWithTime(dim, dim, time_dim),
            ResBlockWithTime(dim, dim, time_dim),
            Downsample(dim)  # Output channels: dim, Size: /2
        ])

        # --- Down 2: 64 -> 128 ---
        # Note: Input is dim (64), we expand to dim*2 (128)
        self.down2 = nn.ModuleList([
            ResBlockWithTime(dim, dim * 2, time_dim),
            ResBlockWithTime(dim * 2, dim * 2, time_dim),
            CrossAttention(dim * 2, cond_dim),
            Downsample(dim * 2)  # Output channels: dim*2, Size: /4
        ])

        # --- Mid: 128 -> 256 -> 128 ---
        mid_dim = dim * 4  # Expand to 256
        self.mid_block1 = ResBlockWithTime(dim * 2, mid_dim, time_dim)
        self.mid_attn = CrossAttention(mid_dim, cond_dim)
        self.mid_block2 = ResBlockWithTime(mid_dim, mid_dim, time_dim)

        # --- Up 2: 256 -> 128 ---
        # Input: mid_dim (256). Skip from Down 2: dim*2 (128).
        self.up2 = nn.ModuleList([
            Upsample(mid_dim),  # 256 -> 256 (Size * 2)
            ResBlockWithTime(mid_dim + dim * 2, dim * 2, time_dim),  # Cat(256+128) -> 128
            ResBlockWithTime(dim * 2, dim * 2, time_dim),
            CrossAttention(dim * 2, cond_dim)
        ])

        # --- Up 1: 128 -> 64 ---
        # Input: dim*2 (128). Skip from Down 1: dim (64).
        self.up1 = nn.ModuleList([
            Upsample(dim * 2),  # 128 -> 128 (Size * 2)
            ResBlockWithTime(dim * 2 + dim, dim, time_dim),  # Cat(128+64) -> 64
            ResBlockWithTime(dim, dim, time_dim)
        ])

        # Final: 64 -> 3
        # Input: dim (64). Skip from Init: dim (64).
        self.final_res = ResBlockWithTime(dim + dim, dim, time_dim)  # Cat(64+64) -> 64
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, cond=None):
        # Time Embedding
        t = self.time_mlp(time.float().unsqueeze(-1))

        # Condition Z pooling
        if cond is not None:
            cond_vec = rearrange(cond, 'b c h w -> b c (h w)').mean(dim=-1).unsqueeze(1)

        # 1. Init
        x = self.init_conv(x)
        r_init = x.clone()  # Save for final skip

        # 2. Down 1
        for block in self.down1:
            if isinstance(block, Downsample):
                r_down1 = x.clone()  # [Fix] Capture BEFORE downsampling
                x = block(x)
            else:
                x = block(x, t)

        # 3. Down 2
        for block in self.down2:
            if isinstance(block, Downsample):
                r_down2 = x.clone()  # [Fix] Capture BEFORE downsampling
                x = block(x)
            elif isinstance(block, CrossAttention):
                x = block(x, cond_vec)
            else:
                x = block(x, t)

        # 4. Mid
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond_vec)
        x = self.mid_block2(x, t)

        # 5. Up 2 (Concatenate with Down 2 output)
        x = self.up2[0](x)  # Upsample
        x = torch.cat((x, r_down2), dim=1)  # Concat with r_down2
        x = self.up2[1](x, t)
        x = self.up2[2](x, t)
        x = self.up2[3](x, cond_vec)

        # 6. Up 1 (Concatenate with Down 1 output)
        x = self.up1[0](x)  # Upsample
        x = torch.cat((x, r_down1), dim=1)  # Concat with r_down1
        x = self.up1[1](x, t)
        x = self.up1[2](x, t)

        # Final
        x = torch.cat((x, r_init), dim=1)
        x = self.final_res(x, t)

        return self.final_conv(x)


class GenerativeTextingNet(nn.Module):
    def __init__(self, encoder: nn.Module, cfg):
        super().__init__()
        self.encoder = encoder

        # Access config via correct path
        txt_cfg = cfg.model.texting_net

        self.unet = DiffusionUNet(
            dim=txt_cfg.unet_dim,
            channels=3,
            cond_dim=2048  # Matches encoder output dim
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, clean_img: torch.Tensor):
        with torch.no_grad():
            cond, _ = self.encoder(clean_img)
        return self.unet(x_t, t, cond), cond